"""Minimal CPU simulation of actual DynamicCache and new checkpoint helpers.

This is an adapter serialization audit, not a model forward or old Phase0 rerun.
Worker source is supplied explicitly as a frozen local text snapshot for audit.
"""
import argparse, ast, copy, hashlib, io, json, pathlib, sys
from types import SimpleNamespace
import torch


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main(source_path, output):
    sys.path.insert(0, '/data/zypan/transformers-qwen35/src')
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    source = pathlib.Path(source_path).read_text()
    tree = ast.parse(source)
    names = {'cpu_tree', 'describe', 'deep_hash'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(torch=torch, copy=copy, hashlib=hashlib, json=json,
              K=SimpleNamespace(tsha=sha_tensor))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), ns)
    config = Qwen3_5TextConfig.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B', local_files_only=True)
    cache = DynamicCache(config=config)
    linear, attention = [], []
    with torch.inference_mode():
        for i, kind in enumerate(config.layer_types):
            if kind == 'linear_attention':
                linear.append(i)
                cache.update_recurrent_state(torch.full((1, 32, 128, 128), (i + 1) / 32, dtype=torch.float32), i)
                cache.update_conv_state(torch.full((1, 8192, 4), (i + 1) / 64, dtype=torch.bfloat16), i)
            else:
                attention.append(i)
                cache.update(torch.full((1, 4, 2, 256), (i + 1) / 64, dtype=torch.bfloat16),
                             torch.full((1, 4, 2, 256), (i + 1) / 128, dtype=torch.bfloat16), i)
    assert len(linear) == 24 and len(attention) == 8
    assert cache.get_seq_length() == 2
    original = ns['deep_hash'](cache)
    cloned = ns['cpu_tree'](cache)
    assert ns['deep_hash'](cloned) == original
    assert cloned.get_seq_length() == 2
    assert all(isinstance(cloned.layers[i].recurrent_states, dict) for i in linear)
    assert all(cloned.layers[i].recurrent_states[0].data_ptr() != cache.layers[i].recurrent_states[0].data_ptr() for i in linear)
    payload = dict(cache=cloned, cursor=2, RNG_CPU=torch.get_rng_state(), metadata=ns['describe'](cache))
    buffer = io.BytesIO(); torch.save(payload, buffer); buffer.seek(0)
    loaded = torch.load(buffer, map_location='cpu', weights_only=False)
    restored = ns['cpu_tree'](loaded['cache'], device='cpu')
    assert loaded['cursor'] == restored.get_seq_length() == 2
    assert ns['deep_hash'](restored) == original
    assert ns['describe'](restored) == loaded['metadata']
    restored.layers[linear[0]].recurrent_states[0].add_(0.5)
    assert ns['deep_hash'](cache) == original
    assert ns['deep_hash'](restored) != original
    for i in attention:
        restored.update(torch.zeros((1, 4, 1, 256), dtype=torch.bfloat16),
                        torch.zeros((1, 4, 1, 256), dtype=torch.bfloat16), i)
    assert restored.get_seq_length() == 3
    assert cache.get_seq_length() == 2
    # Reproduce the inference tensor property that prompted the pre-GPU fix.
    inference_version_raises = False
    try:
        cache.layers[linear[0]].recurrent_states[0]._version
    except RuntimeError as exc:
        inference_version_raises = 'Inference tensors do not track version counter' in str(exc)
    assert inference_version_raises
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', scope='CPU-only actual DynamicCache checkpoint-adapter simulation; no GPU/model forward',
                  worker_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  inspected_function_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
                  actual_cache_class=type(cache).__module__ + '.' + type(cache).__name__,
                  linear_cache_container='dict[0] for recurrent_states and conv_states',
                  linear_layers=linear, attention_layers=attention,
                  checks=['cpu_tree complete tensor and public metadata identity', 'torch.save/load and cpu_tree exact roundtrip',
                          'get_seq_length/cursor exact before and after appended token', 'no shared mutable recurrent storage',
                          'mutating restored cache cannot alter source', 'inference tensor _version inaccessible as expected'],
                  initial_cursor=2, appended_cursor=3, CUDA_initialized=False,
                  GPU_restore='NOT_RUN; actual canary must qualify CUDA transfer and subsequent token identity')
    with pathlib.Path(output).open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.worker_source,a.output)
