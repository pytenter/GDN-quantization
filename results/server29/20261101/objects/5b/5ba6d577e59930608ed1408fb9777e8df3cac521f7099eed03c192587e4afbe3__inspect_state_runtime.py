#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_PATH = '/data/zypan/modelscope_models/Qwen3.5-9B'
OUT = Path('/data/zypan/results/gdn_state_runtime_inspect.json')

def tensor_info(t):
    elems = t.numel(); bytes_ = elems * t.element_size()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device), 'numel': int(elems), 'element_size_bytes': int(t.element_size()), 'memory_bytes': int(bytes_), 'memory_mib': bytes_ / 1024 / 1024}

def iter_state_container(states):
    if states is None:
        return []
    if isinstance(states, dict):
        return list(states.items())
    return list(enumerate(states))

def main():
    import transformers
    print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
    print('transformers', transformers.__version__, os.path.dirname(transformers.__file__))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True).eval()
    print('model_class', model.__class__.__name__)
    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    layer_types = list(getattr(cfg, 'layer_types', []))
    print('gdn_layer_count', sum(1 for x in layer_types if x == 'linear_attention'))
    print('attention_layer_count', sum(1 for x in layer_types if x == 'full_attention'))
    inputs = tokenizer('Inspect recurrent state.', return_tensors='pt')
    first_param = next(model.parameters())
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    print('cache_class', cache.__class__.__name__)
    print('cache_has_layers', hasattr(cache, 'layers'))
    records = []
    for idx, layer in enumerate(getattr(cache, 'layers', [])):
        typ = layer_types[idx] if idx < len(layer_types) else 'unknown'
        rec = {'layer_idx': idx, 'layer_type': typ, 'cache_layer_class': layer.__class__.__name__, 'has_recurrent_states': hasattr(layer, 'recurrent_states'), 'has_conv_states': hasattr(layer, 'conv_states')}
        states = getattr(layer, 'recurrent_states', None)
        items = iter_state_container(states)
        if states is not None:
            rec['recurrent_states_len'] = len(items)
            rec['recurrent_states'] = []
            for key, st in items:
                if torch.is_tensor(st):
                    info = tensor_info(st)
                    rec['recurrent_states'].append({'key': key, **info})
                    print('state layer=%s type=%s key=%s shape=%s dtype=%s device=%s memory_mib=%.4f' % (idx, typ, key, info['shape'], info['dtype'], info['device'], info['memory_mib']))
                else:
                    rec['recurrent_states'].append({'key': key, 'python_type': type(st).__name__, 'is_none': st is None})
        records.append(rec)
    result = {'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'model_path': MODEL_PATH, 'model_class': model.__class__.__name__, 'cache_class': cache.__class__.__name__, 'layer_count': len(layer_types), 'gdn_layer_count': sum(1 for x in layer_types if x == 'linear_attention'), 'attention_layer_count': sum(1 for x in layer_types if x == 'full_attention'), 'records': records}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('wrote', OUT)
if __name__ == '__main__': main()
