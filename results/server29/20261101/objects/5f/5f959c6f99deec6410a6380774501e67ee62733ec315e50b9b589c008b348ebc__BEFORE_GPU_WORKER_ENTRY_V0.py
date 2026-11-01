"""One directly-owned SGLang TP1 model and one frozen non-AIME 512-token trace."""
from pathlib import Path
import argparse
import os
import sys
import time
import json
import hashlib
import traceback
from array import array
from qualification_io import read, write, ref, sha, verify, wait, event


def verify_runtime(root, contract, host):
    runtime = Path(host['runtime_source'])
    manifest = read(Path(root) / 'fixtures/SGLANG_SOURCE_MANIFEST.json')
    for relative, expected in manifest['files'].items():
        p = runtime / relative
        if p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise RuntimeError('RUNTIME_SOURCE_CHANGED:' + relative)
    for relative, expected in contract['model_stat_identity'][host['key']].items():
        stat = (Path(host['model_root']) / relative).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise RuntimeError('MODEL_STAT_IDENTITY_CHANGED:' + relative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--device-key', required=True)
    parser.add_argument('--arm', choices=('B0', 'B1', 'B2', 'B3'), required=True)
    args = parser.parse_args()
    root, job = Path(args.root), Path(args.job)
    contract = read(args.contract)
    device = contract['devices'][args.device_key]
    host = contract['hosts'][device['host']]
    stop = root / 'qualification/control/FIRST_FAILURE.json'
    verify(root, contract)
    verify_runtime(root, contract, host)
    authorization = read(root / 'authorization/CONTROL_REPAIR_AUTHORIZATION.json')
    assert authorization['user_quote'] == '不用管那个要求了，修正后就执行GPU资格吧'
    launch = wait(job / 'LAUNCH.json', job=job, stop=stop, seconds=180)
    sys.path.insert(0, str(root / 'control/runtime'))
    from resource_monitor import read_proc_identity
    actual = read_proc_identity(os.getpid())
    assert all(actual[k] == launch['actual_identity'][k] for k in ('pid', 'uid', 'start_ticks', 'boot_id', 'ppid'))
    assert actual['ppid'] == launch['controller']['pid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == device['uuid']
    write(job / 'ACTUAL_ENV.json', dict(status='PASS', identity=actual,
          CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
          expected_uuid=device['uuid'], python=sys.executable,
          PYTHONPATH=os.environ.get('PYTHONPATH'), CUDA_initialized=False,
          contract=ref(args.contract)))
    import torch
    import numpy as np
    import importlib.metadata as md
    import random
    torch.set_num_threads(1)
    assert torch.__version__ == '2.9.1+cu128'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    torch.cuda.init()
    context_sentinel = torch.empty((1,), dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    assert uuid == device['uuid'] and props.name == device['name']
    assert (props.major, props.minor) == tuple(device['compute_capability'])
    write(job / 'BOOTSTRAP.json', dict(status='ACTUAL_SENTINEL_ALLOCATED_VISIBILITY_PENDING',
          gpu_uuid=uuid, actual_gpu_uuid=uuid, identity=actual,
          GPU_model=props.name, CUDA_initialized=torch.cuda.is_initialized(),
          versions={k: md.version(k) for k in ('torch', 'triton', 'sglang', 'transformers', 'numpy')}))
    wait(job / 'RESOURCE_VISIBLE_ACK.json', job=job, stop=stop, seconds=180)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, host['runtime_source'])
    from canonical_codec import AuthorizedCudaBackend
    from qualified_observer import QualificationBoundary
    from sglang_four_codec_adapter import SingleRequestKDAAdapter
    import sglang.benchmark.one_batch as bench
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import publish
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.arg_groups.overrides import resolving_view

    fixture = read(root / 'fixtures/NON_AIME_512_TOKEN_FIXTURE.json')
    token_ids = fixture['input_ids']
    assert len(token_ids) == 512 and fixture['AIME_content'] is False
    native_backend = AuthorizedCudaBackend(torch, root / contract['cuda_artifact']['relative_path'],
                      gpu_stage_authorized=True,
                      expected_library_sha256=contract['cuda_artifact']['sha256'])
    progress = lambda value: event(job / 'PROGRESS.jsonl', value)
    progress(dict(stage='MODEL_RUNTIME_IMPORT_COMPLETE', arm=args.arm))
    checkpoints, sampled = {}, []
    saved = {}

    def run_qualification(server_args, port_args, bench_args, gpu_id, tp_rank):
        assert gpu_id == tp_rank == 0
        publish(server_args, role='scheduler')
        cfg = resolving_view(server_args)
        assert cfg.tp_size == 1 and cfg.max_running_requests == 1
        assert cfg.sampling_backend == 'pytorch' and cfg.context_length == 262144
        progress(dict(stage='MODEL_LOAD_BEGIN', arm=args.arm))
        load_start = time.monotonic()
        wrapper, tokenizer = bench.load_model(server_args, port_args, gpu_id, tp_rank)
        runner = wrapper.torch_runner
        load_seconds = time.monotonic() - load_start
        assert runner.model_config.context_len == 262144
        assert runner.max_total_num_tokens >= 262144
        # One new process, one new request. Discard all startup allocator state;
        # the first real recurrence must consume explicitly initialized zeros.
        wrapper.clear()
        cache = runner.req_to_token_pool.mamba_pool.mamba_cache
        assert cache.temporal.dtype == torch.float32
        cache.temporal.zero_()
        for conv in cache.conv:
            conv.zero_()
        torch.cuda.synchronize()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        h = torch.ones((1, 1), dtype=torch.float32)
        while h.shape[0] < 128:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        h = (h / (128 ** .5)).to('cuda')
        assert float((h.T @ h - torch.eye(128, device='cuda')).abs().max()) <= 1.e-6
        boundary = QualificationBoundary(torch, args.arm, h, backend=native_backend,
                    evidence_directory=job / 'numerical_evidence', progress=progress)
        adapter = SingleRequestKDAAdapter(boundary)
        adapter.install()
        sampling = SamplingParams(temperature=1.0, top_p=.95, top_k=20,
                                  repetition_penalty=1.0, sampling_seed=1,
                                  max_new_tokens=fixture['qualification_continuation_tokens'])
        sampling.normalize(tokenizer)
        req = Req(rid='fixed_non_aime_' + args.arm, origin_input_text=fixture['text'],
                  origin_input_ids=array('q', token_ids[:32]),
                  sampling_params=sampling, return_logprob=True,
                  eos_token_ids={156895}, vocab_size=157184)
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(0, 32)
        progress(dict(stage='FRESH_REQUEST_ZERO_CACHE_READY', arm=args.arm,
                      cache_shape=list(cache.temporal.shape), cache_dtype=str(cache.temporal.dtype)))

        def checkpoint(count, logits, batch):
            assert sorted(boundary.cursor) == contract['expected_KDA_layer_ids']
            assert all(pos == count - 1 for pos in boundary.cursor.values())
            assert boundary.writes == 18 * count and boundary.consumptions == 18 * (count - 1)
            assert tuple(logits.shape) == (1, 157184) and bool(torch.isfinite(logits).all())
            states = {}
            for layer, (pool, index) in sorted(adapter.actual_caches.items()):
                actual_state = pool.index_select(0, index.to(torch.int64)).detach().cpu().contiguous()
                states[str(layer)] = dict(sha256=hashlib.sha256(actual_state.view(torch.uint8).numpy().tobytes()).hexdigest(),
                                          shape=list(actual_state.shape), dtype=str(actual_state.dtype))
            p = job / ('CHECKPOINT_%04d_LOGITS.npy' % count)
            with p.open('xb') as stream:
                np.save(stream, logits.detach().float().cpu().numpy(), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            si = batch.sampling_info
            assert si.temperatures.flatten().tolist() == [1.] and si.top_ps.tolist() == [np.float32(.95)]
            assert si.top_ks.tolist() == [20] and si.sampling_seed.tolist() == [1]
            value = dict(tokens_processed=count, actual_seq_len=int(batch.seq_lens_cpu[0]),
                         writes=boundary.writes, actual_next_kernel_consumptions=boundary.consumptions,
                         real_cache_states=states, logits=ref(p),
                         prefill_writes=adapter.prefill_writes, decode_writes=adapter.decode_writes,
                         sampled_id=int(sampled[-1]), source_contract=ref(args.contract),
                         original_sampler_backend='pytorch', effective_temperature=1., effective_top_p=float(si.top_ps[0]),
                         effective_top_k=20, effective_seed=1, return_logprob=True,
                         teacher_forced_qualification_only=True, AIME=False)
            assert value['actual_seq_len'] == count
            write(job / ('CHECKPOINT_%04d.json' % count), value)
            checkpoints[str(count)] = value

        with torch.inference_mode():
            next_ids, logits, batch = wrapper.extend([req])
            sampled.append(int(next_ids[0]))
            checkpoint(32, logits, batch)
            for position in range(32, 512):
                if stop.exists():
                    raise RuntimeError('FIRST_FAILURE_LATCHED')
                forced = int(token_ids[position])
                req.output_ids.append(forced)
                feed = torch.tensor([forced], dtype=torch.int64, device='cuda')
                next_ids, logits = wrapper.decode(feed, batch)
                sampled.append(int(next_ids[0]))
                if position + 1 in (128, 512):
                    checkpoint(position + 1, logits, batch)
                if (position + 1) % 8 == 0:
                    progress(dict(stage='REAL_TOKEN_COMPLETE', arm=args.arm,
                                  tokens_processed=position + 1, elapsed_seconds=time.monotonic() - start))
        assert boundary.writes == 9216 and boundary.consumptions == 9198
        assert adapter.prefill_writes == 576 and adapter.decode_writes == 8640
        assert len(boundary.comparisons) == 72 and len(adapter.gate_probes) == 72
        saved.update(model_load_seconds=load_seconds, cache_shape=list(cache.temporal.shape),
                     comparisons=boundary.comparisons, numerical_evidence=boundary.evidence,
                     gate_probes=adapter.gate_probes, native_backend='ORIGINAL_SGLANG_UNPACKED_TRITON_KDA',
                     original_KDA_update_code_unchanged=True, sampler_code_unchanged=True,
                     temporal_dtype=str(cache.temporal.dtype), max_total_num_tokens=runner.max_total_num_tokens,
                     model_context_len=runner.model_config.context_len,
                     sample_ids=sampled, actual_nccl_port=port_args.nccl_port,
                     cache_zeroed_before_first_request=True, independent_request_and_RNG=True,
                     QDQ_write_count=boundary.writes, actual_next_kernel_consumption_count=boundary.consumptions,
                     prefill_QDQ_write_count=adapter.prefill_writes, decode_QDQ_write_count=adapter.decode_writes)
        adapter.close()
        boundary.close()

    cli = argparse.ArgumentParser()
    ServerArgs.add_cli_args(cli)
    argv = list(contract['common_server_args'])
    argv += ['--model-path', host['model_root'], '--host', '127.0.0.1', '--port', str(device['server_port'])]
    parsed = cli.parse_args(argv)
    server_args = ServerArgs.from_cli_args(parsed)
    bench_args = bench.BenchArgs(correctness_test=True, batch_size=(1,), input_len=(32,), output_len=(481,), cut_len=32)
    bench.correctness_test = run_qualification
    bench.main(server_args, bench_args)
    torch.cuda.synchronize()
    verify(root, contract)
    verify_runtime(root, contract, host)
    result = dict(status='PASS', stage='REAL_MODEL_PERSISTENT_512', arm=args.arm,
                  source_verified=True, numerical_pass=True, profile_verified=True,
                  environment_verified=True, gpu_uuid=uuid, identity=actual,
                  device_key=args.device_key, source_contract=ref(args.contract),
                  micro_qualification_gate=contract['arithmetic_micro_gate'],
                  actual_tokens_processed=512, actual_model_instances=1,
                  models_loaded=1, AIME_generations=0, checkpoints=checkpoints,
                  elapsed_seconds=time.monotonic() - start,
                  CUDA_peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                                 reserved_bytes=torch.cuda.max_memory_reserved(),
                                 scope='reset_after_context_sentinel_before_model_load_through_result_record'),
                  runtime_versions={k: md.version(k) for k in ('torch', 'triton', 'sglang', 'transformers', 'numpy')},
                  **saved)
    write(job / 'WORKER_RESULT.json', result)
    wait(job / 'RESULT_OBSERVED_ACK.json', job=job, stop=stop, seconds=1800)
    print('REAL_MODEL_PERSISTENT_512_PASS', args.arm, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
