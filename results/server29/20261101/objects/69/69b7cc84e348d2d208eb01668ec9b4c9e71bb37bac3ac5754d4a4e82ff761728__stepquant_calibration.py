"""Official STEPQuant@6 fit after read-only native-model calibration collection.

Imported by the one shared worker's calibration mode. This file never loads a
second model or launches a process. Root owns all GPU dispatch/cleanup.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
from stepquant.calibration import calibrate
from stepquant.quantization import QuantizationPlan, StateCodec, fit_state
from stepquant.reproducibility import checkpoint_identity, source_identity, package_versions
from stepquant.cli import fingerprint
from stepquant_calibration_adapter import NativeShadowStatistics

LAYERS = (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('No automatic calibration overwrite/retry: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.link(temporary, path); temporary.unlink()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


@torch.inference_mode()
def run_calibration(model, tokenizer, native_module, root, inputs, progress_callback=None):
    root = Path(root)
    torch.set_num_threads(4)  # Exact official CLI CPU-thread setting.
    fixed = dict(status='CPU_INPUTS_FROZEN', segments=32, sequence_length=2048,
        seed=0, sample_every=8, max_snapshots=64, lifetime_horizon=2048,
        nominal_bits=6, pivots=32, value_group_size=32, gamma=.25, fit_device='cpu')
    if any(inputs.get(k) != v for k,v in fixed.items()):
        raise ValueError('Official calibration input/config drift')
    if sha(native_module.__file__) != '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a':
        raise ValueError('Frozen native model source mismatch')
    source = json.loads((root/'OFFICIAL_STEPQUANT_SOURCE_MANIFEST.json').read_text())
    for row in source['assets']:
        p = root / row['path']
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Official STEPQuant source drift: ' + row['path'])
    token_path = root / inputs['segments_path']
    if sha(token_path) != inputs['segments_sha256']:
        raise ValueError('Frozen calibration token file drift')
    records = [json.loads(line) for line in token_path.read_text().splitlines() if line]
    if len(records) != 32:
        raise ValueError('Official segment count changed')
    combined = hashlib.sha256()
    for i, record in enumerate(records):
        ids = torch.tensor(record['input_ids'], dtype=torch.long)
        raw = ids.numpy().astype('<i8', copy=False).tobytes()
        if record['segment_index'] != i or len(ids) != 2048 or hashlib.sha256(raw).hexdigest() != record['token_ids_sha256']:
            raise ValueError('Frozen calibration token identity mismatch')
        combined.update(raw)
    if combined.hexdigest() != inputs['official_combined_token_ids_sha256']:
        raise ValueError('Official aggregate calibration token identity mismatch')
    output = root/'calibration'; output.mkdir(exist_ok=True)
    if any((output/n).exists() for n in ('qwen35_9b_statistics.pt','qwen35_9b_stepquant6_plan.pt','CALIBRATION_RESULT.json')):
        raise FileExistsError('Existing calibration evidence; no automatic retry')
    # 24 layers x 64 samples x 32 heads x 128 x 128 x FP32 = 3 GiB.
    # Retain headroom for atomic storage, temporary serialization and metadata.
    if shutil.disk_usage(output).free < 7*1024**3:
        raise RuntimeError('Insufficient 7GiB calibration serialization/reserve capacity')
    device = model.get_input_embeddings().weight.device
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        # Inference only; freezing weights changes no forward mathematics.
        for p in model.parameters():
            p.requires_grad_(False)
    start = time.monotonic(); progress = []; state_shapes = {}
    def emit(value):
        value = dict(value, elapsed_seconds=time.monotonic()-start)
        progress.append(value)
        if progress_callback:
            progress_callback(value)
    with NativeShadowStatistics(model, native_module, 8, 64) as observer:
        if set(observer.layer_name_to_index.values()) != set(LAYERS) or len(observer.layer_name_to_index) != 24:
            raise ValueError('Qwen3.5-9B GDN layer identity mismatch')
        layer_name_to_index = dict(observer.layer_name_to_index)
        for i, record in enumerate(records):
            segment_start = time.monotonic()
            ids = torch.tensor([record['input_ids']], dtype=torch.long, device=device)
            # Exactly one independent full segment. Only native kernels return
            # actual model output/cache; no quantized feedback in calibration.
            result = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            if not bool(torch.isfinite(result.logits).all()):
                raise FloatingPointError('Nonfinite native calibration logits')
            for name, layer in layer_name_to_index.items():
                states = result.past_key_values.layers[layer].recurrent_states
                state = states[0] if isinstance(states, (dict,list,tuple)) else states
                if tuple(state.shape) != (1,32,128,128) or state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise ValueError('Actual native 9B state shape/dtype/finite mismatch')
                state_shapes[name] = list(state.shape)
            del result, ids
            if torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
            emit(dict(event='official_calibration_segment_complete',segment=i+1,total_segments=32,
                token_ids_sha256=record['token_ids_sha256'], segment_seconds=time.monotonic()-segment_start,
                peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
                peak_cuda_reserved=torch.cuda.max_memory_reserved(device)))
        statistics = observer.export_statistics()
    for name, stat in statistics.items():
        if stat['tokens'] != 65536 or stat['snapshots'].shape != (64,32,128,128):
            raise ValueError('Official actual statistics coverage mismatch: '+name)
    model_root = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
    identity = json.loads((root/'MODEL_TOKENIZER_FRESH_IDENTITY.json').read_text())
    if identity['status'] != 'PASS' or len(identity['files']) != 17 or not all(x['PASS'] for x in identity['files']):
        raise ValueError('Fresh checkpoint identity gate not PASS')
    provenance = dict(checkpoint_fingerprint=fingerprint(model_root),checkpoint_identity=checkpoint_identity(model_root),
        full_model_file_SHA256_identity=identity, data=inputs,
        calibration_versions=package_versions(),calibration_source_sha256=source_identity(),
        official_commit='f2deac58131219e6aecff274f8eabe88e08d2e3c',
        calibration_state_scope='official FP32 shadow driven by actual native-normalized inputs; native outputs/cache unmodified; no quantized feedback',
        layer_name_to_index=layer_name_to_index,state_shape_per_layer=state_shapes)
    saved_statistics = atomic_save(dict(statistics_version=1,statistics=statistics,**provenance),output/'qwen35_9b_statistics.pt')
    emit(dict(event='official_statistics_saved',asset=saved_statistics))
    emit(dict(event='official_full_stepquant6_fit_started',device='cpu'))
    # This is the unchanged full official global allocation, not an all8 ablation.
    artifact = calibrate(statistics,nominal_bits=6,pivots=32,horizon=2048,value_group_size=32,device='cpu')
    artifact.update(provenance,state_format='stepquant',statistics_sha256=saved_statistics['sha256'])
    settings = artifact['settings']
    if settings['integer_budget'] != 4352 or settings['integer_bits_used'] > 4352:
        raise ValueError('Official 9B nominal6 allocation budget mismatch')
    counts = {b:0 for b in (4,6,8,16)}; payload_bytes = 0; per_layer = {}
    for name, value in artifact['plans'].items():
        plan = QuantizationPlan(**value)
        heads = plan.bits[:,0]
        if tuple(plan.bits.shape) != (32,128) or not bool((plan.bits == heads[:,None]).all()):
            raise ValueError('9B official whole-head allocation mismatch')
        actual_counts = {b:int((heads==b).sum()) for b in counts}
        for b,n in actual_counts.items():counts[b]+=n
        sample = statistics[name]['snapshots'][0:1]
        codec = StateCodec(plan); packed = codec.encode(sample); decoded = codec.decode(packed)
        if not bool(torch.isfinite(decoded).all()):raise FloatingPointError('Actual plan codec nonfinite')
        torch.testing.assert_close(decoded,fit_state(sample,plan)[-1],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(decoded[:,plan.bits==16],sample[:,plan.bits==16].half().float(),atol=0,rtol=0)
        payload_bytes += packed.nbytes
        per_layer[name] = dict(layer_index=layer_name_to_index[name],heads_per_bit=actual_counts,
            impact_min=float(plan.impact.min()),impact_max=float(plan.impact.max()),
            encoded_request_payload_bytes=packed.nbytes,row_scales_dtype=str(packed.rows.dtype),
            column_scales_dtype=str(packed.columns.dtype),pivots_dtype=str(packed.pivots.dtype),
            pivot_elements=packed.pivots.numel())
    if counts[16] != 32 or sum(counts.values()) != 768:
        raise ValueError('Official global FP16 pivot/head inventory mismatch')
    saved_plan = atomic_save(artifact,output/'qwen35_9b_stepquant6_plan.pt')
    report = dict(status='PASS',STEPQUANT_PLAN_QUALIFICATION='PASS',statistics=saved_statistics,plan=saved_plan,
        official_commit=provenance['official_commit'],data_token_sha256=inputs['official_combined_token_ids_sha256'],
        total_tokens=65536,GDN_layers=24,heads=768,heads_per_bit=counts,pivots=32,
        actual_mean_data_bits_per_element=sum(b*n for b,n in counts.items())/768,
        all_layers_encoded_request_payload_bytes_B1=payload_bytes,
        official_plan_codec_fit_reference_parity='PASS_atol1e-6_rtol1e-6_pivots_exact',per_layer=per_layer,
        physical_storage_scope='formal runtime uses FP32 QDQ carrier; packed payload here is codec accounting, not claimed persistent packed residency',
        full_generation='NOT_RUN',runtime_seconds=time.monotonic()-start,progress=progress)
    emit(dict(event='official_full_stepquant6_plan_complete',asset=saved_plan))
    with (output/'CALIBRATION_RESULT.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    gate = dict(status='PASS',plan_sha256=saved_plan['sha256'],statistics_sha256=saved_statistics['sha256'],
        official_commit=provenance['official_commit'],data_token_sha256=inputs['official_combined_token_ids_sha256'],
        all_24_layers_checked=True,heads=768,FP16_pivot_heads=32,candidates=[4,6,8],
        integer_budget=4352,integer_bits_used=settings['integer_bits_used'],
        codec_fit_reference_parity='PASS_atol1e-6_rtol1e-6',pivots_FP16_exact=True,
        full_generation='NOT_RUN',persistent_physical_packed_residency='NOT_CLAIMED_FP32_QDQ_CARRIER')
    with (output/'PLAN_CPU_QUALIFICATION.json').open('x') as f:
        json.dump(gate,f,ensure_ascii=False,indent=2);f.write('\n')
    return report
