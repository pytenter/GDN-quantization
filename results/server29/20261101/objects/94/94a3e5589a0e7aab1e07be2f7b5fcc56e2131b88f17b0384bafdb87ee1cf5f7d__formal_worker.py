"""Sampled full-vocabulary KL with one native GPU cache and V4 state diagnostics."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path
import functional_worker as w

D = Path(os.environ['LING_PROSPECTIVE_RESULTS']).resolve()
STATE_PROBES = (8192, 16383, 24575, 32767)


def original_state_diagnostic(torch, capture, direction, cache, reference, layers, student_snapshot=None):
    # Exactly the per-layer sums and final expressions in V4 state_diagnostic.
    error_squared, fp_squared, branch_squared, numel = 0., 0., 0., 0
    for i, layer in enumerate(layers):
        fp = reference[i:i+1].to(capture.device).float()
        branch = (direction.cache_state(capture, cache, layer) if student_snapshot is None
            else student_snapshot[i:i+1].to(capture.device)).float()
        error_squared += float((branch - fp).square().sum())
        fp_squared += float(fp.square().sum())
        branch_squared += float(branch.square().sum())
        numel += fp.numel()
    return dict(accumulated_error_l2=math.sqrt(error_squared),
        accumulated_error_mse=error_squared/numel,fp_state_l2=math.sqrt(fp_squared),
        student_state_l2=math.sqrt(branch_squared))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--document',required=True)
    ap.add_argument('--condition',required=True); a = ap.parse_args()
    started = time.time(); capture = audit = None
    tag = f'{a.document}__{a.condition}__formal'
    root = D / 'formal'; root.mkdir(exist_ok=True)
    result_path = root / (tag+'.json')
    original_qdq = core = None
    try:
        assert not (D / 'STOP.json').exists()
        gates = w.read(D / 'QUALIFICATION_GATES_COMPLETED.json')
        assert gates['status'] == 'ALL_EIGHT_GATES_PASS'
        assert w.read(D/'FP_PAIRING_CONTRACT_GATE.json')['status']=='PASS'
        assert w.read(D/'FP_PAIRING_CONTRACT_GATE.json')['spec_sha256']==w.sha(D/'FROZEN_SPEC.json')
        freeze = w.read(D / 'FORMAL_CODE_FREEZE.json')
        for name,h in freeze['files'].items(): assert w.sha(D / name) == h, ('FORMAL_SOURCE_CHANGED',name)
        for name,h in freeze['dependencies'].items(): assert w.sha(name) == h, ('DEPENDENCY_CHANGED',name)
        assert not result_path.exists(), 'ALREADY_EXECUTED_NO_RETRY'
        spec = w.read(D / 'FROZEN_SPEC.json')
        assert w.sha(w.V4_CODE/'persistent_engine_v1.py') == spec['engine_sha256'], 'V4_ENGINE_CHANGED'
        secondary = w.read(D / 'SECONDARY_SPEC.json')
        assert secondary['E_future'] == secondary['Hidden_KL'] == 'NOT_FROZEN_NOT_COMPUTED'
        from prospective_common import verify
        cfg = verify(); gpu = w.route(cfg)
        assert cfg['selected_gpu_uuid']==w.read(D/'EXECUTION_ROUTING.json')['document_devices'][a.document]
        qualification=w.read(D/'qualification_units'/(a.document+'__'+a.condition+'.json'))
        assert qualification['status']=='PASS' and qualification['qualified_gpu_uuid']==cfg['selected_gpu_uuid']
        assert qualification['parent_sha256']==w.sha(D/'runs'/f'{a.document}__{a.condition}__parent.json')
        assert qualification['resumed_sha256']==w.sha(D/'runs'/f'{a.document}__{a.condition}__resumed_parity.json')
        entry = next(x for x in spec['documents'] if x['document'] == a.document)
        assert any(u['document']==a.document and u['rotation']==a.condition for u in spec['new_units']), 'UNAUTHORIZED_NEW_SCIENTIFIC_KEY'
        w.publish(root/(tag+'.STARTED.json'),dict(pid=os.getpid(),started_epoch=started,
            document=a.document,condition=a.condition,spec_sha256=w.sha(D/'FROZEN_SPEC.json'),
            secondary_sha256=w.sha(D/'SECONDARY_SPEC.json'),formal_code_freeze_sha256=w.sha(D/'FORMAL_CODE_FREEZE.json')))
        import torch
        runtime = [w.initialize(torch, cfg['seed'])]; identity = w.proc_identity(os.getpid())
        assert w.device(gpu,cfg)['exclusive'], 'PRELAUNCH_GPU_BUSY'
        for name,row in spec['source_assets'].items():
            s = Path(name).stat()
            assert s.st_size == row['bytes'] and s.st_mtime_ns == row['mtime_ns'], ('ASSET_METADATA_CHANGED',name)
        base = w.load_module(w.PRIOR/'run_short_horizon_proxy_v2.py','formal_verified_base')
        direction = w.load_module(base.DIRECTION_RUNNER,'formal_verified_direction')
        direction.GPU_UUIDS[gpu] = cfg['gpu_uuids'][str(gpu)]
        host = direction.gpu_guard(torch,gpu)
        assert all(host[k] == cfg['runtime'][k] for k in ('kernel','driver','torch','cuda_runtime','selector'))
        _,_,capture_module,core = direction.load_assets(gpu)
        from prospective_common import load_inputs
        _,tokens,token_path,rotations,rotation_ids = load_inputs(torch,cfg,a.document,
            [a.condition])
        capture = capture_module.Capture(); capture.model.eval()
        assert not any(p.requires_grad for p in capture.model.parameters())
        audit = w.engine.PersistentAudit(torch,direction,core,capture,cfg['layers'],[a.condition])
        cp = D/'runs'/f'{a.document}__{a.condition}__8192.pt'
        parent = w.read(D/'runs'/f'{a.document}__{a.condition}__parent.json')
        assert w.sha(cp) == parent['checkpoint_sha256']
        saved = torch.load(cp,map_location='cpu',weights_only=False)
        assert saved['next_position'] == 8192 and saved['token_sha256'] == w.sha(token_path)
        assert saved['frozen_spec_sha256'] == w.sha(D/'FROZEN_SPEC.json')
        cache = w.transfer(torch,saved['cache'],capture.device)
        assert w.object_fingerprint(torch,cache) == saved['cache_fingerprint']
        if a.condition != 'FP': w.restore_audit(audit,a.condition,w.transfer(torch,saved['audit_state'],capture.device))
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng']); __import__('numpy').random.set_state(saved['numpy_rng'])
        del saved; gc.collect()
        rotation = rotations.get(a.condition)
        if rotation is not None: rotation = rotation.to(capture.device)
        positions = spec['primary']['positions']; sample_set = set(positions)
        refdir = root/(a.document+'__FP_reference')
        if a.condition == 'FP':
            refdir.mkdir(); fp_manifest = None
        else:
            # Original FROZEN_SPEC remains byte-identical for saved checkpoint
            # identity; the new execution routes its own real FP payload.
            assert w.read(D/'EXECUTION_ROUTING.json')['reference_experiment']==str(D)
            fp_manifest = w.read(D/'formal'/(a.document+'__FP__formal.json'))
            assert fp_manifest['status'] == 'PASS' and fp_manifest['sample_positions'] == positions
            assert fp_manifest['token_sha256'] == w.sha(token_path)
            assert fp_manifest['gpu_uuid'] == host['gpu_uuid'] == w.read(D/'EXECUTION_ROUTING.json')['document_devices'][a.document]
            assert fp_manifest['primary_spec_sha256'] == w.sha(D/'FROZEN_SPEC.json')
            assert fp_manifest['runtime'][0]['torch'] == torch.__version__
        scale = dict(min=float('inf'),max=0.,sum=0.,count=0)
        codes = dict(min=127,max=-127,total=0,saturated=0,out_of_range=0,pre_clamp_clipped=0)
        # This is the frozen replay_core J_INJ norm formula, a quantization
        # diagnostic, never a newly constructed future functional proxy.
        injection = dict(sum=0.,count=0,min=float('inf'),max=0.)
        original_qdq = core.qdq_r128
        def observe_qdq(value):
            result = original_qdq(value)
            dequant,code,sc = result
            if a.condition != 'FP':
                codes['min'] = min(codes['min'],int(code.min()))
                codes['max'] = max(codes['max'],int(code.max()))
                codes['total'] += code.numel()
                codes['saturated'] += int((code.abs()==127).sum())
                codes['out_of_range'] += int(((code < -127)|(code > 127)).sum())
                rounded = torch.round(value.float()/sc)
                codes['pre_clamp_clipped'] += int(((rounded < -127)|(rounded > 127)).sum())
                norm = lambda x: x.square().sum(dim=(-3,-2,-1))
                inj = float((norm(dequant-value)/(norm(value)+1e-12)).mean())
                assert math.isfinite(inj)
                injection['sum'] += inj; injection['count'] += 1
                injection['min'] = min(injection['min'],inj); injection['max'] = max(injection['max'],inj)
            return result
        core.qdq_r128 = observe_qdq
        kl = []; state = {}; refs = []; memory = []; sample_bytes = hashlib.sha256()
        parity_hash = hashlib.sha256(); suffix_parity_pass = False
        deferred = None
        def collect_metrics(position,payload,student_logits,student_snapshot=None):
            if position in sample_set:
                ref_logits = payload['logits'].to(capture.device)
                student_logits = student_logits.to(capture.device)
                value = w.engine.full_kl(torch,ref_logits,student_logits)
                assert math.isfinite(value)
                kl.append(dict(position=position,KL=value,vocabulary_dimension=int(student_logits.shape[-1])))
            if position in STATE_PROBES:
                state[str(position)] = original_state_diagnostic(torch,capture,direction,cache,
                    payload['state'],cfg['layers'],student_snapshot)
        torch.cuda.reset_peak_memory_stats(); loop_start = time.perf_counter()
        for position in range(8192,32768):
            assert not (D/'STOP.json').exists(), 'SAFE_TOKEN_STOP'
            assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            audit.active = None if a.condition == 'FP' else a.condition
            cache,logits,_ = direction.model_step(torch,capture,cache,tokens[position],position)
            assert bool(torch.isfinite(logits).all()) and logits.shape[-1] == 157184
            if a.condition != 'FP':
                metadata = audit.quantize(a.condition,cache,rotation,position,False)
                for item in metadata.values():
                    scale['min'] = min(scale['min'],item['scale_min']); scale['max'] = max(scale['max'],item['scale_max'])
                    scale['sum'] += item['scale_mean']; scale['count'] += 1
            if position < 8256:
                parity_hash.update(position.to_bytes(8,'little')); parity_hash.update(w.engine.tensor_bytes(torch,logits))
            if position == 8255:
                assert parity_hash.hexdigest() == parent['uninterrupted_suffix_full_logit_sha256'], 'FORMAL_OBSERVER_LOGITS_MUTATION'
                assert w.object_fingerprint(torch,cache) == parent['final_full_native_cache_sha256'], 'FORMAL_OBSERVER_CACHE_MUTATION'
                suffix_parity_pass = True
                if deferred is not None:
                    collect_metrics(*deferred)
                    deferred = None
            if a.condition == 'FP' and (position in sample_set or position in STATE_PROBES):
                payload = dict(position=position,token=tokens[position],logits=None,state=None)
                if position in sample_set:
                    payload['logits'] = logits.detach().cpu().contiguous()
                    sample_bytes.update(position.to_bytes(8,'little')); sample_bytes.update(w.engine.tensor_bytes(torch,payload['logits']))
                if position in STATE_PROBES:
                    payload['state'] = torch.cat([direction.cache_state(capture,cache,l) for l in cfg['layers']]).detach().cpu().contiguous()
                path = refdir/f'position_{position:05d}.pt'; torch.save(payload,path)
                refs.append(dict(position=position,path=str(path),sha256=w.sha(path),bytes=path.stat().st_size,
                    logits=position in sample_set,state=position in STATE_PROBES))
                del payload
            elif a.condition != 'FP' and (position in sample_set or position in STATE_PROBES):
                reference = next(r for r in fp_manifest['references'] if r['position'] == position)
                assert w.sha(reference['path']) == reference['sha256'], 'FP_REFERENCE_CHANGED'
                payload = torch.load(reference['path'],map_location='cpu',weights_only=False)
                assert payload['position'] == position and payload['token'] == tokens[position], 'FP_STUDENT_POSITION_TOKEN_MISMATCH'
                if not suffix_parity_pass:
                    assert position == 8192 and deferred is None
                    snapshot = torch.cat([direction.cache_state(capture,cache,l) for l in cfg['layers']]).detach().cpu().contiguous()
                    deferred = (position,payload,logits.detach().cpu().contiguous(),snapshot)
                else:
                    collect_metrics(position,payload,logits)
                    del payload
            if (position+1)%512 == 0:
                guard = w.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,
                    evidence=root/(tag+'.GPU_EXCLUSIVITY.jsonl'))
                assert guard['exclusive'],guard['failure_conditions']
                assert shutil.disk_usage(D).free > 2*1024**3,'PHYSICAL_STORAGE_EXHAUSTED'
                row = dict(position=position+1,elapsed=time.perf_counter()-loop_start,
                    allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                memory.append(row); print(json.dumps(dict(event='formal_progress',tag=tag,**row)),flush=True)
            del logits
        assert suffix_parity_pass and deferred is None
        runtime.append(w.snapshot(torch,'after_formal',cfg['seed']))
        if a.condition != 'FP':
            assert [x['position'] for x in kl] == positions and len(kl) == 384
            primary = sum(x['KL'] for x in kl)/384
            integrity = audit.integrity(a.condition,32768)
            assert scale['count'] == injection['count'] == 24576*18
            assert codes['out_of_range'] == 0
            quant = dict(window=[8192,32768],scale=dict(min=scale['min'],max=scale['max'],
                mean=scale['sum']/scale['count'],count=scale['count']),codes=codes,
                saturation_fraction=codes['saturated']/codes['total'],clipping_count=codes['pre_clamp_clipped'],
                J_INJ_original_replay_norm=dict(mean=injection['sum']/injection['count'],
                    min=injection['min'],max=injection['max'],count=injection['count']),
                no_prefix_scale_history_invented=True)
        else:
            assert len([r for r in refs if r['logits']]) == 384
            primary = integrity = quant = None
        final_inventory = w.engine.hybrid_inventory(torch,cache,32767,cfg['layers'],24)
        w.publish(result_path,dict(status='PASS',document=a.document,condition=a.condition,endpoint=32768,
            primary_mean_KL=primary,primary_samples=kl,sample_positions=positions,
            references=refs,sampled_logits_content_sha256=sample_bytes.hexdigest() if a.condition=='FP' else None,
            vocabulary_dimension=157184,token_sha256=w.sha(token_path),gpu_uuid=host['gpu_uuid'],
            primary_spec_sha256=w.sha(D/'FROZEN_SPEC.json'),secondary_spec_sha256=w.sha(D/'SECONDARY_SPEC.json'),
            checkpoint_sha256=w.sha(cp),runtime=runtime,persistent_integrity=integrity,
            formal_observer_uninterrupted_parity_PASS=suffix_parity_pass,
            observer_qualified_before_first_primary_reduction=True,
            state_diagnostics=state,quantization_diagnostics=quant,final_cache_inventory=final_inventory,
            all_finite=True,memory=memory,started_epoch=started,completed_epoch=time.time(),
            elapsed_seconds=time.time()-started,E_future='NOT_FROZEN_NOT_COMPUTED',Hidden_KL='NOT_FROZEN_NOT_COMPUTED'))
    except BaseException as e:
        failure = dict(status='STOP',stage='formal',document=a.document,condition=a.condition,
            reason=repr(e),traceback=traceback.format_exc(),started_epoch=started,stopped_epoch=time.time(),
            physical_resource_stop='out of memory' in str(e).lower(),no_retry=True,
            science_verdict='INCOMPLETE_NOT_EVALUATED')
        if not (root/(tag+'.STOP.json')).exists(): w.publish(root/(tag+'.STOP.json'),failure)
        if not (D/'STOP.json').exists(): w.publish(D/'STOP.json',failure)
        raise
    finally:
        if original_qdq is not None: core.qdq_r128 = original_qdq
        if audit is not None: audit.close()
        if capture is not None: capture.close()


if __name__ == '__main__': main()
