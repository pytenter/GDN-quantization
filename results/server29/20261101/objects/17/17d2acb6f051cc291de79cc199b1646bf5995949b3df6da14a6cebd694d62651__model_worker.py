"""Bounded single-cache full-model worker for the new Dual-axis factorial task.

No CLI from an old scientific experiment is invoked. The native model_step and
original R128 cache writer are delegated unchanged; Dual-M only replaces the
codec boundary after current-token logits, before next-token consumption.
Importing this module does not initialize Torch or a model.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

TASK = 'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
W = Path(__file__).resolve().parents[1]
D = Path(os.environ.get('LING_PROSPECTIVE_RESULTS', '/data/zypan/experiment_runs/' + TASK)).resolve()
GPU_UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'
LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
CONDITIONS = ('FP_NATIVE','H_R128','H_DUAL_M','L6_R128','L6_DUAL_M')
PROBES = {0,511,8191,8192,8447}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def publish(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial.{os.getpid()}')
    with tmp.open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(tmp, path); tmp.unlink()


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload_path(document, kind, first128=False):
    scope = 'formal' if kind == 'formal' else 'qualification'
    tail = '_first128' if first128 else ''
    return D / 'payloads' / f'{scope}_{document}_FP_NATIVE{tail}.npy'


def verify_source(prereg):
    manifest_path = D / 'SOURCE_MANIFEST.json'; manifest = read(manifest_path)
    assert isinstance(manifest['files'], dict) and str(Path(__file__).resolve()) in manifest['files']
    for path, digest in manifest['files'].items():
        assert Path(path).is_absolute() and sha(path) == digest, ('SOURCE_CHANGED', path)
    identity = prereg['identity']
    assert identity['target_index'] == 6 and identity['target_uuid'] == GPU_UUID
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=W, text=True).strip() == identity['source_commit']
    model_audit_path = Path(identity['model_identity_audit_path'])
    assert sha(model_audit_path) == identity['model_identity_audit_sha256']
    model = read(model_audit_path)
    assert model['status'].startswith('PASS') and model['model_directory'] == identity['model_directory']
    for item in model['files']:
        st = Path(item['path']).stat()
        assert st.st_size == item['bytes'] and st.st_mtime_ns == item['mtime_ns'] and st.st_ino == item['inode'], ('MODEL_BOUND_METADATA_CHANGED',item['path'])
    for name in ('H','L6'):
        item = identity['rotation_files'][name]
        assert sha(item['path']) == item['sha256'], ('ROTATION_CHANGED',name)
    return {'source_manifest_sha256': sha(manifest_path), 'model_identity_audit_sha256': sha(model_audit_path),
            'source_commit': identity['source_commit'], 'model_files_metadata_checked': len(model['files'])}


def load_document(args, prereg):
    frozen = read(D / 'FROZEN_DOCUMENTS.json')
    entries = frozen['formal_documents'] if args.kind == 'formal' else [frozen['canary_document']]
    entry = next((x for x in entries if x['document'] == args.document), None)
    assert entry is not None, 'DOCUMENT_NOT_FROZEN_FOR_THIS_SCOPE'
    assert sha(entry['token_file_path']) == entry['token_file_sha256']
    assert sha(entry['raw_text_path']) == entry['raw_sha256']
    tokens = read(entry['token_file_path'])
    assert isinstance(tokens, list) and len(tokens) == entry['full_token_count'] and len(tokens) >= 8449
    assert all(type(x) is int and 0 <= x < prereg['identity']['vocab_size'] for x in tokens)
    canonical = hashlib.sha256(b''.join(x.to_bytes(4,'little') for x in tokens[:8449])).hexdigest()
    assert canonical == entry['canonical_prefix_token_sha256'], 'CANONICAL_TOKEN_PREFIX_CHANGED'
    return entry, tokens


class HiddenObserver:
    def __init__(self, capture, enabled):
        self.value = None; self.enabled = enabled
        def observe(_module, inputs):
            if self.enabled: self.value = inputs[0].detach().cpu().clone()
        self.handle = capture.model.lm_head.register_forward_pre_hook(observe)
    def close(self): self.handle.remove()


def make_audit(torch, engine, direction, core, capture, condition, codec):
    class CodecAudit(engine.PersistentAudit):
        def __init__(self):
            super().__init__(torch, direction, core, capture, LAYERS, [condition])
            self.diagnostics = []; self.gpu_reference_parity = None
        def first_reference(self, rotated):
            # The historical codec contract is same-device official parity.
            # CPU/GPU reduction results are observational, never a new bitwise gate.
            if self.gpu_reference_parity is not None: return
            if condition.endswith('R128'):
                dequant, codes, scale = core.qdq_r128(rotated)
                independent_scale = rotated.float().abs().amax(-1,keepdim=True).clamp_min(1e-12)/127.
                independent_code = torch.round(rotated.float()/independent_scale).clamp(-127,127).to(torch.int8)
                independent_dequant = independent_code.float()*independent_scale
                assert torch.equal(codes, independent_code) and torch.equal(scale, independent_scale) and torch.equal(dequant, independent_dequant), 'R128_ORIGINAL_GPU_REFERENCE_MISMATCH'
                cpu = rotated.detach().cpu().float()
                cs = cpu.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127.
                cz = torch.round(cpu/cs).clamp(-127,127).to(torch.int8)
                cd = cz.float()*cs
                self.gpu_reference_parity = {'status':'PASS_SAME_GPU_ORIGINAL_R128_REFERENCE_EXACT','CPU_GPU_diagnostic_only':{
                    'code_equal':bool(torch.equal(codes.cpu(),cz)), 'scale_equal':bool(torch.equal(scale.cpu(),cs)),
                    'max_dequant_abs':float((dequant.cpu()-cd).abs().max())}}
            elif condition.endswith('DUAL_M'):
                encoded = codec.encode_dualm(rotated)
                oz,orr,oc,_none,od = codec.official_reference(rotated,architecture='kda')
                assert torch.equal(encoded.codes,oz.to(torch.int8)), 'DUAL_OFFICIAL_GPU_CODES_MISMATCH'
                assert torch.equal(encoded.rows,orr.unsqueeze(-1)) and torch.equal(encoded.columns,oc.unsqueeze(-2)), 'DUAL_OFFICIAL_GPU_SCALES_MISMATCH'
                assert torch.equal(encoded.decode(),od), 'DUAL_OFFICIAL_GPU_DEQUANT_MISMATCH'
                headwise=[codec.encode_dualm(rotated[:,h:h+1]) for h in range(rotated.shape[1])]
                for field in ('codes','rows','columns'):
                    separate=torch.cat([getattr(panel,field) for panel in headwise],dim=1)
                    together=getattr(encoded,field)
                    assert torch.equal(together.contiguous().view(torch.uint8),separate.contiguous().view(torch.uint8)), ('DUAL_BATCH_INDIVIDUAL_BYTE_MISMATCH',field)
                separate_dequant=torch.cat([panel.decode() for panel in headwise],dim=1)
                assert torch.equal(encoded.decode().contiguous().view(torch.uint8),separate_dequant.contiguous().view(torch.uint8)), 'DUAL_BATCH_INDIVIDUAL_DEQUANT_BYTE_MISMATCH'
                del headwise,separate,separate_dequant,together
                cz,cr,cc,cd = codec.direct_equation_reference(rotated.detach().cpu())
                self.gpu_reference_parity = {'status':'PASS_SAME_GPU_OFFICIAL_DUALM_REFERENCE_EXACT',
                    'batch_individual_byte_exact':True,'batch_individual_heads_tested':int(rotated.shape[1]),
                    'dequant_torch_equal':True,'dequant_byte_equal':bool(torch.equal(encoded.decode().contiguous().view(torch.uint8),od.contiguous().view(torch.uint8))),
                    'CPU_GPU_diagnostic_only':{'code_equal':bool(torch.equal(encoded.codes.cpu(),cz)),
                    'row_scale_equal':bool(torch.equal(encoded.rows.cpu(),cr)), 'column_scale_equal':bool(torch.equal(encoded.columns.cpu(),cc)),
                    'max_dequant_abs':float((encoded.decode().cpu()-cd).abs().max())}}
        def quantize(self, _condition, cache, rotation, position, probe):
            if condition.endswith('R128'):
                if position == 0: self.first_reference(direction.cache_state(capture,cache,LAYERS[0]).float()@rotation[0][None])
                before = [direction.cache_state(capture,cache,l).detach().clone() for l in LAYERS] if probe else None
                # Delegate the original physical R128 writer and its exact audit.
                result = super().quantize(condition,cache,rotation,position,probe)
                if probe:
                    for i,l in enumerate(LAYERS):
                        state=direction.cache_state(capture,cache,l)
                        self.diagnostics.append(self.diag(before[i],state,capture.current[l],position,l,None))
                    del before
                return result
            for i,layer in enumerate(LAYERS):
                state = direction.cache_state(capture,cache,layer)
                before = state.detach().clone() if probe else None
                rotated = state.float() @ rotation[i][None]
                trace = None
                if condition.endswith('DUAL_M'):
                    if position == 0 and i == 0: self.first_reference(rotated)
                    if probe: encoded,trace = codec.trace_encode_dualm(rotated)
                    else: encoded = codec.encode_dualm(rotated)
                    assert encoded.codes.dtype == torch.int8 and encoded.rows.dtype == encoded.columns.dtype == torch.float16
                    decoded = encoded.decode()
                    assert bool(torch.isfinite(decoded).all()), 'DUAL_NONFINITE_DEQUANT'
                    if probe and i == 0:
                        self.proofs[condition].append({'position':position,'codec':'DUAL_M','all_INT8':True,
                            'row_column_scales_FP16':True,'physical_packed_persistent_INT8':False,
                            'tensor_payload_bytes':encoded.tensor_payload_bytes(),'no_high_precision_next_token_shadow':True})
                else:
                    assert condition.endswith('NOQUANT')
                    decoded = rotated
                physical = decoded @ rotation[i][None].transpose(-1,-2)
                assert bool(torch.isfinite(physical).all()), 'NONFINITE_PHYSICAL_WRITEBACK'
                with torch.inference_mode(): state.copy_(physical.to(state.dtype))
                if probe:
                    assert torch.equal(state,physical.to(state.dtype)), 'PHYSICAL_INVERSE_WRITEBACK_MISMATCH'
                    self.diagnostics.append(self.diag(before,state,capture.current[layer],position,layer,trace))
                del rotated,decoded,physical,before,trace
                if condition.endswith('DUAL_M'): del encoded
            self.writes[condition] += len(LAYERS)
            for anchor in (2048,8192):
                if position >= anchor: self.after_anchors[condition][anchor] += len(LAYERS)
            self.previous[condition] = {l:direction.cache_state(capture,cache,l).detach().clone() for l in LAYERS}
            return {}
        def diag(self,before,after,record,pos,layer,trace):
            diff=(after-before).double();base=before.double()
            mse=diff.square().sum((-2,-1))/(base.square().sum((-2,-1))+1e-12)
            retention=record['log_decay'].float().exp()[:,0]
            q=record['q'].float()[:,0]
            q=q/torch.sqrt(q.square().sum(-1,keepdim=True)+1e-6)*(128**-0.5)
            e=(diff*q.double().unsqueeze(-1)).sum(-2)
            y=(base*q.double().unsqueeze(-1)).sum(-2)
            row={'position':pos,'layer':layer,'native_local_QDQ_relative_mse_per_head':mse.detach().cpu().tolist(),
                 'retention_min':float(retention.min()),'retention_max':float(retention.max()),'retention_mean':float(retention.mean()),
                 'matched_same_input_local_readout_relative_mse_per_head':(e.square().sum(-1)/(y.square().sum(-1)+1e-12)).cpu().tolist(),
                 'readout_scope':'exploratory current native-q projection of one local pre/post codec state; not a full-model causal mechanism'}
            if trace is not None:
                stats={}
                for key in ['unrounded_over127_count','rounded_clamp_count','unrounded_nonfinite_count','code_count','effective_scale_min','effective_scale_max','affected_energy_fraction_per_batch_head','affected_relative_mse_per_batch_head','local_relative_mse_per_batch_head','observational_only']:
                    value=trace[key]
                    stats[key]=value.detach().cpu().tolist() if torch.is_tensor(value) else value
                stats['scale_records']=[{key:(value.detach().cpu().tolist() if torch.is_tensor(value) else value)
                    for key,value in item.items() if key!='raw_input'} for item in trace['scale_records']]
                row['DualM_scale_clamp_diagnostics']=stats
            return row
    return CodecAudit()


def save_checkpoint(torch, tools, cache, audit, condition, binding, target):
    cpu = tools.transfer(torch,cache,'cpu')
    fp = tools.fingerprint(torch,cache)
    assert tools.fingerprint(torch,cpu)==fp, 'CHECKPOINT_CPU_COPY_MISMATCH'
    saved_audit=None if condition=='FP_NATIVE' else tools.transfer(torch,tools.audit_state(audit,condition),'cpu')
    payload={'schema':'dual_axis_native_hybrid_cpu_checkpoint_v1','cache':cpu,'cache_fingerprint':fp,
             'audit_state':saved_audit,'next_position':8192,'binding':binding,'rng':tools.rng_snapshot(torch),
             'no_high_precision_shadow':True,'state_carrier':'FP32 native cache containing post-QDQ values; not physical packed INT8'}
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+f'.partial.{os.getpid()}')
    with tmp.open('xb') as stream: torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.link(tmp,target);tmp.unlink()
    return {'path':str(target),'sha256':sha(target),'cache_fingerprint':fp,'next_position':8192}


def parity_snapshot(torch,tools,cache,audit,condition,suffix):
    return {'positions':[8192,8256],'full64_V_logits_sha256':suffix.hexdigest(),
            'full_native_hybrid_cache_sha256':tools.fingerprint(torch,cache),
            'audit_state_sha256':None if condition=='FP_NATIVE' else tools.fingerprint(torch,tools.audit_state(audit,condition)),
            'RNG_sha256':tools.fingerprint(torch,tools.rng_snapshot(torch)),
            'integrity':None if condition=='FP_NATIVE' else audit.integrity(condition,8256)}


def run(args):
    import numpy as np
    prereg=read(D/'PREREGISTRATION.json');identity=prereg['identity'];run_cfg=prereg['run']
    assert prereg['task_id']==TASK and not (D/'FIRST_FAILURE.json').exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6' and os.environ.get('LING_TARGET_GPU_UUID')==GPU_UUID
    assert (run_cfg['tokens_per_trajectory'],run_cfg['evaluation_start'],run_cfg['evaluation_count'],run_cfg['noquant_tokens'])==(8448,8192,256,128)
    source_binding=verify_source(prereg);entry,tokens=load_document(args,prereg)
    sys.path.insert(0,str(W));sys.path.insert(0,str(W/'dual_axis_task'))
    import torch
    from v2_runtime import initialize,snapshot
    from frozen_runtime_profile import validate_environment,validate_loaded_profile
    import persistent_engine_v1 as engine
    import cache_tools as tools
    import codec_adapter as codec
    from metrics_cpu import score_pair, aggregate
    runtime=[initialize(torch,0)];initial_profile=validate_environment()
    assert sys.flags.hash_randomization==0
    direction=import_file(identity['direction_runner'],'ling_dual_axis_native_direction')
    host=direction.gpu_guard(torch,6)
    assert host['gpu_uuid']==GPU_UUID
    _,_,capture_module,core=direction.load_assets(6)
    rotation=None;rotation_info=None
    if args.condition!='FP_NATIVE':
        rotation_name='L6' if args.condition.startswith('L6_') else 'H'
        rotation_info=identity['rotation_files'][rotation_name]
        asset=torch.load(rotation_info['path'],map_location='cpu',weights_only=False)
        assert asset['condition']==rotation_name and list(asset['layer_ids'])==LAYERS
        rotation=torch.stack([asset['rotations'][l] for l in LAYERS])
    binding={'task_id':TASK,'document':args.document,'condition':args.condition,'GPU_UUID':GPU_UUID,
             'token_file_sha256':entry['token_file_sha256'],'canonical_prefix_token_sha256':entry['canonical_prefix_token_sha256'],
             'rotation_sha256':None if rotation_info is None else rotation_info['sha256'],
             'preregistration_sha256':sha(D/'PREREGISTRATION.json'),**source_binding}
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    capture=audit=observer=None;cache=None;payloads=[]
    started=time.time();load_start=time.perf_counter()
    try:
        capture=capture_module.Capture();capture.model.eval()
        assert len(capture.model.model.layers)==24 and not any(p.requires_grad for p in capture.model.parameters())
        assert all(p.device.type=='cuda' for p in capture.model.parameters())
        load_seconds=time.perf_counter()-load_start
        if rotation is not None: rotation=rotation.to(capture.device)
        audit=make_audit(torch,engine,direction,core,capture,args.condition,codec)
        observer=HiddenObserver(capture,args.kind=='qual_long')
        runtime.append(snapshot(torch,'after_native_helpers_and_model_load',0))
        checkpoint_path=D/'checkpoints'/f'{args.document}__{args.condition}__8192.pt'
        checkpoint_info=None;begin=0
        if args.kind=='qual_fresh':
            parent_path=args.parent_result or D/'qualification'/f'{args.document}__{args.condition}.qual_long.json'
            parent=read(parent_path)
            assert parent['status']=='PASS' and parent['kind']=='qual_long' and parent['binding']==binding
            checkpoint_info=parent['checkpoint'];assert sha(checkpoint_info['path'])==checkpoint_info['sha256']
            cp=torch.load(checkpoint_info['path'],map_location='cpu',weights_only=False)
            assert cp['binding']==binding and cp['next_position']==8192 and cp['no_high_precision_shadow'] is True
            cache=tools.transfer(torch,cp['cache'],capture.device)
            assert tools.fingerprint(torch,cache)==cp['cache_fingerprint'], 'RESTORED_NATIVE_CACHE_NOT_EXACT'
            if args.condition!='FP_NATIVE': tools.restore_audit(audit,args.condition,tools.transfer(torch,cp['audit_state'],capture.device))
            tools.rng_restore(torch,cp['rng']);begin=8192;del cp;gc.collect()
        endpoint=128 if args.kind=='noquant' else 8256 if args.kind=='qual_fresh' else 8448
        fp256=None;fpfirst=None;ref=None;firstref=None;FP_reference_binding=None
        if args.condition!='FP_NATIVE':
            reference_result=args.fp_result or (D/('formal' if args.kind=='formal' else 'qualification')/f'{args.document}__FP_NATIVE.{"formal" if args.kind=="formal" else "qual_long"}.json')
            reference=read(reference_result)
            assert reference['status']=='PASS' and reference['condition']=='FP_NATIVE' and reference['document']==args.document and reference['UUID']==GPU_UUID
            assert reference['kind']==('formal' if args.kind=='formal' else 'qual_long')
            for key in ('document','GPU_UUID','token_file_sha256','canonical_prefix_token_sha256','preregistration_sha256','source_manifest_sha256','source_commit','model_identity_audit_sha256'):
                assert reference['binding'][key]==binding[key], ('FP_REFERENCE_BINDING_MISMATCH',key)
            target=payload_path(args.document,args.kind,first128=args.kind=='noquant')
            record=next(x for x in reference['FP_payloads'] if x['path']==str(target))
            assert sha(target)==record['sha256'], 'FP_REFERENCE_PAYLOAD_CHANGED'
            FP_reference_binding={'result_path':str(reference_result),'result_sha256':sha(reference_result),'payload':record}
        if args.condition=='FP_NATIVE' and args.kind in ('qual_long','formal'):
            D.joinpath('payloads').mkdir(exist_ok=True)
            target=payload_path(args.document,args.kind);assert not target.exists()
            temp=target.with_name(target.name+f'.partial.{os.getpid()}')
            fp256=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(256,157184));payloads.append((temp,target,fp256))
            if args.kind=='qual_long':
                target=payload_path(args.document,args.kind,first128=True);assert not target.exists()
                temp=target.with_name(target.name+f'.partial.{os.getpid()}')
                fpfirst=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(128,157184));payloads.append((temp,target,fpfirst))
        elif args.kind=='noquant':
            firstref=np.load(payload_path(args.document,args.kind,first128=True),mmap_mode='r')
            assert firstref.dtype==np.float32 and firstref.shape==(128,157184)
        elif args.condition!='FP_NATIVE':
            ref=np.load(payload_path(args.document,args.kind),mmap_mode='r')
            assert ref.dtype==np.float32 and ref.shape==(256,157184)
        prefix=hashlib.sha256();suffix=hashlib.sha256();all_logits=hashlib.sha256()
        scores=[];noquant_diffs=[];memory=[];parity=None;loop_start=time.perf_counter();postwarm_seconds=[]
        for pos in range(begin,endpoint):
            assert not (D/'FIRST_FAILURE.json').exists(), 'SAFE_TOKEN_STOP'
            tick=time.perf_counter();audit.active=None if args.condition=='FP_NATIVE' else args.condition
            cache,logits,_=direction.model_step(torch,capture,cache,tokens[pos],pos)
            assert logits.shape[-1]==157184 and bool(torch.isfinite(logits).all()), 'NONFINITE_OR_WRONG_V_LOGITS'
            # Current token readout/logits are already computed. Only the native cache
            # for the next token is quantized here, exactly as the frozen baseline.
            if args.condition!='FP_NATIVE': audit.quantize(args.condition,cache,rotation,pos,pos in PROBES)
            raw=tools.tensor_bytes(torch,logits);packet=pos.to_bytes(8,'little')+raw;all_logits.update(packet)
            if pos<8192: prefix.update(packet)
            elif pos<8256: suffix.update(packet)
            if pos>=8192 or fpfirst is not None and pos<128 or args.kind=='noquant':
                current=logits.detach().float().cpu().numpy().reshape(-1)
                assert current.shape==(157184,) and np.isfinite(current).all()
                if fpfirst is not None and pos<128: fpfirst[pos]=current
                if args.kind=='noquant':
                    baseline=np.asarray(firstref[pos]);delta=current.astype(np.float64)-baseline.astype(np.float64)
                    scored=score_pair(baseline,current)
                    noquant_diffs.append({'position':pos,'max_abs':float(np.max(np.abs(delta))),
                        'RMS':float(np.sqrt(np.mean(delta*delta))),**scored})
                elif pos>=8192:
                    baseline=current if args.condition=='FP_NATIVE' else np.asarray(ref[pos-8192])
                    if fp256 is not None: fp256[pos-8192]=current
                    scored=score_pair(baseline,current)
                    assert all(math.isfinite(float(scored[k])) for k in ('kl_fp32','kl_reference_extended','arithmetic_error_radius'))
                    scores.append({'position':pos,'consumed_token_id':tokens[pos],'next_token_id':tokens[pos+1],**scored})
                del current
            if args.kind=='qual_long' and pos==8191:
                checkpoint_info=save_checkpoint(torch,tools,cache,audit,args.condition,binding,checkpoint_path)
            if args.kind in ('qual_long','qual_fresh') and pos==8255:
                parity=parity_snapshot(torch,tools,cache,audit,args.condition,suffix)
            observer.value=None
            if pos>=511: postwarm_seconds.append(time.perf_counter()-tick)
            if (pos+1)%512==0 or pos+1==endpoint:
                torch.cuda.synchronize()
                row={'position':pos+1,'elapsed':time.perf_counter()-loop_start,'allocated':torch.cuda.memory_allocated(),
                     'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),
                     'peak_reserved':torch.cuda.max_memory_reserved()}
                memory.append(row);print(json.dumps({'event':'dual_axis_progress','kind':args.kind,'document':args.document,'condition':args.condition,**row}),flush=True)
            del logits,raw,packet
        torch.cuda.synchronize()
        actual_profile=validate_loaded_profile(True)
        runtime.append(snapshot(torch,'loop_finished',0))
        if args.kind=='qual_fresh':
            assert parity==parent['parity64'], 'OWN_PARENT_FRESH64_EXACT_MISMATCH'
        payload_records=[]
        for temp,target,array in payloads:
            array.flush()
            with temp.open('rb') as stream: os.fsync(stream.fileno())
            os.link(temp,target);temp.unlink()
            payload_records.append({'path':str(target),'sha256':sha(target),'shape':list(array.shape),'dtype':'float32','storage':'CPU/disk full vocabulary, per-position GPU offload'})
        score_summary=None
        if scores:
            n=len(scores);assert n==(64 if args.kind=='qual_fresh' else 256)
            score_summary={'count':n,'positions':[x['position'] for x in scores],
                'K_fp32_fsum':math.fsum(float(x['kl_fp32']) for x in scores)/n,
                'K_reference_extended_fsum':math.fsum(float(x['kl_reference_extended']) for x in scores)/n,
                'arithmetic_error_radius':math.fsum(float(x['arithmetic_error_radius']) for x in scores)/n,
                'scope':'formal science' if args.kind=='formal' else 'qualification canary only, not a formal document outcome'}
        integrity=None if args.condition=='FP_NATIVE' else audit.integrity(args.condition,endpoint)
        result={'status':'PASS','task_id':TASK,'kind':args.kind,'document':args.document,'condition':args.condition,
            'UUID':GPU_UUID,'host':host,'binding':binding,'input':entry,'begin':begin,'endpoint':endpoint,
            'tokens_consumed_in_this_process':endpoint-begin,'new_formal_trajectory':args.kind=='formal',
            'checkpoint':checkpoint_info,'parity64':parity,'fresh64_exact':True if args.kind=='qual_fresh' else None,
            'observer_enabled':args.kind=='qual_long','native_cache_final_sha256':tools.fingerprint(torch,cache),
            'full_logits_digest':all_logits.hexdigest(),'prefix_logits_digest':prefix.hexdigest(),'integrity':integrity,
            'GPU_codec_reference_parity':audit.gpu_reference_parity,'local_diagnostics':audit.diagnostics,
            'scores':scores,'score_summary':score_summary,'aggregate':aggregate(scores) if len(scores)==256 else None,
            'token_file_sha256':entry['token_file_sha256'],'canonical_prefix_token_sha256':entry['canonical_prefix_token_sha256'],
            'rotation_file_sha256':None if rotation_info is None else rotation_info['sha256'],
            'payloads':payload_records,'FP_payloads':payload_records,'FP_reference_binding':FP_reference_binding,'noquant_logit_differences':noquant_diffs,
            'noquant_max_abs':max((x['max_abs'] for x in noquant_diffs),default=None),
            'interpretation':'operational codec effect with frozen finite-precision rotations; no strict pure-quantization decomposition',
            'physical_packed_INT8_persistent_cache_verified':False,
            'runtime':runtime,'actual_runtime_profile':actual_profile,'source_verified':source_binding,
            'memory':memory,'CUDA_lifetime_peak':{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()},
            'CUDA_peak_scope':'reset immediately before model construction; includes loading, forward, codecs, CPU checkpoint/offload and GPU cleanup performed before result',
            'model_load_seconds':load_seconds,'loop_seconds':time.perf_counter()-loop_start,
            'postwarm_token_seconds_mean':math.fsum(postwarm_seconds)/len(postwarm_seconds) if postwarm_seconds else None,
            'started_epoch':started,'result_epoch':time.time(),'worker_elapsed_seconds':time.time()-started,
            'GPUh_scope':'controller records actual process launch through wait/cleanup; this result does not substitute for that ledger'}
        publish(args.result,result)
        publish(args.result.with_suffix('.PROFILE.json'),{'status':'PASS_ACTUAL_FROZEN_PROFILE','profile':actual_profile,
            'initial_profile':initial_profile,'actual_environment':{k:os.environ.get(k) for k in ['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID']},
            'hash_randomization':sys.flags.hash_randomization,'CUDA_lifetime_peak':result['CUDA_lifetime_peak'],
            'source_binding':binding,'epoch':time.time()})
    finally:
        if observer is not None: observer.close()
        if audit is not None: audit.close()
        if capture is not None: capture.close()
        cache=None;gc.collect()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=('qual_long','qual_fresh','noquant','formal'),required=True)
    parser.add_argument('--document',required=True)
    parser.add_argument('--condition',choices=CONDITIONS+('H_NOQUANT','L6_NOQUANT'),required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--parent-result',type=Path)
    parser.add_argument('--fp-result',type=Path)
    args=parser.parse_args()
    assert args.result.is_absolute() and args.result.resolve().is_relative_to(D)
    assert (args.kind=='noquant')==args.condition.endswith('NOQUANT'), 'NOQUANT_SCOPE_MISMATCH'
    assert not args.result.exists(), 'NO_RETRY_RESULT_EXISTS'
    try: run(args)
    except BaseException as exc:
        error={'status':'FAIL','task_id':TASK,'kind':args.kind,'document':args.document,'condition':args.condition,
               'reason':repr(exc),'traceback':traceback.format_exc(),'epoch':time.time(),'no_retry':True,
               'scope':'worker evidence only; controller owns exclusive FIRST_FAILURE identity and resource-first ordering'}
        target=args.result.with_suffix('.ERROR.json')
        if not target.exists(): publish(target,error)
        raise

if __name__=='__main__':main()
