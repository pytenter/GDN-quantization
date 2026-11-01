"""Narrow C6R context/condition adapter over byte-frozen scientific sources.

No GPU operation occurs at import or during --audit. Real phases are admitted by
the independent controller; this entry never retries or launches another phase.
"""
import argparse
import ast
import copy
import fcntl
import functools
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA = {
    'core.py': 'aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575',
    'worker.py': 'e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773',
    'codec.py': 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f',
    'reducer.py': '7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f',
}
NAMES = ('C6R_C128', 'C6R_DUAL_M')
UUIDS = {0:'GPU-a78176ed-f97f-c54f-6201-5a01b570f422',1:'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',2:'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',3:'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}
ASSIGNMENT = {1:0,2:1,3:2,4:3,5:0}
OLD4_ENTRY = pathlib.Path('/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/control/worker_entry_4gpu.py')
OLD4_ENTRY_SHA = '7b017150183e6d9f3174e5274aad6253ee976a9b65426ee7766dfbdc75bc6bab'
PHASES = ('canary_initial', 'canary_fresh') + tuple('science_doc%d_%s' % (d,c) for d in range(1,6) for c in NAMES)
FORMAL_SHA = '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
ROTATION_SHA = 'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6'
MANIFEST_SHA = 'f365dbab44e31b127abc5f9cc37916e432c4f3c88cc7706c82886812686d1103'
ROTATION = pathlib.Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1/rotations/C6R/frozen_rotation_fp32.pt')
MANIFEST = ROTATION.parents[2]/'configs/c6r_rotation_manifest.json'
EXTRA = {'file_sha256','tensor_sha256','FrozenRotationLayer','FrozenRotationBank','load_frozen_rotation','QwenFrozenRotationPatch'}
CONDITION_REPLACEMENTS = {'H_C128':'C6R_C128','H_DUAL_M':'C6R_DUAL_M'}

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class ReplaceConditions(ast.NodeTransformer):
    def __init__(self, mapping): self.mapping=mapping; self.counts={k:0 for k in mapping}
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in self.mapping:
            self.counts[node.value]+=1
            return ast.copy_location(ast.Constant(self.mapping[node.value]),node)
        return node

def adapted_run():
    tree=ast.parse((ROOT/'src/worker.py').read_text())
    old=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    change=ReplaceConditions(CONDITION_REPLACEMENTS)
    new=change.visit(copy.deepcopy(old))
    inverse=ReplaceConditions({v:k for k,v in CONDITION_REPLACEMENTS.items()}).visit(copy.deepcopy(new))
    assert change.counts=={'H_C128':1,'H_DUAL_M':1}, change.counts
    assert ast.dump(inverse,include_attributes=False)==ast.dump(old,include_attributes=False)
    return old,new,change.counts

def audit():
    actual={n:sha(ROOT/'src'/n) for n in SOURCE_SHA}
    assert actual==SOURCE_SHA, 'Frozen scientific source bytes changed'
    old,new,counts=adapted_run()
    return dict(status='PASS',scope='CPU source and adapter control audit only; no model, CUDA, or GPU qualification',GPU_calls=0,scientific_sources=actual,run_AST_original_SHA256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),run_AST_candidate_SHA256=hashlib.sha256(ast.dump(new,include_attributes=False).encode()).hexdigest(),run_inverse_condition_AST_exact=True,only_run_changes=counts,allowed_phases=list(PHASES),formal_source_SHA256=FORMAL_SHA,rotation_SHA256=ROTATION_SHA,deployment_manifest_SHA256=MANIFEST_SHA,rotation_transform='original normalize_qk; value.float().matmul(H128).matmul(Delta), separate products',GDN_operator_changes=False,quantizer_algorithm_changes=False,FP_H_forwards_authorized_or_dispatched=False,secondary_scale_observer='Original Dual-M function and encode called unchanged; encoded/raw scale scalars only at original metrics positions; additional reductions/synchronizations charged as observation cost; GPU qualification pending')

def install_scale_observer(W):
    """Scalar-only secondary observation at the original metrics positions.

    The exact original dual function still performs every encode/decode,
    safety decision and writeback. No tensor or new scale is returned to it.
    The additional reductions/synchronizations are charged as observer cost.
    """
    original=W.dual
    codec=W.C
    def observed(*args,**kwargs):
        metrics=args[3] if len(args)>3 else kwargs.get('metrics',False)
        if not metrics:return original(*args,**kwargs)
        trace=codec.trace_encode
        measured=[]
        def collect(*a,**kw):
            encoded,raw=trace(*a,**kw)
            assert len(raw)==3, 'Original Dual-M scale trace must have three stages'
            started=time.monotonic()
            measured.append(dict(encoded_row_scale_min=float(encoded.rows.min()),encoded_row_scale_max=float(encoded.rows.max()),encoded_row_scale_dtype=str(encoded.rows.dtype),encoded_column_scale_min=float(encoded.columns.min()),encoded_column_scale_max=float(encoded.columns.max()),encoded_column_scale_dtype=str(encoded.columns.dtype),raw_row_scale_min=float(raw[0].min()),raw_row_scale_max=float(raw[0].max()),raw_first_column_scale_min=float(raw[1].min()),raw_first_column_scale_max=float(raw[1].max()),raw_second_column_scale_min=float(raw[2].min()),raw_second_column_scale_max=float(raw[2].max()),scalar_scale_observation_seconds=time.monotonic()-started,scale_observer_scope='Secondary scalars from original encoded/raw scales at unchanged metrics positions; additional reduction/synchronization charged to observation cost; no tensor retained'))
            return encoded,raw
        codec.trace_encode=collect
        try:result=original(*args,**kwargs)
        finally:codec.trace_encode=trace
        assert len(measured)==1, 'Original Dual-M encode trace must execute exactly once'
        result[1].update(measured[0])
        return result
    W.dual=observed

def install(physical,phase):
    proof=audit()
    assert not (ROOT/'FIRST_FAILURE.json').exists(), 'Master first failure prohibits any later GPU entry'
    assert physical in UUIDS and os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical)
    if phase not in ('canary_initial','canary_fresh'):
        assert ASSIGNMENT[int(phase.split('_',2)[1][3:])]==physical
    output_root=ROOT/'qualification_gpu'/('gpu%d'%physical) if phase in ('canary_initial','canary_fresh') else ROOT
    assert not (output_root/'FIRST_FAILURE.json').exists(), 'Lane first failure prohibits any later GPU entry'
    for rel in ('source_manifest.json','preregistration.json'):
        assert sha(output_root/rel)==sha(ROOT/rel),'Lane/master frozen identity mismatch'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    matching=[[v.strip() for v in row.split(',')] for row in rows.splitlines() if row.split(',')[0].strip()==str(physical)]
    assert len(matching)==1 and matching[0][1]==UUIDS[physical] and 'RTX 3090' in matching[0][2] and matching[0][3]=='24576','Physical GPU UUID/type/capacity mismatch'
    sys.path.insert(0,str(ROOT/'src'))
    import core as K
    import worker as W
    assert K.ROOT==ROOT and W.ROOT==ROOT
    assert sha(OLD4_ENTRY)==OLD4_ENTRY_SHA,'Original qualified four-GPU metadata adapter changed'
    helpers=K.extract(OLD4_ENTRY,{'need','configure_ast','install_checkpoint_hook'},dict(ast=ast,copy=copy,hashlib=hashlib,Path=pathlib.Path,sys=sys,functools=functools,fcntl=fcntl,time=time))
    candidate,device_audit=helpers.configure_ast(inspect.getsource(K.configure),physical)
    K.ROOT=W.ROOT=output_root;K.UUID=UUIDS[physical]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),str(ROOT/'src/core.py'),'exec'),K.__dict__)
    mapped=K.configure
    def configure():
        K.check(K.torch.cuda.device_count()==1,'Worker must expose exactly one logical device')
        mapped()
        K.save('analysis/DEVICE_ADAPTER_'+phase+'.json',dict(device_audit,status='PASS',physical_GPU_ID=physical,GPU_UUID=K.UUID,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],logical_device='cuda:0',source_manifest_sha256=K.sha(ROOT/'source_manifest.json'),historical_device_adapter_SHA256=OLD4_ENTRY_SHA))
    K.configure=configure
    K.CONDITIONS=NAMES
    source=K.OLD/'runtime_sources/formal_c128_runner.py'
    assert sha(source)==FORMAL_SHA
    assert sha(ROTATION)==ROTATION_SHA and sha(MANIFEST)==MANIFEST_SHA
    old_formal=K.formal
    bank_cache={}
    patch_audits=[]

    def formal(memoize=False):
        f=old_formal(memoize)
        # The original class uses exactly the H factory returned by old_formal.
        # Candidate memoizes H only; reference reconstructs original H each call.
        ns=vars(f).copy(); ns.update(hashlib=hashlib,json=json)
        original=K.extract(source,EXTRA,ns)
        if not bank_cache:
            bank,identity=original.load_frozen_rotation(ROTATION,MANIFEST,'cuda:0','C6R')
            K.check(identity['rotation_sha256']==ROTATION_SHA and identity['deployment_manifest_sha256']==MANIFEST_SHA,'C6R frozen rotation identity failed')
            K.check(identity['direct_frozen_load'] is True and identity['theta_reconstruction'] is False,'C6R direct loading semantics failed')
            for p in bank.parameters():p.requires_grad_(False)
            bank_cache.update(bank=bank,identity=identity)
        def context(model):
            patch=original.QwenFrozenRotationPatch(model,bank_cache['bank'],bank_cache['identity'])
            patch_audits.append(patch)
            return patch
        f.QwenKeyHadamardPatch=context
        f.C6R_identity=bank_cache['identity']
        return f

    K.formal=formal
    _,new,_=adapted_run()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new],type_ignores=[])),str(ROOT/'src/worker.py'),'exec'),W.__dict__)
    old_run=W.run

    def run(*args,**kwargs):
        condition=args[2] if len(args)>2 else kwargs['condition']
        K.check(condition in NAMES,'C6R adapter forbids FP/H/new conditions')
        first=len(patch_audits)
        result=old_run(*args,**kwargs)
        patches=patch_audits[first:]
        expected=result['consumed_tokens']*24
        K.check(len(patches)==1 and type(patches[0]).__name__=='QwenFrozenRotationPatch','Original C6R context did not execute exactly once')
        K.check(patches[0].calls==expected and patches[0].layers==set(K.formal(False).GDN_LAYERS),'C6R rotation did not execute every layer/token')
        result.update(C6R=dict(bank_cache['identity']),rotation_calls=patches[0].calls,rotation_layers=sorted(patches[0].layers),rotation_phase_counts=patches[0].phase_counts,adapter_run_inverse_AST_exact=True)
        # No model cache/tensor snapshots are retained by this metadata audit.
        del patch_audits[first:]
        return result

    W.run=run
    install_scale_observer(W)
    old_save=K.save
    def save(rel,obj):
        if rel.startswith('analysis/ENV_'):
            obj=dict(obj,C6R=dict(bank_cache['identity']),actual_conditions=list(NAMES),rotation_adapter=proof,physical_GPU_ID=physical,logical_device='cuda:0',output_root=str(output_root))
        return old_save(rel,obj)
    K.save=save

    def initial(model,qmod):
        ids=W.canary_ids(); results={}
        plan=[('C6R_C128','A_REFERENCE',True,True),('C6R_C128','A_OFF',False,False),('C6R_C128','A_ON',True,False),('C6R_DUAL_M','D_OFF',False,False),('C6R_DUAL_M','D_ON',True,False)]
        for condition,tag,on,reference in plan:
            results[tag]=W.run(model,qmod,condition,ids,'OLD_FROZEN_CANARY_DOC_01',tag,on,reference,restore_test=tag=='A_ON')
            K.save('analysis/'+tag+'.json',results[tag])
            if tag=='A_ON':
                K.check(W.comparable(results['A_REFERENCE']['rows'],True)==W.comparable(results['A_ON']['rows'],True),'Original C6R/C128 unmemoized vs memoized executor exact128 parity failed')
                K.check(W.comparable(results['A_OFF']['rows'])==W.comparable(results['A_ON']['rows']),'C6R C128 observer OFF/ON changed outputs')
            if tag=='D_ON':K.check(W.comparable(results['D_OFF']['rows'])==W.comparable(results['D_ON']['rows']),'C6R Dual-M observer OFF/ON changed outputs')
        result=dict(status='PASS',tags=list(results),C6R_original_reference='BYTE_EXACT_ALL_128',canonical_C128_direct_function=True,observer_A_D='BYTE_EXACT',checkpoint_resume='BYTE_EXACT_AT_64_NO_REPEATED_PREFIX',DualM_official_all8_ones='PASS_TORCH_EQUAL_CODES_SCALES_DEQUANT',FP_H_model_forwards=0,C6R=dict(bank_cache['identity']))
        K.save('CANARY_INITIAL.json',result)
        return result

    def fresh(model,qmod):
        ids=W.canary_ids(); summaries={}
        for condition,tag,old in [('C6R_C128','A_FRESH','A_ON'),('C6R_DUAL_M','D_FRESH','D_ON')]:
            result=W.run(model,qmod,condition,ids,'OLD_FROZEN_CANARY_DOC_01',tag,True)
            K.check(W.comparable(result['rows'],True)==W.comparable(K.read(output_root/('analysis/'+old+'.json'))['rows'],True),'C6R same-GPU fresh-process exact128 parity failed')
            K.save('analysis/'+tag+'.json',result)
        for condition,tag in [('C6R_C128','A_OFF'),('C6R_DUAL_M','D_OFF')]:
            result=K.read(output_root/('analysis/'+tag+'.json')); times=[v['full_model_plus_QDQ_seconds'] for v in result['performance'][16:]]
            p95=sorted(times)[W.math.ceil(len(times)*.95)-1]
            summaries[condition]=dict(sample_count=len(times),median_seconds=W.statistics.median(times),p95_seconds=p95,tokens_per_second=1/W.statistics.median(times),peak_allocated=result['peak_allocated'],peak_reserved=result['peak_reserved'],load_included_elsewhere=True)
        kv32=2*8*4*256*32768*2
        projected=max(v['peak_reserved'] for v in summaries.values())+kv32+512*1024**2
        admissible=min(24*1024**3,W.torch.cuda.get_device_properties(0).total_memory)-256*1024**2
        runtime=dict(status='PASS',throughput=summaries,actual_checkpoint=K.read(output_root/'analysis/A_ON.json')['checkpoints'][0],physical_GPU_ID=physical,GPU_UUID=K.UUID,FP_H_model_forwards=0,old_N2_G4_NOT_REUSED=True,projected_peak_reserved_bytes=projected,admissible_memory_bytes=admissible,KV_cache_32K_bytes=kv32,additional_memory_margin_bytes=512*1024**2,device_total_reserve_bytes=256*1024**2,memory_projection_rule='Original max observed canary peak_reserved + original1GiB32K KV +512MiB; below min(24GiB,actual total)-256MiB; not an empirical32K proof',budget_gate='CONTROLLER_CUMULATIVE_20GPUH_PROTECTIVE_19_5GPUH_REQUIRED',scope='128-token resource observation; not a full32K capacity or cross-device bound',C6R=dict(bank_cache['identity']))
        K.save('C6R_CANARY_RUNTIME.json',runtime)
        result=dict(status='PASS',fresh_process='BYTE_EXACT_C6R_C128_DUAL_M_ALL128',FP_H_model_forwards=0,C6R=dict(bank_cache['identity']))
        K.save('CANARY_FRESH.json',result)
        return result

    W.initial=initial; W.fresh=fresh
    waits=[];hooked=[]
    helpers.install_checkpoint_hook(ROOT/'src/worker.py',ROOT/'control/CHECKPOINT_IO.lock',waits,lambda:hooked.append(True))
    return W,K,output_root,waits,hooked

def main():
    if sys.argv[1:]==['--audit']:
        print(json.dumps(audit(),ensure_ascii=False,indent=2,allow_nan=False));return
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=range(4),required=True);parser.add_argument('phase',choices=PHASES)
    args=parser.parse_args()
    W,K,output_root,waits,hooked=install(args.gpu,args.phase)
    sys.argv=[str(ROOT/'src/worker.py'),args.phase]
    try:
        W.main()
        K.check(hooked==[True] and sys.gettrace() is None,'Entry main-call hook not removed exactly once')
    finally:
        sys.settrace(None)
        K.save('analysis/CHECKPOINT_LOCK_'+args.phase+'.json',dict(status='MEASURED',phase=args.phase,physical_GPU_ID=args.gpu,GPU_UUID=K.UUID,historical_device_adapter_SHA256=OLD4_ENTRY_SHA,checkpoint_function_body_unchanged=True,lock_scope='Complete original snapshot/save/replace/hash/catalog/optional-restore function',global_retention='ONE task-wide rolling CURRENT; retired versions not retained per lane',main_call_hook_count=len(hooked),entry_trace_removed_before_main=True,calls=waits,total_lock_wait_seconds=sum(r['lock_wait_seconds'] for r in waits)))

if __name__=='__main__':main()
