"""Synthetic control-only tests. No scientific files, Torch, model, CUDA or NVML."""
import ast,copy,hashlib,importlib.util,json,math,os,pathlib,statistics,sys,tempfile,types
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'control/worker_entry.py'
    spec=importlib.util.spec_from_file_location('c6r_entry_audit',path);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    proof=e.audit();proof['entry_SHA256']=e.sha(path)
    tree=ast.parse(path.read_text());install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    functions=[n for n in install.body if isinstance(n,ast.FunctionDef) and n.name in ('initial','fresh')]
    assert len(functions)==2
    saved={};calls=[]
    def check(ok,message):
        if not ok:raise RuntimeError(message)
    def run(model,qmod,condition,ids,doc,tag,on,reference=False,**kwargs):
        assert condition in e.NAMES and len(ids)==128
        calls.append((condition,tag,on,reference,kwargs.get('restore_test',False)))
        return dict(status='PASS',condition=condition,tag=tag,rows=[dict(t=i,condition=condition) for i in range(128)],performance=[dict(full_model_plus_QDQ_seconds=0.1) for _ in range(128)],checkpoints=[dict(bytes=1,wall_seconds=0.1)] if tag=='A_ON' else [],peak_allocated=1,peak_reserved=2)
    def save(rel,obj):
        assert rel not in saved
        saved[rel]=obj
    K=types.SimpleNamespace(save=save,check=check,read=lambda p:saved[str(p.relative_to(ROOT))],UUID=e.UUIDS[0])
    W=types.SimpleNamespace(canary_ids=lambda:list(range(128)),run=run,comparable=lambda rows,helpers=False:rows,math=math,statistics=statistics,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda gpu:types.SimpleNamespace(total_memory=24*1024**3))))
    ns=dict(W=W,K=K,ROOT=ROOT,output_root=ROOT,physical=0,bank_cache={'identity':{'SYNTHETIC_TEST_ONLY':True}})
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path),'exec'),ns)
    assert ns['initial'](None,None)['status']=='PASS'
    assert ns['fresh'](None,None)['status']=='PASS'
    assert [r[1] for r in calls]==['A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','A_FRESH','D_FRESH']
    assert sum(r[3] for r in calls)==1 and sum(r[4] for r in calls)==1
    assert not any(r[0] in ('FP_NATIVE','H_C128','H_DUAL_M') for r in calls)
    assert set(saved['C6R_CANARY_RUNTIME.json']['throughput'])==set(e.NAMES)
    assert all(saved[p]['status']=='PASS' for p in ('CANARY_INITIAL.json','CANARY_FRESH.json','C6R_CANARY_RUNTIME.json'))
    # The identity comparison must stop, not record a successful initial receipt.
    saved.clear();calls.clear();W.comparable=lambda rows,helpers=False:(rows if not helpers else object())
    try:ns['initial'](None,None)
    except RuntimeError as ex:assert 'exact128' in str(ex)
    else:raise AssertionError('Injected reference mismatch did not fail closed')
    assert 'CANARY_INITIAL.json' not in saved
    # Resolve the existing sibling through the explicit local workspace, not inference from remote names.
    old=pathlib.Path('/Users/pytenter/Documents/临时项目1/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/control/worker_entry_4gpu.py')
    assert e.sha(old)==e.OLD4_ENTRY_SHA
    oldspec=importlib.util.spec_from_file_location('original_four_gpu_entry',old);o=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(o)
    configure=next(n for n in ast.parse((ROOT/'src/core.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='configure')
    source=ast.get_source_segment((ROOT/'src/core.py').read_text(),configure)
    device=[]
    for gpu in range(4):
        _,record=o.configure_ast(source,gpu);device.append(dict(physical_GPU_ID=gpu,GPU_UUID=e.UUIDS[gpu],**record))
    # The additive secondary observer preserves exact original objects and
    # delegates the original encode once; exceptions restore the function.
    class ScalarFixture:
        def __init__(self,lo,hi,dtype):self.lo,self.hi,self.dtype=lo,hi,dtype
        def min(self):return self.lo
        def max(self):return self.hi
    encoded=types.SimpleNamespace(rows=ScalarFixture(1,2,'fixture.float16'),columns=ScalarFixture(3,4,'fixture.float16'))
    raw=[ScalarFixture(5,6,'fixture.float32'),ScalarFixture(7,8,'fixture.float32'),ScalarFixture(9,10,'fixture.float32')]
    trace_calls=[];dual_calls=[]
    def trace(*args,**kwargs):trace_calls.append((args,kwargs));return encoded,raw
    codec=types.SimpleNamespace(trace_encode=trace)
    def dual(*args,**kwargs):
        dual_calls.append((args,kwargs))
        if kwargs.get('metrics',False) or len(args)>3 and args[3]:
            a,b=codec.trace_encode('FIXTURE','DUAL_M_INT8');assert a is encoded and b is raw
        return {},{'original_stat':True},{'original_timing':True}
    fake=types.SimpleNamespace(dual=dual,C=codec);e.install_scale_observer(fake)
    out=fake.dual(None,None,metrics=False)
    assert not trace_calls and out==({},dict(original_stat=True),dict(original_timing=True))
    out=fake.dual(None,None,metrics=True)
    assert len(trace_calls)==1 and len(dual_calls)==2 and codec.trace_encode is trace
    assert out[1]['encoded_row_scale_min']==1.0 and out[1]['encoded_column_scale_max']==4.0 and out[1]['raw_second_column_scale_max']==10.0
    assert out[0]=={} and out[2]==dict(original_timing=True) and all(not isinstance(v,ScalarFixture) for v in out[1].values())
    def broken(*args,**kwargs):
        codec.trace_encode('FIXTURE','DUAL_M_INT8');raise RuntimeError('SYNTHETIC_FAILURE')
    fake=types.SimpleNamespace(dual=broken,C=codec);e.install_scale_observer(fake)
    try:fake.dual(None,None,metrics=True)
    except RuntimeError as ex:assert str(ex)=='SYNTHETIC_FAILURE'
    else:raise AssertionError('Observer swallowed original failure')
    assert codec.trace_encode is trace
    # Stop guards execute before NVML, importing Torch or initializing CUDA.
    prior_root,prior_audit,visible=e.ROOT,e.audit,os.environ.get('CUDA_VISIBLE_DEVICES')
    with tempfile.TemporaryDirectory() as td:
        e.ROOT=pathlib.Path(td);e.audit=lambda:dict(status='SYNTHETIC')
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        (e.ROOT/'FIRST_FAILURE.json').write_text('{}')
        try:e.install(0,'canary_initial')
        except AssertionError as ex:assert 'Master first failure' in str(ex)
        else:raise AssertionError('Master stop bypass')
        (e.ROOT/'FIRST_FAILURE.json').unlink();lane=e.ROOT/'qualification_gpu/gpu0';lane.mkdir(parents=True)
        (lane/'FIRST_FAILURE.json').write_text('{}')
        try:e.install(0,'canary_initial')
        except AssertionError as ex:assert 'Lane first failure' in str(ex)
        else:raise AssertionError('Lane stop bypass')
    e.ROOT,e.audit=prior_root,prior_audit
    if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    else:os.environ['CUDA_VISIBLE_DEVICES']=visible
    proof.update(secondary_scale_observer_synthetic='PASS_original_objects_unchanged_single_encode_scalar_only',secondary_scale_observer_failure_restoration='PASS',master_and_lane_first_failure_before_NVML_Torch='PASS',secondary_scalar_observation_scope='Actual encoded/raw scales at original metrics positions only; extra synchronization/time charged as observer cost; requires actual GPU OFF/ON/fresh parity qualification')
    proof.update(synthetic_only=True,canary_plan_tags=['A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','A_FRESH','D_FRESH'],canary_initial_fresh_mock='PASS',injected_reference_mismatch_fail_closed='PASS',original_4gpu_adapter_SHA256=e.OLD4_ENTRY_SHA,device_mapping_CPU_AST=device,original_checkpoint_hook_self_test=o.self_test(),ten_science_phases=len([p for p in e.PHASES if p.startswith('science_')]),canary_source_not_formal_5docs='Original W.canary_ids() SHA-bound 128 input retained; full roster overlap independently checked by Phase0',no_formal_scientific_outputs_written=True)
    return proof

if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2,allow_nan=False))
