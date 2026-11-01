#!/usr/bin/env python3
"""Test unexecuted GPU observer code with real tiny CPU cache and raw fixtures."""
import ast,hashlib,json,math,os,sys,tempfile
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def run(worker_source,fixture_source):
    scope={'__name__':'cpu_fixture','__file__':'CPU_ONLY_STDIN_FIXTURE'}
    exec(compile(fixture_source,'real_qwen_cpu_fixture.py','exec'),scope)
    assert scope['result']['status']=='PASS_REAL_CLASS_TINY_CPU_ONLY'
    torch=scope['torch'];model=scope['model'];cache=scope['cache'];out=scope['out']
    tree=ast.parse(worker_source);functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    ns={'torch':torch,'ids':[3,7,11,13],'math':math}
    for name in ['arrays','difference']:
        node=functions[name];module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'worker_AST_'+name,'exec'),ns)
    tensors,metadata,aliases=ns['arrays'](cache,out,3)
    names=[x[0] for x in tensors]
    checks=[]
    def check(name,b,**details):
        checks.append(dict(name=name,passed=bool(b),**details));assert b,name
    check('real_tiny_cache_dict_recurrent_included',any('recurrent_states/int:0' in n for n in names),tensor_names=names)
    check('real_tiny_cache_dict_conv_included',any('conv_states/int:0' in n for n in names))
    check('real_tiny_cache_attention_KV_included',any(n.endswith('/keys') for n in names)and any(n.endswith('/values') for n in names))
    check('real_tiny_cache_metadata_class_recorded',metadata['cache']['class'].startswith('transformers.'))
    check('fresh_metadata_JSON_roundtrip_identical',json.loads(json.dumps(metadata))==metadata)
    check('all_observer_tensors_CPU',all(x.device.type=='cpu' for _,x in tensors))
    # A metadata-only corruption must be visible; metadata is part of fresh equality.
    old=cache.offloading;cache.offloading=not old
    _,changed,_=ns['arrays'](cache,out,3);cache.offloading=old
    check('metadata_only_change_detected',metadata!=changed)
    # Preserve actual byte mismatch when two values are numerically equal signed zero.
    with tempfile.TemporaryDirectory(prefix='qwen_historical_observer_CPU_') as temp:
        base=Path(temp);p=base/'a';q=base/'b';a=torch.tensor([0.,1.],dtype=torch.float32);b=torch.tensor([-0.,1.],dtype=torch.float32)
        aa=a.view(torch.uint8).numpy().tobytes();bb=b.view(torch.uint8).numpy().tobytes();p.write_bytes(aa);q.write_bytes(bb)
        def trace(raw):return {'tensors':[dict(name='fixture',dtype='torch.float32',shape=[2],stride=[1],offset=0,bytes=8,sha256=hashlib.sha256(raw).hexdigest())],'cache_metadata':{},'tensor_storage_aliases':[]}
        diff=ns['difference'](trace(aa),trace(bb),p,q)
        check('signed_zero_byte_mismatch_not_erased',diff['first_tensor_difference']['kind']=='ACTUAL_BYTES_DIFFER'and diff['first_tensor_difference']['max_absolute']==0,difference=diff)
        b[1]=1.25;bb=b.view(torch.uint8).numpy().tobytes();q.write_bytes(bb);diff=ns['difference'](trace(aa),trace(bb),p,q)
        check('actual_difference_size_quantified',diff['first_tensor_difference']['max_absolute']==.25)
    result=dict(status='PASS_TINY_REAL_CPU_CACHE_AND_RAW_OBSERVER_TESTS',checks=checks,GPU_dispatches=0,GPU_queries=0,
                worker_source_sha256=hashlib.sha256(worker_source.encode()).hexdigest(),scope='CPU_ONLY_OBSERVER_PREPARATION_NOT_GPU_QUALIFICATION')
    print('CPU_OBSERVER_RESULT='+json.dumps(result,sort_keys=True,allow_nan=False));return result
if __name__=='__main__':
    if len(sys.argv)>1:
        root=Path(sys.argv[1]);run((root/'worker/restoration_worker.py').read_text(),(root/'qualification/real_qwen_cpu_fixture.py').read_text())
