#!/usr/bin/env python3
"""Real repaired metadata positive check plus semantic-tamper negatives; CPU only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path.insert(0,str(ROOT/'src'))
import resident_archive as A
import resident_execution_control as C

def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def put(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2)+chr(10))
def metadata(root,name):
    p=root/name;return dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
D=load(ROOT/'operations/download_verified_delivery.py','tested_delivery_scope')
scope={};exec(D.REMOTE_HELPERS,scope)
rows,_=A.catalog(ROOT,{});by={x['path']:x for x in rows}
native=read(ROOT/'analysis/RUNTIME_NATIVE_CLOSURE.json')
result=scope['loader_probe_scope'](ROOT,by,D.sha(ROOT/'analysis/RUNTIME_NATIVE_CLOSURE.json'),len(native['files']))
assert result['status']=='PASS_REPAIRED_FULL_LOADER_TWO_FRESH_EXITS_ALL_ENGINEERING_HISTORY_PRESERVED'
assert len(result['repaired_loaders'])==2 and len(result['engineering_resource_records'])==12 and result['worker_identity']
checks=['real_repaired_loader_history_complete_and_two_exact_exits']
with tempfile.TemporaryDirectory(prefix='resident-real-delivery-scope-')as tmp:
    fixture=Path(tmp)/'metadata'
    shutil.copytree(ROOT,fixture,ignore=shutil.ignore_patterns('__pycache__','*.npz','*.npy','*.bin','*.pt','*.pth','*.safetensors','*.safetensor'))
    binding_name='operations/ENGINEERING_GPU_EVIDENCE_BINDING.json'
    original_binding=read(fixture/binding_name)
    def update_binding(row_name,newvalue):
        put(fixture/row_name,newvalue);record=metadata(fixture,row_name);b=copy.deepcopy(original_binding)
        def visit(v):
            if isinstance(v,dict):
                if v.get('path')==row_name:v.update(record)
                for x in list(v.values()):visit(x)
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(b);put(fixture/binding_name,b)
        catalog=copy.deepcopy(by);catalog[row_name]=record;catalog[binding_name]=metadata(fixture,binding_name)
        return catalog
    for field,row_name in [('actual_wait_returncode',original_binding['original_failed_completion']['path']),
                           ('model_forward_calls',original_binding['active_completion']['path'])]:
        original_bytes=(fixture/row_name).read_bytes();v=read(fixture/row_name);v[field]=0 if field=='actual_wait_returncode'else 1
        changed=update_binding(row_name,v)
        try:scope['loader_probe_scope'](fixture,changed,D.sha(fixture/'analysis/RUNTIME_NATIVE_CLOSURE.json'),len(native['files']))
        except(AssertionError,KeyError):checks.append('semantic_tamper_rejected_'+field)
        else:raise AssertionError('Semantic change incorrectly admitted:'+field)
        (fixture/row_name).write_bytes(original_bytes);put(fixture/binding_name,original_binding)
finisher=load(ROOT/'scripts/resident_finalize_offline.py','actual_cost_scope_function')
cost=finisher.engineering_probe_scope(C,ROOT)
assert cost['engineering_dispatch_count']==12 and cost['qualification_units_contributed']==cost['formal_samples_contributed']==0
assert cost['EXACT_physical_GPUh'].startswith('UNRESOLVED')and cost['reservation_GPUh_upper']>0
checks.append('all12_engineering_intervals_separate_no_scientific_credit')
put(ROOT/'analysis/ACTUAL_REPAIRED_DELIVERY_SCOPE_CPU_REVIEW.json',dict(status='PASS_REAL_REPAIRED_SCOPE_AND_SEMANTIC_TAMPER_CPU_CHECKS',
    checks=checks,count=len(checks),current_downloader_sha256=D.sha(ROOT/'operations/download_verified_delivery.py'),
    current_finisher_sha256=D.sha(ROOT/'scripts/resident_finalize_offline.py'),real_scope=result,engineering_cost_function=cost,
    no_SSH_commands=True,no_GPU_dispatch=True,no_Torch_import='torch'not in sys.modules,negative_fixture_outside_science_root=True))
print(json.dumps(dict(status='PASS_REAL_REPAIRED_SCOPE_AND_SEMANTIC_TAMPER_CPU_CHECKS',count=len(checks))))
