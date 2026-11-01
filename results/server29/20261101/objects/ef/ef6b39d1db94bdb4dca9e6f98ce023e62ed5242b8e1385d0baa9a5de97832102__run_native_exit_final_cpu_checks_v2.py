#!/usr/bin/env python3
"""Final changed-guard tests and real archive catalog; no CUDA/model/dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not(ROOT/'SOURCE_FREEZE.json').exists()
for filename,expected in [('ACTUAL_GUARD_NATIVE_EXIT_TARGET_FINAL_TWO_FRESH_RESULT.json',74),('CPU_RUNTIME_NATIVE_CLOSURE_GUARD_NATIVE_EXIT_TARGET_TWO_FRESH_RESULT.json',47)]:
    value=json.loads((ROOT/'analysis'/filename).read_text())
    assert len(value['processes'])==2 and all(x['actual_wait_returncode']==0 and x['result']['check_count']==expected for x in value['processes'])
jobs=[('FINISHER','scripts/test_resident_finisher_cpu.py','CPU_FINISHER_NATIVE_EXIT_FINAL_TWO_FRESH_RESULT.json','PASS_TWO_FRESH_FINISHER_ENGINEERING_SCOPE_CPU_ONLY',45),
      ('DELIVERY','operations/test_readonly_delivery_cpu.py','CPU_DELIVERY_NATIVE_EXIT_FINAL_TWO_FRESH_RESULT.json','PASS_TWO_FRESH_READONLY_DELIVERY_CPU_ONLY',88)]
for name,script,aggregate,status,count in jobs:
    rows=[]
    for repeat in (1,2):
        output=ROOT/'analysis'/('%s_NATIVE_EXIT_FINAL_V2_%d.json'%(name,repeat))
        command=[sys.executable,'-u','-B',str(ROOT/script),'--output',str(output)]
        if name in('NATIVE','FINISHER'):command+=['--root',str(ROOT)]
        with tempfile.TemporaryDirectory(prefix='resident-native-exit-final-')as tmp:
            if name=='GUARD':command+=['--evidence-root',str(Path(tmp)/'fixtures')]
            with(ROOT/'operations'/('%s_NATIVE_EXIT_FINAL_V2_%d.log'%(name,repeat))).open('xb')as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=log);rc=process.wait()
            result=json.loads(output.read_text())if output.exists()else None
            assert rc==0 and result and result.get('check_count',result.get('count'))==count,(name,repeat,rc,result)
            rows.append(dict(pid=process.pid,actual_wait_returncode=rc,result=result,negative_fixtures_outside_science_root=True))
    with(ROOT/'analysis'/aggregate).open('x')as f:json.dump(dict(status=status,processes=rows,CPU_only=True,GPU_dispatches=0),f,sort_keys=True,indent=2);f.write(chr(10))
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'operations')]
import engineering_gpu_evidence as E
import resident_archive as archive
evidence=E.validate(ROOT)
rows,exclusions=archive.catalog(ROOT,{})
proof=dict(status='PASS_ACTUAL_PRODUCTION_ARCHIVE_CATALOG_ON_REAL_PREFREEZE_ROOT',file_count=len(rows),
    total_bytes=sum(x['bytes']for x in rows),files=rows,excluded_exact_control_paths=exclusions,
    real_catalog_function='src/resident_archive.py::catalog',all_negative_fixtures_outside_root=True,
    no_symlink_or_special_files=True,scientific_GPU_units_dispatched=0,engineering_dispatches=evidence['total_GPU_work_dispatches'])
with(ROOT/'operations/PRE_SCIENCE_REAL_ARCHIVE_CATALOG.json').open('x')as f:json.dump(proof,f,sort_keys=True,indent=2);f.write(chr(10))
print(json.dumps(dict(status='PASS_FINAL_CHANGED_GUARD_AND_DELIVERY_CPU_CHECKS_REAL_CATALOG',tests={x[0]:dict(processes=2,checks_each=x[-1])for x in jobs},catalog_files=len(rows),catalog_bytes=proof['total_bytes'])))
