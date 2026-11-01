#!/usr/bin/env python3
"""Separately authorized exact load-only path under GDB; original failed probe untouched."""
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'operations/model_native_exit_gdb_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb')as f:f.write(json.dumps(value,sort_keys=True,indent=2).encode()+bytes([10]));f.flush();os.fsync(f.fileno())
p=json.loads((OUT/'PERMIT.json').read_text())
assert p['script_sha256']==sha(__file__)
assert p['human_authorization_sha256']==sha(ROOT/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json')
assert p['original_probe_script_sha256']==sha(ROOT/'operations/runtime_loader_native_probe.py')
assert p['model_forward_calls_allowed']==0 and os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid']
assert not(ROOT/'SOURCE_FREEZE.json').exists()and not(ROOT/'CONTROLLER_STARTED.json').exists()
spec=importlib.util.spec_from_file_location('original_load_only_module',ROOT/'operations/runtime_loader_native_probe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.TARGET=OUT/'load_only_observation'
new('WRAPPER_STARTED.json',dict(identity=module.identity(os.getpid()),monotonic=time.monotonic(),permit_sha256=sha(OUT/'PERMIT.json')))
atexit.register(lambda:new('PYTHON_ATEXIT_ENTERED.json',dict(monotonic=time.monotonic(),pid=os.getpid())))
module.worker()
new('ORIGINAL_WORKER_RETURNED.json',dict(monotonic=time.monotonic(),pid=os.getpid(),model_forward_calls=0,QDQ_calls=0,AIME_generations=0))
