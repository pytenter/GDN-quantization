#!/usr/bin/env python3
"""Prospective metadata-only correction before the sole GPU attempt."""
import os
from pathlib import Path
import deployment_closure as A
R=A.ROOT
def main():
 A.require(not (R/'logs/observer_repair_gpu_attempt_started.json').exists(),'GPU already attempted; no amendment allowed')
 p=R/'manifests/observer_repair_pre_gpu_sha256.txt';raw=p.read_bytes();log=R/'logs/observer_repair_freeze.log';rows={}
 for line in raw.decode().splitlines():
  h,name=line.split('  ',1);rows[name]=h
  if name!=str(log):A.require(A.sha(name)==h,'non-console frozen identity mismatch '+name)
 A.require(rows[str(log)]=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','unexpected original log hash')
 A.require(log.read_text()=='PRE_GPU_GATES_AND_FREEZE_PASS_NO_MODEL_ATTEMPT\n','unexpected completed console log')
 preserved=R/'manifests/observer_repair_pre_gpu_sha256.initial_console_incident_001.txt'
 with preserved.open('xb') as f:f.write(raw)
 A.save('analysis/observer_repair_manifest_amendment_001.json',dict(status='PROSPECTIVE_METADATA_ONLY',cause='Freeze manifest accidentally included its own console log before completion line was flushed',old_manifest_SHA256=A.sha(p),preserved_initial_manifest=str(preserved),changed_entry=str(log),old_console_SHA256=rows[str(log)],completed_console_SHA256=A.sha(log),scientific_source_or_protected_artifact_changes=False,GPU_model_attempts=0,runner_patch_changed=False,action='Freeze completed console bytes and record incident; verify every other original entry unchanged'))
 rows[str(log)]=A.sha(log)
 for item in (preserved,R/'analysis/observer_repair_manifest_amendment_001.json',Path(__file__)):
  rows[str(item)]=A.sha(item)
 tmp=p.with_name(p.name+'.prospective.tmp')
 with tmp.open('x') as f:
  for name,h in sorted(rows.items()):f.write(h+'  '+name+'\n')
 os.replace(tmp,p)
 for name,h in rows.items():A.require(A.sha(name)==h,'final prospective manifest verification failed '+name)
 print('CORRECTED_PROSPECTIVE_MANIFEST_PASS '+A.sha(p),flush=True)
if __name__=='__main__':main()
