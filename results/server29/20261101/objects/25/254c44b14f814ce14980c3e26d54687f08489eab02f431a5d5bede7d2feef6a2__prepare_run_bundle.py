#!/usr/bin/env python3
"""CPU-only staging into a NEW local/remote filesystem root; no network or GPU.
Run in the copied task folder on55, targeting a fresh gpu_runs/CUBLAS_CONFIG_V1
or a new root. The sealed CPU delivery is never used as mutable execution root.
"""
import argparse,json,shutil,sys
from pathlib import Path
from run_cublas_controller import local_identity,sha,once,TASK

def prepare(source,destination):
 source=source.resolve();destination=destination.resolve()
 manifest=local_identity(source)
 if destination==source or destination.is_relative_to(source):raise RuntimeError('unsafe execution destination')
 destination.mkdir(parents=True,exist_ok=False)
 for row in manifest['execution_sources']:
  src=source/row['path'];target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
  if sha(target)!=row['sha256'] or target.stat().st_size!=row['bytes']:raise RuntimeError('staged identity mismatch')
 shutil.copy2(source/'SOURCE_MANIFEST.json',destination/'SOURCE_MANIFEST.json')
 once(destination/'RUN_COPY_ORIGIN.json',{'task_id':TASK,'status':'CPU_PREPARED_NOT_AUTHORIZED','source_root':str(source),'source_manifest_sha256':sha(source/'SOURCE_MANIFEST.json'),'GPU_dispatches':0})
 once(destination/'RESOURCE_LEDGER.json',{'task_id':TASK,'ALLOW_GPU_DIAG':'NO','NEW_GPUh':0,'GPU_dispatches':0,'events':[],'resource_status':'NOT_RUN_NO_GPU_AUTHORIZATION'})
 local_identity(destination)
 return {'status':'CPU_RUN_BUNDLE_PREPARED_NOT_AUTHORIZED','destination':str(destination),'files':len(manifest['execution_sources']),'GPU_queries':0,'GPU_dispatches':0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--destination',type=Path,required=True);a=ap.parse_args();print(json.dumps(prepare(a.source,a.destination),ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
