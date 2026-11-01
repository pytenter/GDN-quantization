#!/usr/bin/env python3
"""Pure CPU source/preparation identity check. Never imports Torch or CUDA."""
import argparse,hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
from run_cublas_controller import local_identity,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root.resolve()
 m=local_identity(r);c=json.loads((r/'analysis/STRICT_BASELINE_CONTRACT.json').read_text())
 packages={}
 for n in ['torch','transformers','numpy','safetensors']:
  try:packages[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:packages[n]='NOT_INSTALLED_ON_THIS_CPU_HOST'
 out={'status':'PASS_LOCAL_FROZEN_SOURCE_IDENTITY_ONLY','task_id':m['task_id'],'root':str(r),
 'source_manifest_sha256':sha(r/'SOURCE_MANIFEST.json'),'contract_sha256':sha(r/'analysis/STRICT_BASELINE_CONTRACT.json'),
 'python_this_CPU_host':sys.version,'platform_this_CPU_host':platform.platform(),'packages_this_CPU_host':packages,
 'not_the_remote_GPU_runtime_environment':True,'GPU_queries':0,'GPU_dispatches':0,
 'remote_dependency_live_qualification':'NOT_RUN_BY_THIS_LOCAL_CHECK','Fast_mode':'DENIED_NO_VALIDATION'}
 print(json.dumps(out,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
