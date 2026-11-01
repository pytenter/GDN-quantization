#!/usr/bin/env python3
"""Read-only CPU identity check. No Torch import, GPU query, or execution option."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--print-env',action='store_true');a=p.parse_args()
    if not a.check and not a.print_env:p.print_help();return
    planned={'status':'UNQUALIFIED_HISTORICAL_PORT_CANDIDATE','ALLOW_GPU_RESTORE':read(R/'ROOT.json')['ALLOW_GPU_RESTORE'],
             'environment_role':'RECORDED_HISTORICAL_TARGET_NOT_OBSERVED_CURRENT_GPU_SETTINGS',
             'deterministic':False,'intraop_threads':24,'interop_threads':24,'matmul_TF32':False,
             'cudnn_TF32':True,'cudnn_deterministic':False,
             'must_be_actual_unset':['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_DEVICE_ORDER'],
             'math_source_sha256':'90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a',
             'runtime_binary_exactness':'HISTORICAL_UNKNOWN_CURRENT_FILES_BOUND','GPU_queries':0,'GPU_dispatches':0}
    if a.print_env:print(json.dumps(planned,ensure_ascii=False,indent=2));return
    freeze=read(R/'SOURCE_FREEZE.json');seen=set()
    for x in freeze['files']:
        q=Path(x['path']);assert not q.is_absolute() and '..'not in q.parts and x['path'] not in seen
        seen.add(x['path']);q=R/q;assert q.is_file()and not q.is_symlink()and q.stat().st_size==x['bytes']and sha(q)==x['sha256'],str(q)
    remote=Path('/data/ydai/miniconda3/envs/bitdecode/bin/python').is_file()
    external='NOT_CHECKED_ON_THIS_HOST_CURRENT_REMOTE_CPU_EVIDENCE_RETAINED'
    if remote and sys.platform.startswith('linux'):
        import restoration_controller
        restoration_controller.check_inputs(R);external='PASS_CURRENT_HOST_EXTERNAL_SOURCE_MODEL_NATIVE_SHA'
    print(json.dumps(dict(status='PASS_CPU_FROZEN_LOCAL_SOURCE_IDENTITY_ONLY',source_files=len(seen),
                         external_runtime_identity=external,not_GPU_qualification=True,
                         ALLOW_GPU_RESTORE=planned['ALLOW_GPU_RESTORE'],GPU_queries=0,GPU_dispatches=0)))
if __name__=='__main__':main()
