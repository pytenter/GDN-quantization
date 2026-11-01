"""GPU placement/output routing only; imports byte-identical qualified mathematics."""
import argparse
import os
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import capture_original_drive as C
PAR=C.ROOT/'amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1'
C.AMEND=PAR
import isolated_replay_engineering as E
import persistent_anchor_replay as P
import scientific_lifecycle as S
import capture_extended_drive as M

def verify_admin_freeze():
    receipt=C.read(PAR/'controller_commit_receipt.json')
    for rel,h in receipt['files'].items():assert C.sha(PAR/rel)==h,'Parallel controller freeze drift'

def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['topology','capture','replay','release','finalize','stop'],required=True)
    a.add_argument('--gpu',type=int);a.add_argument('--document',type=int);a.add_argument('--condition');a.add_argument('--failed-stage')
    v=a.parse_args();verify_admin_freeze()
    schedule=C.read(PAR/'protocol/parallel_schedule.json')
    try:
        if v.stage=='topology':
            assert v.gpu in [0,1,2,3] and os.environ['CUDA_VISIBLE_DEVICES']==str(v.gpu)
            # Unique administrative output namespace; same engineering fixture and unchanged P.execute.
            target=PAR/'engineering'/('topology_gpu%d'%v.gpu)
            target.mkdir(parents=True,exist_ok=False)
            for name in ['scripts','protocol']:(target/name).symlink_to(PAR/name,target_is_directory=True)
            (target/'long_qualification_freeze.json').symlink_to(PAR/'long_qualification_freeze.json')
            (target/'engineering').mkdir()
            (target/'engineering/capture_01').symlink_to(PAR/'engineering/capture_01',target_is_directory=True)
            P.AMEND=target
            P.execute('engineering',256,'01',None,None)
        elif v.stage=='capture':
            assert v.gpu==0 and os.environ['CUDA_VISIBLE_DEVICES']=='0'
            S.verify_freeze();M.execute('discovery',v.document)
        elif v.stage=='replay':
            assert schedule['condition_gpu'][v.condition]==v.gpu and os.environ['CUDA_VISIBLE_DEVICES']==str(v.gpu)
            S.verify_freeze();assert C.read(E.GATES/'cross_gpu_parity.json')['gate']=='PASS'
            P.execute('discovery',32768,None,v.document,v.condition)
            result=PAR/'scientific'/('doc_%02d'%v.document)/'results'/v.condition/'replay_result.json'
            import torch
            C.save(result.with_name('physical_gpu_metadata.json'),dict(physical_gpu=v.gpu,
                visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],logical_device=0,
                hardware_name=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
                result_sha256=C.sha(result),schedule_sha256=C.sha(PAR/'protocol/parallel_schedule.json')))
        elif v.stage=='release':S.seal_release(v.document)
        elif v.stage=='finalize':S.finalize()
        else:S.stop(v.failed_stage)
    except Exception as e:
        out=PAR/'analysis'/('stage_%s_doc%s_gpu%s.error.json'%(v.stage,v.document,v.gpu))
        if not out.exists():C.save(out,dict(FINAL_ACTION='FAIL_AND_STOP',exception=type(e).__name__,
             message=str(e),traceback=traceback.format_exc(),stage=v.stage,document=v.document,gpu=v.gpu,retry=False))
        raise

if __name__=='__main__':main()
