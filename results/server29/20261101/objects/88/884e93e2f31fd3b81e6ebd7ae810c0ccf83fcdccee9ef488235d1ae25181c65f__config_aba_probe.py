"""One predeclared A/B/A configuration intervention, engineering evidence only."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

D=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
W=Path(__file__).resolve().parent
UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'

def worker():
    import functional_worker as w
    import resume_diagnostic as diag
    from prospective_common import verify,load_inputs
    cfg=verify();gpu=w.route(cfg);assert gpu==3
    plan=w.read(D/'ABA_PLAN.json')
    assert os.environ['FLA_CACHE_MODE']=='always'
    for c in plan['configs'].values():assert w.sha(c['path'])==c['sha256']
    import torch
    runtime=[w.initialize(torch,0)];capture=audit=observer=None;started=time.time()
    try:
        identity=w.proc_identity(os.getpid());assert w.device(gpu,cfg)['exclusive']
        base=w.load_module(w.PRIOR/'run_short_horizon_proxy_v2.py','ABA_verified_base')
        direction=w.load_module(base.DIRECTION_RUNNER,'ABA_verified_direction');direction.GPU_UUIDS[gpu]=UUID
        host=direction.gpu_guard(torch,gpu);_,_,module,core=direction.load_assets(gpu)
        _,tokens,_,_,_=load_inputs(torch,cfg,'pg19-test-26183',['FP'])
        capture=module.Capture();capture.model.eval();audit=w.engine.PersistentAudit(torch,direction,core,capture,cfg['layers'],['FP'])
        audit.active=None;observer=w.HiddenObserver(torch,capture,False)
        runtime.append(w.snapshot(torch,'after_original_helpers',0))
        import fla.ops.utils.cache as cache_module
        assert cache_module.FLA_CACHE_MODE.value=='always'
        cp=plan['checkpoint'];assert w.sha(cp['path'])==cp['sha256']
        payload=torch.load(cp['path'],map_location='cpu',weights_only=False)
        torch.cuda.reset_peak_memory_stats();branches=[]
        for name,key in [('A1_BD128_W4','A'),('B_BD8_W8','B'),('A2_BD128_W4','A')]:
            os.environ['FLA_CONFIG_DIR']=str(Path(plan['configs'][key]['path']).parent)
            if hasattr(cache_module.get_fla_config_dir,'cache_clear'):cache_module.get_fla_config_dir.cache_clear()
            assert Path(cache_module.get_fla_config_dir())==Path(os.environ['FLA_CONFIG_DIR'])
            cache=diag.restore(torch,payload,capture)
            row=diag.suffix(torch,direction,capture,tokens,cache,observer,name)
            row['configuration_key']=key;row['configuration_sha256']=plan['configs'][key]['sha256']
            branches.append(row);del cache
        guard=w.device(gpu,cfg,os.getpid(),binding=identity,require_context=True,evidence=D/'config_aba.GPU_EXCLUSIVITY.jsonl')
        assert guard['exclusive'],guard['failure_conditions']
        w.publish(D/'config_aba.json',dict(status='PASS',meaning='ENGINEERING_CAUSAL_COLLECTION_ONLY',
                  numerical_qualification=False,scientific_trajectory_count=0,branches=branches,
                  host=host,runtime=runtime,checkpoint_sha256=cp['sha256'],
                  memory=[dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())],
                  started_epoch=started,completed_epoch=time.time()))
    finally:
        if observer is not None:observer.close()
        if audit is not None:audit.close()
        if capture is not None:capture.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.worker:return worker()
    from prospective_common import verify
    from gpu_admission import admit
    from recovery_state import atomic_json,lock
    from execution04_launcher import worker_environment
    held=lock(D/'controller.lock')
    try:
        cfg=verify();plan=json.loads((D/'ABA_PLAN.json').read_text())
        assert not (D/'config_aba.INTENT.json').exists(),'NO_RETRY'
        admit(cfg,D/'admission/config_aba.window.jsonl')
        base=D/'config_aba';intent=base.with_suffix('.INTENT.json')
        ident=dict(document='pg19-test-26183',rotation='FP',stage='engineering_ABA_diagnostic',target_gpu_uuid=UUID,execution=str(D))
        atomic_json(intent,dict(base=str(base),command=[sys.executable,'-B','-u',str(__file__),'--worker'],unit_identity=ident,source_commit=plan['source_commit'],tag='config_aba',stage='engineering_ABA_diagnostic'))
        p=subprocess.Popen([sys.executable,'-B','-u',str(W/'unit_runner.py'),str(intent)],env=worker_environment(os.environ,0,UUID,3))
        try:rc=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            atomic_json(D/'STOP.json',dict(status='STOP',stage='ABA_TIMEOUT',epoch=time.time(),no_retry=True));p.wait(timeout=120);raise
        assert rc==0,('ABA_DIAGNOSTIC_FAILED',rc)
    except BaseException as e:
        if not (D/'STOP.json').exists():atomic_json(D/'STOP.json',dict(status='STOP',stage='ABA_DIAGNOSTIC',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise
    finally:held.close()

if __name__=='__main__':main()
