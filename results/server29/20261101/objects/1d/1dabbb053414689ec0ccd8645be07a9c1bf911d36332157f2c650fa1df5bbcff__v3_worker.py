"""Reuse the byte-identical V2 numerical worker; namespace/guard/telemetry only."""
import os
import time
import traceback
import v2_worker as inherited
import persistent_engine_v1 as engine
import v3_common as v3
from v3_guard import proc_identity


def main():
    # No old controller/guard is executed. The V2 main's numerical code and strict
    # init remain byte-identical, but its globals are explicitly bound to V3.
    for name in ('CODE','OLD','OUT','RUNS','FREEZE','LATCH','verify','checked','emit','stop','device','MODES','tag','fingerprint'):
        setattr(inherited,name,getattr(v3,name))
    args=inherited.argparse.ArgumentParser()
    args.add_argument('--gpu',type=int,required=True); args.add_argument('--document',required=True)
    args.add_argument('--mode',required=True); args.add_argument('--repeat',type=int,default=0)
    a=args.parse_args(); t=v3.tag(a.mode,a.document,a.gpu,a.repeat); started=time.time()
    samples=[]; original_kl=engine.full_kl; count=0
    try:
        cfg,_,freeze=v3.verify(); receipt=v3.checked(v3.RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert receipt['wrapper_sha256']==v3.digest(__file__) and receipt['gpu']==a.gpu
        v3.emit(v3.RUNS/(t+'.PROCESS_IDENTITY.json'),dict(identity=proc_identity(os.getpid()),gpu=a.gpu,
            gpu_uuid=receipt['gpu_uuid'],wrapper_sha256=v3.digest(__file__),receipt_sha256=v3.digest(v3.RUNS/(t+'.LAUNCH_AUTHORIZATION.json')),
            code_commit=freeze['code_commit'],epoch=time.time()))
        if a.mode=='qualification':
            def observed_kl(torch,reference,student):
                nonlocal count
                value=original_kl(torch,reference,student); position=count//2; condition_index=count%2; count+=1
                if position in (0,255,511):
                    # No model/recurrent arithmetic, return value or tolerance changes.
                    # Independent CPU FP64 reduction of the same qualified FP32 logits.
                    r=reference[:,-1,:].detach().cpu().double(); s=student[:,-1,:].detach().cpu().double()
                    cpu64=float((torch.softmax(r,-1)*(torch.log_softmax(r,-1)-torch.log_softmax(s,-1))).sum(-1).mean())
                    assert torch.isfinite(r).all() and torch.isfinite(s).all()
                    samples.append(dict(position=position,condition_index=condition_index,gpu_original_fp32_kl=value,
                        cpu_same_logits_fp64_kl=cpu64,absolute_arithmetic_difference=abs(value-cpu64),vocab=int(r.shape[-1])))
                return value
            engine.full_kl=observed_kl
        inherited.main()
        if a.mode=='qualification':
            assert len(samples)==6
            v3.emit(v3.RUNS/(t+'.REDUCER_DIAGNOSTIC.json'),dict(status='PASS',samples=samples,
                numerical_worker_or_engine_return_changed=False,cpu_diagnostic_not_FP64_model_reference=True,epoch=time.time()))
    except Exception as exc:
        fail=v3.RUNS/(t+'.STOP.json')
        if not fail.exists():
            v3.emit(fail,dict(status='STOP',error=repr(exc),traceback=traceback.format_exc(),gpu=a.gpu,
                document=a.document,mode=a.mode,pid=os.getpid(),started_epoch=started,stopped_epoch=time.time()))
        v3.stop(repr(exc),a.mode,run_stop=str(fail)); raise
    finally: engine.full_kl=original_kl

if __name__=='__main__': main()
