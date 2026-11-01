"""Namespace, guard and read-only byte/reducer observers; engine unmodified."""
import hashlib
import math
import os
import time
import traceback
import v2_worker as inherited
import persistent_engine_v1 as engine
import v4_persistent_common as v4

def main():
    for name in ('CODE','OLD','OUT','RUNS','FREEZE','LATCH','verify','checked','emit','stop','device','MODES','tag','fingerprint'):
        setattr(inherited,name,getattr(v4,name))
    ap=inherited.argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--document',required=True); ap.add_argument('--mode',required=True)
    ap.add_argument('--repeat',type=int,default=0); a=ap.parse_args()
    t=v4.tag(a.mode,a.document,a.gpu,a.repeat); started=time.time()
    original_bytes,original_kl=engine.tensor_bytes,engine.full_kl
    byte_calls=kl_calls=0; prefixes={}; diagnostics=[]
    try:
        cfg,sel,freeze=v4.verify(); receipt=v4.checked(v4.RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert receipt['wrapper_sha256']==v4.digest(__file__) and a.gpu==6
        v4.emit(v4.RUNS/(t+'.PROCESS_IDENTITY.json'),dict(identity=v4.proc_identity(os.getpid()),gpu=6,gpu_uuid=v4.UUID,
            receipt_sha256=v4.digest(v4.RUNS/(t+'.LAUNCH_AUTHORIZATION.json')),epoch=time.time()))
        entry=next(e for e in sel['documents'] if e['document']==a.document)
        conditions=sorted([entry['pair']['condition_a'],entry['pair']['condition_b']]); branches=['FP']+conditions
        hashes={c:hashlib.sha256() for c in branches}
        # The byte payload is already copied by the original engine. Reuse it,
        # return it unchanged, add no model/tensor operation or CUDA allocation.
        def observed_bytes(torch,value):
            nonlocal byte_calls
            payload=original_bytes(torch,value); pos,index=divmod(byte_calls,3); byte_calls+=1
            branch=branches[index]; hashes[branch].update(pos.to_bytes(8,'little')); hashes[branch].update(payload)
            if pos in (511,8447):
                prefixes.setdefault(str(pos+1),{})[branch]=hashes[branch].hexdigest()
                if index==2:
                    source=None
                    if a.mode=='micro' and pos==511:
                        source=v4.checked(v4.RUNS/(v4.tag('qualification',a.document,6,0)+'.PREFIX_FINGERPRINTS.json'))
                    elif a.mode=='formal':
                        prior_mode='qualification' if pos==511 else 'micro'
                        source=v4.checked(v4.RUNS/(v4.tag(prior_mode,a.document,6,0)+'.PREFIX_FINGERPRINTS.json'))
                    if source is not None:
                        assert prefixes[str(pos+1)]==source['prefixes'][str(pos+1)],('SAME_GEOMETRY_PREFIX_NUMERIC_FAIL',a.document,pos+1)
            return payload
        engine.tensor_bytes=observed_bytes
        if a.mode=='qualification':
            def observed_kl(torch,reference,student):
                nonlocal kl_calls
                value=original_kl(torch,reference,student); pos,index=divmod(kl_calls,2); kl_calls+=1
                if pos in (0,255,511):
                    r=reference[:,-1,:].detach().cpu().double(); s=student[:,-1,:].detach().cpu().double()
                    cpu64=float((torch.softmax(r,-1)*(torch.log_softmax(r,-1)-torch.log_softmax(s,-1))).sum(-1).mean())
                    assert math.isfinite(cpu64)
                    diagnostics.append(dict(position=pos,condition=conditions[index],gpu_original_FP32_KL=value,
                        same_logits_CPU64_KL=cpu64,absolute_arithmetic_difference=abs(value-cpu64),vocab=int(r.shape[-1])))
                return value
            engine.full_kl=observed_kl
        inherited.main()
        done=v4.checked(v4.RUNS/(t+'.json'))
        assert byte_calls==3*done['endpoint'],'ORIGINAL_ENGINE_BYTES_COVERAGE_FAIL'
        v4.emit(v4.RUNS/(t+'.PREFIX_FINGERPRINTS.json'),dict(status='PASS',prefixes=prefixes,
            endpoint=done['endpoint'],full_hashes={c:hashes[c].hexdigest() for c in branches},
            canonical_sha256=v4.digest(v4.RUNS/(t+'.json')),engine_sha256=v4.ENGINE_SHA,
            observer_reuses_exact_returned_bytes_without_tensor_ops=True,
            shape_chunk_cache_semantics='original single-token model_step at same positions with independently fresh 3 native caches; probes at511 identical; no horizon input to model_step',
            prefix_checks_are_full_byte_exact_not_KL_ranking=True))
        assert hashes['FP'].hexdigest()==done['stream']['fp_reference_full_logit_sha256']
        if a.mode=='qualification':
            assert len(diagnostics)==6
            v4.emit(v4.RUNS/(t+'.REDUCER_DIAGNOSTIC.json'),dict(status='PASS',samples=diagnostics,
                model_or_engine_return_unchanged=True,CPU64_same_FP32_logits_not_FP64_model=True))
    except Exception as exc:
        p=v4.RUNS/(t+'.STOP.json')
        if not p.exists(): v4.emit(p,dict(status='STOP',reason=repr(exc),traceback=traceback.format_exc(),
            started_epoch=started,stopped_epoch=time.time(),document=a.document,mode=a.mode,pid=os.getpid()))
        v4.stop(repr(exc),a.mode,run_stop=str(p)); raise
    finally: engine.tensor_bytes=original_bytes; engine.full_kl=original_kl

if __name__=='__main__': main()
