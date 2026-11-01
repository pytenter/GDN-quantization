"""Pure Python AST and synthetic bookkeeping audit; no Torch/model/GPU."""
import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import time
import types

from recovery_worker_entry import load_qualified,compile_recovery_install
from resume_adapter import build_run,PHASE_A,PHASE_D


def synthetic(candidate,cursor):
    # Deliberately replace model loop only in this in-memory synthetic fixture.
    # The separate production AST proof requires the real loop body unchanged.
    node=copy.deepcopy(candidate)
    loop=next(x for x in node.body if isinstance(x,ast.With)).body[0]
    loop.body=ast.parse("""seen.append((t,token))
for layer in f.GDN_LAYERS:obs.calls[(t,layer)]+=1
if t in R.SCORE_POSITIONS:scores[t]=('SYNTHETIC_SUFFIX',t)
""").body
    saved=[];events=[];seen=[];positions=tuple(range(8192,32768,64))
    prefix={t:('SYNTHETIC_PREFIX',t) for t in positions if t<cursor}
    old_prefix=copy.deepcopy(prefix)
    phase=PHASE_A if cursor else PHASE_D
    condition='C6R_C128' if cursor else 'C6R_DUAL_M'
    class State:
        def data_ptr(self):return 123
    f=types.SimpleNamespace(GDN_LAYERS=tuple(range(24)),get_state=lambda c,l:State(),QwenKeyHadamardPatch=lambda m:contextlib.nullcontext())
    def restore(*args):
        events.append('restore')
        return object() if cursor else None,prefix,dict(cursor=cursor,prefix_score_count=len(prefix),restore_audit_status='PASS' if cursor else 'NOT_APPLICABLE_FRESH_FROM_TOKEN0',checkpoint_sha256='SYNTHETIC_ONLY')
    def record(meta,previous,tag):
        assert not seen
        assert len(previous)==(24 if cursor else 0)
        events.append('record')
    class Observe:
        def __init__(self,*args):self.calls=collections.Counter();self.previous={}
        def close(self):pass
    def check(ok,message):
        if not ok:raise AssertionError(message)
    K=types.SimpleNamespace(formal=lambda memo:f,tsha=lambda t:'0'*64,check=check,sha=lambda p:'1'*64,gpu_memory=lambda:dict(peak_allocated=0,peak_reserved=0))
    ns=dict(K=K,Observe=Observe,ROOT=Path('/SYNTHETIC_NO_FILES'),R=types.SimpleNamespace(SCORE_POSITIONS=positions),torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext,save=lambda obj,p:saved.append(obj),cuda=types.SimpleNamespace(empty_cache=lambda:None)),time=time,hashlib=hashlib,json=json,gc=types.SimpleNamespace(collect=lambda:None),_continuation_restore=restore,_continuation_record_restore=record,seen=seen)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<synthetic-only-not-production>','exec'),ns)
    result=ns['run'](None,None,condition,list(range(32768)),'SYNTHETIC_DOC',phase,False,science=True)
    assert events==['restore','record'] and seen[0]==(cursor,cursor) and seen[-1]==(32767,32767) and len(seen)==32768-cursor
    assert result['consumed_tokens']==32768 and result['new_model_tokens']==32768-cursor
    assert result['writebacks']==result['new_writebacks']==(32768-cursor)*24
    assert result['next_token_consumption_checks']==result['new_next_token_consumption_checks']==(32768-cursor-(0 if cursor else 1))*24
    assert list(saved[0]['logits'])==list(positions) and len(saved[0]['logits'])==384
    for t,x in old_prefix.items():assert saved[0]['logits'][t]==x
    assert result['prefix_score_count']==(128 if cursor else 0) and result['new_score_count']==(256 if cursor else 384)
    assert result['continuation']['prefix_diagnostics']==('NOT_RETAINED' if cursor else 'NOT_APPLICABLE')
    return dict(status='PASS_SYNTHETIC_ONLY',cursor=cursor,first_new_index=seen[0][0],last_new_index=seen[-1][0],new_tokens=len(seen),prefix_scores=len(old_prefix),new_scores=result['new_score_count'],writebacks=result['writebacks'],next_consumption_checks=result['next_token_consumption_checks'],prefix_scores_unmodified=True,restore_record_precedes_first_forward=True,no_files_written=True,no_Torch_or_model_loaded=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);a=p.parse_args()
    E=load_qualified(a.source_root);_,renamed,_=E.adapted_run();candidate,runproof=build_run(renamed);proof=compile_recovery_install(E)
    proof['synthetic_bookkeeping']=[synthetic(candidate,16384),synthetic(candidate,0)]
    proof['scope']='CPU AST equality and synthetic token-index/score/diagnostic bookkeeping only; no scientific logits or GPU qualification fabricated'
    print(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
