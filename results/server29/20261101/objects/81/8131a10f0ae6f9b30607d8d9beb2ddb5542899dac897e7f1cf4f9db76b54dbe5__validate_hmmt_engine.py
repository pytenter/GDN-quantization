"""Actual 9B generation/sampler and synthetic forced-EOS lifecycle, no HMMT."""
import json,sys,gc,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import pace_runtime as R
from hmmt_engine import HMMTEngine
from execution_common import atomic
from model_engineering import runtime_identity

def progress(phase,**kw):print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**kw)),flush=True)
def main():
    progress('LOAD_MODEL_HMMT_PROTOCOL_DIAGNOSTIC')
    model,tokenizer,native=R._load_frozen_model(ROOT)
    fixture=json.loads((ROOT/'protocol/non_aime_fixture.json').read_text())['tokens'][:16]
    rows=[]
    for config in ['C0','C1','C3']:
        engine=HMMTEngine(ROOT,config,model,tokenizer,native,progress)
        r=engine.generate(fixture,problem_id='NON_HMMT_DIAGNOSTIC',max_new_tokens=33,diagnostic=True)
        assert r['sampler_calls']==r['generated_token_count'] and r['actual_decode_forwards']==r['generated_token_count']-1
        assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==r['full_output']
        rows.append(dict(case='ACTUAL_SAMPLER_33_CAP',result=r))
        # Independent fresh cache/RNG. Force EOS on sample2 to test final EOS is
        # not forwarded. Only this diagnostic replaces sampler, never formal.
        original=engine.f.sample_next;calls=[0];stop=min(engine.f.stop_ids(tokenizer))
        def forced(warpers,logits,mask,generator):
            calls[0]+=1
            if calls[0]==2:return torch.tensor([[stop]],device='cuda',dtype=torch.long),1
            token,n=original(warpers,logits,mask,generator)
            if int(token.item())==stop:token=torch.tensor([[fixture[0]]],device='cuda',dtype=torch.long)
            return token,n
        engine.f.sample_next=forced
        e=engine.generate(fixture,problem_id='NON_HMMT_FORCED_EOS_DIAGNOSTIC',max_new_tokens=33,diagnostic=True)
        assert e['finish_reason']=='eos' and e['generated_token_count']==2 and e['actual_decode_forwards']==1
        if config!='C0':assert all(x['pending_tokens']==1 and x['decode_tokens']==1 and x['writeback_count']==1 for x in e['quantizer_audit']['storage_by_layer'].values())
        rows.append(dict(case='FORCED_EOS2_LAST_NOT_FORWARDED_PRESERVE',result=e))
        del engine;gc.collect();torch.cuda.empty_cache();progress('ENGINE_CONFIG_DIAGNOSTIC_PASS',config=config)
    atomic(ROOT/'validation/HMMT_ENGINE_NON_HMMT_DIAGNOSTIC.json',dict(status='PASS_NEW_ENGINE_SAMPLER_AND_TAIL',runtime=runtime_identity(),rows=rows,formal_generated=0,allowed_configs=['C0','C1','C3'],C2='DENIED_FP_H_FULL_MODEL_GATE'))
if __name__=='__main__':main()
