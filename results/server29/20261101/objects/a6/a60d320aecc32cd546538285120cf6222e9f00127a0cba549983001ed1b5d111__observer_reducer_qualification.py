"""Qualification of exact production scale streaming, observer and KL reducers."""
import gc,json,math,sys,time,traceback
import pilot_engineering as P
import anchor_qualification as A
import formal_worker as W

def run(index):
    F,core,policy,model,tokenizer,started=P.setup();ids=P.N.read(policy['documents'][0]['token_path'])[:36]
    h=P.Harness(model,F,core,P.torch);clean=[];logits=[];cache=None;chunks=[]
    with P.torch.inference_mode():
        for p,token in enumerate(ids):
            out,cache=h.forward(token,cache,p,mode='CLEAN_FP_REFERENCE',capture=True)
            row=P.torch.stack([W.scale(F.get_state(cache,l)).cpu() for l in P.S.LAYERS])
            for i,l in enumerate(P.S.LAYERS):P.check(P.torch.equal(row[i],h.quant.original(h.pre[l])[0].cpu()),'new streaming exact canonical current scale coordinates')
            clean.append(row);logits.append(out.logits.detach().float().cpu())
        del cache,out;gc.collect();P.torch.cuda.empty_cache()
        for first in range(0,36,12):
            art=A.pt_save(f'qualification/observer_{index}/scale_{first:03d}.pt',{
                'first_token_index':first,'last_token_index_exclusive':first+12,'layers':list(P.S.LAYERS),
                'scales':P.torch.stack(clean[first:first+12]),'dtype':'FP32_lossless'})
            chunks.append(dict(art,first_token_index=first,last_token_index_exclusive=first+12))
        reader=W.ScaleReader({'chunks':chunks});stream_hashes=[]
        for p in range(36):
            actual=reader.get(p)
            P.check(P.torch.equal(actual.cpu(),clean[p]),'lossless atomic chunk SHA/token/layer/group reconstruction')
            stream_hashes.append(P.N.tsha(actual))
        P.check(reader.index==3,'all scale chunks consumed exactly once')
        outcomes=[];parent_logits=[];legacy_parity=None
        for enabled in [False,True]:
            h.previous={};cache=None;trace=[];legacy=W.legacy_reducer();stats=W.Statistics()
            for p,token in enumerate(ids):
                if enabled:
                    out,cache=h.forward(token,cache,p,capture=True,qual=True)
                    before=P.fingerprint(cache,model,out.logits,h.codes)
                    for i,l in enumerate(P.S.LAYERS):
                        adaptive=W.scale(h.pre[l]);c=clean[p][i].to('cuda:0')
                        codes=P.torch.round(h.pre[l].float()/adaptive).clamp(-127,127)
                        vals=W.statistics_values(h.pre[l],adaptive,c,adaptive,codes)
                        stats.add(0,l,vals)
                        legacy.add('PARENT',l,8193+p,P.torch.log2(adaptive/c),adaptive!=adaptive,int(vals[6]),int(vals[8]),parity=(l==0))
                        P.check(P.torch.equal(F.get_state(cache,l),codes*adaptive),'production observer original Parent QDQ exact')
                    if (p+1)%12==0:stats.flush();legacy.flush()
                    P.check(before==P.fingerprint(cache,model,out.logits,h.codes),'full production statistics and legacy reducer no mutation')
                else:
                    out,cache=h.step('C6R',token,cache,p,qual=True)
                    parent_logits.append(out.logits.detach().float().cpu())
                trace.append(P.digest(P.fingerprint(cache,model,out.logits,h.codes)))
            outcomes.append(trace)
            if enabled:
                legacy_parity=legacy.parity_report();P.check(legacy_parity['G8_STREAMING_REDUCER_PARITY']=='PASS','unchanged32K reducer CPU integer-histogram reference parity')
                rows=legacy.summary();sr=stats.result({'qualification':index})
                P.check(sum(r['count'] for r in rows)==int(math.fsum(r['groups'] for r in sr)),'legacy/new streaming count parity')
                P.check(sum(r['clamp_count'] for r in rows)==0,'Parent observer never invokes clamp')
            del cache,out;gc.collect();P.torch.cuda.empty_cache()
        P.check(outcomes[0]==outcomes[1],'production scale-stream/metrics/reducer observer OFF/ON Parent trajectory bit parity')
        errors=[];values=[]
        pairs=list(zip(logits,parent_logits))
        # Independent prospective dense stress fixtures cover broad KL and logits
        # range; no formal-document outputs inform numerical resolution.
        axis=P.torch.arange(248320,dtype=P.torch.float32)
        for amplitude in [1.,10.,80.,1000.]:
            teacher=P.torch.sin(axis*.0013)*amplitude;student=P.torch.cos(axis*.0017)*amplitude
            for shift in [0.,1000.]:pairs.append((teacher+shift,student+shift))
        for fp,student in pairs:
            observed=P.kl32(fp.to('cuda:0'),student.to('cuda:0'));reference=P.kl64(fp,student)
            errors.append(abs(observed-reference));values.append(observed)
        P.check(P.kl32(logits[0].to('cuda:0'),logits[0].to('cuda:0'))==0,'full vocab identical KL exactly0')
        repeated=(values*384)[:384];stable=P.S.mean(repeated)
        independent=float(P.torch.tensor(repeated,dtype=P.torch.float64).mean())
        reducer_error=abs(stable-independent);resolution=2*max(errors)+2*reducer_error
        P.save(f'qualification/observer_reducer_{index}.json',{'status':'PASS','engineering_only':True,'process_index':index,
            'scale_stream_hashes':stream_hashes,'Parent_observer_trace':outcomes[0],'legacy_reducer_parity':legacy_parity,
            'legacy_reducer_source_SHA':P.sha(W.LEGACY_REDUCER_SOURCE),'full_vocab_size':248320,
            'FP32_vs_FP64_max_abs_error':max(errors),'stable_time_reducer_max_abs_error':reducer_error,
            'pre_outcome_DELTA_resolution':resolution,'KL_test_values':values,'all_FP32_FP64_errors':errors,
            'hypothesis_and_primary_unchanged':True,**A.receipt(started)})
    h.close();print('PRODUCTION_OBSERVER_SCALE_STREAM_KL_REDUCER_QUALIFICATION_PASS',index,flush=True)

if __name__=='__main__':
    try:run(int(sys.argv[1]))
    except Exception as exc:
        if not (P.ROOT/'FIRST_FAILURE_GATE.json').exists():P.save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP',
            'stage':'PRODUCTION_OBSERVER_REDUCER_QUALIFICATION','utc':P.N.now(),'exception':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'formal_outcomes_produced':False,'no_retry':True})
        raise
