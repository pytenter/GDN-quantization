"""Prospective Phase A, production-helper reuse on an old non-AIME dev probe.

No discovery/confirmation measurement, training, scoring or generation.
"""
import argparse, ast, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import propagation_recovery_closure as A
ROOT=A.ROOT
READINESS=A.SRC/'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py'
PHYSICAL=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py')

def mod(n,p):
    sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()

def freeze():
    A.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU policy freeze')
    A.check(A.read(ROOT/'provenance/rotation_parameterization_identity.json')['ROTATION_PARAMETERIZATION_IDENTITY_GATE']=='PASS','parameterization not PASS')
    p=dict(status='PHASE_A_ONLY_DISCOVERY_NOT_AUTHORIZED',task=A.TASK,
           source_sha256={str(p):A.sha(p) for p in [A.ARCH,A.FORMAL,A.CAYLEY,READINESS,PHYSICAL,Path(__file__),HERE/'propagation_recovery_closure.py']},
           helper='unchanged torch_recurrent_gated_delta_rule; zero_write sets only value=zeros_like(value)',
           native_mapping='compiled unchanged archived physical_state; C6R internal alias C5, C7 alias C6 only for the frozen helper condition switch',
           inverse_mapping='H128 applied by einsum then Delta.T applied by einsum; no precomposition',
           numerical_policy=dict(frozen_before_full_parity_panel=True,fp32_unit_roundoff=2**-24,reduction_width=128,
              gamma128=128*(2**-24)/(1-128*(2**-24)),roundoff_multiplier=8,
              comparison='max_abs_residual <= 8*gamma128*max(1,max_abs_of_compared_inputs_and_outputs)',
              rationale='FP32 128-coordinate reductions plus separate H/Delta and inverse mapping reductions; conservative dtype-derived absolute forward-error bound. Difference cancellation is scaled to original operands, not tiny residual.',
              exact_checks=['FP production helper vs model captured FP final states','repeat deterministic operator','rotation file and per-layer tensor SHA'],
              invariance='exact tensor equality; FAIL allowed, then condition-specific production transition must be used'),
           test_seed=73619,conditions=['H','C6R','C7'],dev_probe='historical TRAIN:first-row first32 token IDs; not discovery or confirmation',
           expected_dev_token_sha256='1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e668a606d',
           random_state_panel_steps=[0,7,15,30],actual_driven_panel='all31 steps,24layers,3conditions',
           J_DECAY_AVAILABLE='YES',decay_source='exp(g_t) per batch/value-head, broadcast K,V; no averaged scalar',
           scientific_docs_selected=False,rotation_bank_created=False,discovery_units_started=0,confirmation_units_started=0,
           NEW_TRAINING='NO',NEW_FREE_GENERATION=0,NEW_AIME_GENERATION=0,NEW_C8_TRAINING_AUTHORIZED='NO',
           full_premeasurement_freeze_complete=False,PREMEASUREMENT_COMMIT_SHA=None)
    A.save('protocol/preregistration.json',p)
    A.save('provenance/PHASE_A_PARITY_FREEZE.json',dict(preregistration_sha256=A.sha(ROOT/'protocol/preregistration.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_parity_panel_results_read=True))
    print('PHASE_A_POLICY_FROZEN',flush=True)

def execute():
    import torch
    p=A.read(ROOT/'protocol/preregistration.json')
    for q,h in p['source_sha256'].items():A.check(A.sha(q)==h,'source drift '+q)
    A.check(A.sha(ROOT/'protocol/preregistration.json')==A.read(ROOT/'provenance/PHASE_A_PARITY_FREEZE.json')['preregistration_sha256'],'policy drift')
    A.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    F=mod('propagation_original_formal',A.FORMAL);F.setup_seed(p['test_seed'])
    R=mod('propagation_original_readiness',READINESS);core=R.load_core()
    model,tok=F.load_model_and_tokenizer()
    ids,_=core.tokenize(tok,core.corpus_rows('TRAIN')[0]);ids=ids[:32]
    A.check(tsha(torch.tensor(ids,dtype=torch.long))==p['expected_dev_token_sha256'],'dev probe drift')
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    helper=qmod.torch_recurrent_gated_delta_rule
    layers=F.GDN_LAYERS;captured={l:[] for l in layers};initial={};active=[None];handles=[]
    model_layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
    for l in layers:
        block=model_layers[l];module=getattr(block,'linear_attn',None) or getattr(block,'self_attn',None)
        A.check(module is not None,'missing GDN module')
        handles.append(module.register_forward_pre_hook(lambda _m,_args,l=l:active.__setitem__(0,l)))
    def capture(query,key,value,*args,**kw):
        A.check(not args,'unexpected positional recurrence controls')
        l=active[0];A.check(l in captured,'unknown GDN layer')
        if l not in initial:initial[l]=kw['initial_state'].detach().cpu().clone()
        record={n:v.detach().cpu().clone() for n,v in [('query',query),('key',key),('value',value),('g',kw['g']),('beta',kw['beta'])]}
        record['use_norm']=kw.get('use_qk_l2norm_in_kernel',False)
        out,state=helper(query,key,value,**kw)
        A.check(tuple(state.shape)==(1,32,128,128) and state.dtype==torch.float32,'actual state shape/dtype')
        record['fp_post_state_sha256']=tsha(state);captured[l].append(record)
        return out,state
    qmod.torch_recurrent_gated_delta_rule=capture
    try:
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([[ids[0]]],device='cuda:0'),use_cache=True)
            cache=out.past_key_values
            for i in ids[1:]:
                out=model(input_ids=torch.tensor([[i]],device='cuda:0'),past_key_values=cache,use_cache=True);cache=out.past_key_values
    finally:
        qmod.torch_recurrent_gated_delta_rule=helper
        for h in handles:h.remove()
    A.check(all(len(captured[l])==31 for l in layers),'dev capture coverage')
    del model,out,cache;__import__('gc').collect();torch.cuda.empty_cache()
    trace=ROOT/'traces/dev_phase_a_trace.pt';trace.parent.mkdir(parents=True,exist_ok=True);A.check(not trace.exists(),'no dev trace overwrite')
    with trace.open('xb') as f:torch.save(dict(initial=initial,records=captured,ids=ids,discovery=False,confirmation=False),f);f.flush();os.fsync(f.fileno())
    trace.chmod(0o444)
    A.save('traces/dev_phase_a_manifest.json',dict(path=str(trace),sha256=A.sha(trace),layers=list(layers),steps=31,fp_teacher_forced=True,generated_tokens=0,initial_state_layout=[1,32,128,128],stored_states='24 initial states only; no full state sequence',probe_ids_sha256=p['expected_dev_token_sha256']))
    print('DEV_FP_TRANSITION_CAPTURE=PASS',flush=True)
    ns={'torch':torch};text=PHYSICAL.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='physical_state');exec(compile(ast.get_source_segment(text,node),str(PHYSICAL),'exec'),ns)
    physical=ns['physical_state'];banks={};patches={'H':F.QwenKeyHadamardPatch(None)};aliases={'H':'H','C6R':'C5','C7':'C6'}
    recovery=A.read(ROOT/'provenance/anchor_recovery.json')
    for x in recovery['anchors']:
        if x['condition'] not in ['C6R','C7']:continue
        payload=torch.load(x['matrix'],map_location='cpu',weights_only=True)
        bank=F.FrozenRotationBank(payload['layer_ids'],payload['matrices']).to('cuda:0');banks[aliases[x['condition']]]=bank
        patches[x['condition']]=F.QwenFrozenRotationPatch(None,bank,{'source':x['matrix'],'sha256':x['matrix_sha256']})
    h=F.hadamard(128,device='cuda:0')
    def native(c,l,s):return physical(F,aliases[c],l,s,banks)
    def stored(c,l,s):
        y=torch.einsum('ij,...jv->...iv',h,s)
        if c!='H':y=torch.einsum('ij,...jv->...iv',banks[aliases[c]].layer(l).matrix().T,y)
        return y
    def step(c,l,s,r,zero=False):
        patch=patches[c];patch.current_layer=l
        q=patch._transform(r['query'],r['use_norm']);k=patch._transform(r['key'],r['use_norm'])
        _,state=helper(q,k,torch.zeros_like(r['value']) if zero else r['value'],g=r['g'],beta=r['beta'],initial_state=s,output_final_state=True,use_qk_l2norm_in_kernel=False)
        return state
    def propagate_zero_write(c,l,shadow,r):return native(c,l,step(c,l,stored(c,l,shadow),r,zero=True))
    policy=p['numerical_policy'];roundoff=policy['roundoff_multiplier']*policy['gamma128']
    def compare(a,b,*scales):
        maximum=max([1.]+[float(s.abs().max()) for s in (a,b)+scales]);d=float((a-b).abs().max());bound=roundoff*maximum
        return dict(max_abs_residual=d,bound=bound,operand_max_abs=maximum,PASS=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and d<=bound))
    mapping=[];difference=[];driven=[];invariance=[];fp_exact=[];first_failure=None
    fp={l:initial[l].to('cuda:0') for l in layers}
    state={c:{l:stored(c,l,fp[l]) for l in layers} for c in p['conditions']}
    gen=torch.Generator(device='cuda:0').manual_seed(p['test_seed'])
    with torch.inference_mode():
        for l in layers:
            test=torch.randn((1,32,128,128),generator=gen,device='cuda:0')
            for c in p['conditions']:
                row=dict(condition=c,layer=l,**compare(native(c,l,stored(c,l,test)),test,test));mapping.append(row)
                if not row['PASS']:first_failure=dict(gate='NATIVE_BASIS_MAPPING_GATE',row=row);break
            if first_failure:break
        if not first_failure:
            for t in range(31):
                records={l:{n:(v.to('cuda:0') if torch.is_tensor(v) else v) for n,v in captured[l][t].items()} for l in layers}
                fp_old=fp;fp={}
                pre={c:{} for c in p['conditions']};pred={c:{} for c in p['conditions']}
                for l in layers:
                    r=records[l]
                    _,fp[l]=helper(r['query'],r['key'],r['value'],g=r['g'],beta=r['beta'],initial_state=fp_old[l],output_final_state=True,use_qk_l2norm_in_kernel=r['use_norm'])
                    exact=tsha(fp[l])==r['fp_post_state_sha256'];fp_exact.append(dict(token=t,layer=l,exact=exact))
                    if not exact:first_failure=dict(gate='FP_TRANSITION_TRACE_REPLAY_EXACT',token=t,layer=l);break
                    native_zero=None
                    for c in p['conditions']:
                        old=native(c,l,state[c][l]);e=old-fp_old[l]
                        pre[c][l]=step(c,l,state[c][l],r);y=native(c,l,pre[c][l]);pred[c][l]=propagate_zero_write(c,l,e,r)
                        row=dict(condition=c,token=t,layer=l,kind='preQ_error',**compare(y-fp[l],pred[c][l],y,fp[l],old,fp_old[l]));driven.append(row)
                        if not row['PASS']:first_failure=dict(gate='GDN_DRIVEN_ERROR_RECURRENCE_GATE',row=row);break
                        if t in p['random_state_panel_steps']:
                            aa=torch.randn((1,32,128,128),generator=gen,device='cuda:0');bb=torch.randn((1,32,128,128),generator=gen,device='cuda:0')
                            ua=native(c,l,step(c,l,stored(c,l,aa),r));ub=native(c,l,step(c,l,stored(c,l,bb),r));z=propagate_zero_write(c,l,aa-bb,r)
                            row=dict(condition=c,token=t,layer=l,**compare(ua-ub,z,aa,bb,ua,ub));difference.append(row)
                            if not row['PASS']:first_failure=dict(gate='GDN_ZERO_WRITE_TRANSITION_PARITY_GATE',row=row);break
                            # Identical deterministic vector for all three maps, no condition-specific random seed.
                            shadow=fp_old[l]
                            zero=propagate_zero_write(c,l,shadow,r)
                            if native_zero is None:
                                _,native_zero=helper(r['query'],r['key'],torch.zeros_like(r['value']),g=r['g'],beta=r['beta'],initial_state=shadow,output_final_state=True,use_qk_l2norm_in_kernel=r['use_norm'])
                            invariance.append(dict(condition=c,token=t,layer=l,exact=torch.equal(zero,native_zero),max_abs_diff=float((zero-native_zero).abs().max())))
                    if first_failure:break
                if first_failure:break
                for c in p['conditions']:
                    cache=SimpleNamespace(recurrent_states=pre[c]);audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
                    before={l:native(c,l,pre[c][l]) for l in layers}
                    F.quantize_c128(cache,audit)
                    A.check(audit['calls']==1 and sorted(audit['layers'])==list(layers),'production QDQ coverage')
                    for l in layers:
                        post=native(c,l,pre[c][l]);eta=post-before[l]
                        row=dict(condition=c,token=t,layer=l,kind='postQ_driven_error',**compare(post-fp[l],pred[c][l]+eta,post,fp[l],before[l]));driven.append(row)
                        if not row['PASS']:first_failure=dict(gate='GDN_DRIVEN_ERROR_RECURRENCE_GATE',row=row);break
                    state[c]=pre[c]
                    if first_failure:break
                if first_failure:break
                print('PARITY_STEP='+str(t+1),flush=True)
    expected_zero=3*24*4;expected_driven=3*24*31*2
    zero_pass=len(difference)==expected_zero and all(x['PASS'] for x in difference)
    driven_pass=len(driven)==expected_driven and all(x['PASS'] for x in driven)
    mapping_pass=len(mapping)==72 and all(x['PASS'] for x in mapping)
    inv_status='PASS' if invariance and all(x['exact'] for x in invariance) else 'FAIL' if len(invariance)==expected_zero else 'NOT_EVALUATED'
    A.save('provenance/native_basis_mapping.json',dict(NATIVE_BASIS_MAPPING_GATE='PASS' if mapping_pass else 'FAIL',rows=mapping,source=A.source(PHYSICAL,'physical_state'),inverse_definition=p['inverse_mapping'],policy=policy))
    A.save('provenance/transition_rotation_invariance.json',dict(PHYSICAL_TRANSITION_ROTATION_INVARIANCE=inv_status,rows=invariance,criterion='exact tensor equality',shared_transition_forced=False,required_operator='actual condition-specific native-basis production zero-write',analytical='Exact orthogonal arithmetic gives invariance. Actual FP32 separate maps, matrix orthogonality residuals and reduction rounding can differ. No full-network or autoregressive invariance claim.'))
    A.save('provenance/GDN_ZERO_WRITE_PARITY.json',dict(GDN_ZERO_WRITE_TRANSITION_PARITY_GATE='PASS' if zero_pass else 'FAIL',rows=difference,expected=expected_zero,policy=policy,helper_sha256=A.sha(A.ARCH)))
    A.save('provenance/GDN_ERROR_RECURRENCE_PARITY.json',dict(GDN_DRIVEN_ERROR_RECURRENCE_GATE='PASS' if driven_pass else 'FAIL',rows=driven,expected=expected_driven,policy=policy,fp_trace_replay_exact=fp_exact,first_failure=first_failure))
    result=dict(PHASE_A_STATUS='PASS' if mapping_pass and zero_pass and driven_pass and first_failure is None else 'FAIL',first_failure=first_failure,PHYSICAL_TRANSITION_ROTATION_INVARIANCE=inv_status,DISCOVERY_COMPLETE='0/48',CONFIRMATION_COMPLETE='0/48',full_premeasurement_freeze_complete=False,next_stage='prepare fresh documents and exact12 bank ONLY after Phase A PASS',NEW_TRAINING='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0)
    A.save('analysis/phase_a_result.json',result);A.manifest('provenance/PHASE_A_SHA256SUMS')
    print(json.dumps(result),flush=True)
    return 0 if result['PHASE_A_STATUS']=='PASS' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['freeze','execute'],required=True);args=parser.parse_args()
    if args.mode=='freeze':freeze()
    else:
        try:sys.exit(execute())
        except Exception as e:
            A.save('analysis/phase_a.error.json',dict(status='STOP',exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),scientific_units_started=0,no_automatic_retry=True));raise
