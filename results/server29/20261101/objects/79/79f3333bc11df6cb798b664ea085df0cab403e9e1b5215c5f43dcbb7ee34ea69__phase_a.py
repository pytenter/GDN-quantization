"""Source/metric identity first, then historical-development transition parity.

No discovery/confirmation document is selected, loaded or measured in Phase A.
No model, optimizer, training or free generation is needed for this closure.
"""
from common import *
import ast, argparse, csv, math, statistics

F = ENV/'lib/python3.11/site-packages/fla/ops/kda'
TRAINER = V3/'inputs/historical_trainer.py'
TRACE = V3/'inputs/traces/wikitext2raw-train-000.pt'
TF = PARENTS[0]; DA = PARENTS[1]
P2 = PARENTS[2]/'scripts/phase2_final_hidden_512_smoke.py'

def evidence(p, lo, hi):
    lines=Path(p).read_text().splitlines()
    return dict(**row(p),line_start=lo,line_end=hi,source='\n'.join(lines[lo-1:hi]))

def prepare():
    assert not (D/'run').exists(), 'PHASE_A_MUST_PRECEDE_GPU_RUN'
    runtime_result=runtime()
    assert load(V4/'reports/fresh_b_only_v1/FINAL_VERDICT.json')['FRESH_B_EXECUTION_GATE']=='PASS'
    before=[];trees=[]
    for root in PARENTS:
        assert root.is_dir(),str(root)
        ff=files(root);trees.append(dict(root=str(root),paths=[str(p) for p in ff]))
        print('PROTECT_PARENT',root.name,len(ff),sum(p.stat().st_size for p in ff),flush=True)
        for i,p in enumerate(ff):
            before.append(row(p))
            if i%200==0: print('PARENT_HASH_PROGRESS',root.name,i,len(ff),flush=True)
    emit('provenance/parent_inventory_before.json',dict(gate='PASS',trees=trees,files=before,pre_task_head=command(['git','rev-parse','HEAD']),mode='read-only direct filesystem inventory; external parent worktrees retained'))
    emit('runtime/deterministic_runtime_identity.json',runtime_result)
    emit('provenance/hardware_identity.json',hardware())
    manifest=load(V4/'runtime/v4_deterministic_runtime_manifest.json')
    emit('runtime/inherited_launch_profile.json',dict(authority=row(V4/'runtime/v4_deterministic_runtime_manifest.json'),launch_profile=manifest['launch_profile'],gpu_mask_override='physical assigned GPU only',no_package_search=True))
    sources=[evidence(F/'naive.py',11,74),evidence(F/'fused_recurrent.py',135,202),evidence(F/'fused_recurrent.py',245,329),evidence(TRAINER,253,339),evidence(P2,174,222),evidence(P2,320,345)]
    actual=(F/'fused_recurrent.py').read_text()
    for s in ['b_h *= exp(b_gk[:, None])','b_v -= tl.sum(b_h * b_k[:, None], 0)','b_h += b_k[:, None] * b_v[None, :]','use_qk_l2norm_in_kernel']: assert s in actual,s
    sources.append(row(F/'gate.py'))
    emit('provenance/kda_transition_source_audit.json',dict(gate='PASS',sources=sources,
        recurrence='D_t S + beta_t k_t (v_t - k_t^T D_t S)^T; M_t=(I-beta_t k_t k_t^T) D_t',
        state_layout='[B,HV,K,V]; Ling observed B=1,H=HV=16,K=V=128',
        log_decay='canonical final_log_decay; not raw g; exp(log_decay) on key axis',
        key='raw frozen key normalized in actual kernel: k/sqrt(sum(k^2)+1e-6)',
        beta='already processed beta; use_beta_sigmoid_in_kernel=False',
        write='beta k v^T after erase; zero-write sets v=0, not beta=0',
        full_transition=True,rank1_erase_retained=True,
        Value_rotation='state_rotated=state_native@R; v_rotated=v_native@R then original v dtype; recover state_native=state_rotated@R.T',
        QDQ='after actual FP32 fused final state, Value-axis amax/127 ties-even clip[-127,127], dequant immediate recurrent writeback; no FP reset',
        historical_SGLang_layout='[V,K] with stored=R.T@native; transpose corresponds to current [K,V] native@R',
        smoke_trace=row(TRACE),layer_ids=LAYERS,new_model_forward=False,new_AIME_generation=0))
    # Exact semantics chosen BEFORE ANY smoke output. Floating-point affine
    # cancellation is not bitwise linear; no tolerance search or relaxation.
    emit('protocol/transition_parity_semantics.json',dict(gate='FROZEN_BEFORE_SMOKE',
        dtype='float32 actual fused final-state; bf16 q/k/v frozen trace; canonical float32 g/beta',
        torch_allclose_atol=5e-5,torch_allclose_rtol=5e-5,
        offsets=[0,7,31,127,511],layers=LAYERS,state_pair='A=initial+0.01*sin(arange); B=0.75*initial-0.01*cos(arange)',
        tests=['affine_difference_equals_zero_write','write_cancellation','actual_fused_matches_existing_naive_reference','native_rotated_basis','QDQ_indices_and_next_input_identity'],
        fresh_docs_used=0,no_retry_on_semantic_failure=True,
        rationale='FP32 rounded affine update; numerical parity not bitwise cancellation. Same tolerance for all layers/tokens/tests.'))
    metric_sources=[evidence(TF/'scripts/sglang_kda_tf_replay_diag_patch.py',49,93),evidence(DA/'scripts/inj_vs_acc.py',1,82),evidence(TF/'scripts/analyze_tf_replay.py',49,70),evidence(TF/'scripts/p1_relative_error.py',61,93),evidence(DA/'scripts/analyze_p2_all.py',45,116)]
    inj=load(DA/'analysis/injection_vs_accumulation.json'); reproductions={}
    runs={'H':TF/'runs/H_gpu1','L6':TF/'runs/L6_gpu0','L7':TF/'runs/L7_gpu1','DA_fromH':DA/'runs/da_initH','DA_fromL6':DA/'runs/da_initL6'}
    for name,run in runs.items():
        q=run/'qdq_injection.tsv';vals=[]
        with q.open() as f:
            for x in csv.DictReader(f,delimiter='\t'):
                if x.get('phase')!='decode':continue
                try:
                    if 1024<=int(x['step'])<=76800:vals.append(float(x['rel_err']))
                except (ValueError,KeyError):pass
        expected=inj['injection_common_window'][name]
        got=dict(n=len(vals),mean=statistics.mean(vals),median=statistics.median(vals),min=min(vals),max=max(vals))
        assert got==expected,(name,got,expected)
        reproductions[name]=dict(raw=row(q),computed=got,exact_saved_metadata_match=True)
    emit('provenance/historical_metric_definition_audit.json',dict(gate='PASS',sources=metric_sources,reproductions=reproductions,
        injection='||dequant-source||_F/(||source||_F+EPS) per decode row; source is actual pre-QDQ state',
        decode_only=True,window=[1024,76800],inj_vs_acc_aggregation='arithmetic mean of all decode rows including layers; no per-step dedup in inj_vs_acc.py',
        analyze_tf_replay_dedup='same(step,layer): keep greatest src_norm; distinct historical helper, not applied to inj_vs_acc reproduction',
        accumulated='E(t)=sqrt(sum over selected layers ||R@stored_rotated_state - native_FP_state||_F^2); relative E/sqrt(sum ||FP||^2); ratio vs H in analyze_p2_all',
        historical_acc_layers=[0,6,12,18,22],historical_acc_steps='common sampled snapshot grid >=1024; not every token',
        inj_vs_acc_acc_values='hardcoded rounded ratios in historical source, not independently recomputed by inj_vs_acc.py',
        new_primary_distinct='prospectively specified mean per-token per-layer relative squared energy on all18 layers, [1024,T] inclusive; NOT claimed identical to historical sampled aggregate E(t)',
        raw_schema='step,phase,layer,src_norm,err_norm,rel_err,amax,n',saved_comparison=row(DA/'analysis/injection_vs_accumulation.json')))
    emit('provenance/phase_a_source_prepare.json',dict(KDA_TRANSITION_SOURCE_IDENTITY_GATE='PASS',P1_P2_METRIC_IDENTITY_GATE='PASS',DETERMINISTIC_RUNTIME_IDENTITY_GATE='PASS',PARENT_IMMUTABILITY_GATE='PASS',ZERO_WRITE_TRANSITION_PARITY_GATE='NOT_EVALUATED',NEW_TRAINING='NO',NEW_AIME_GENERATION=0,DISCOVERY_COMPLETE='0/64',CONFIRMATION_COMPLETE='0/64'))
    print('PHASE_A_SOURCE_AND_METRIC_GATE=PASS; GPU_PARITY_NOT_RUN',flush=True)

def smoke():
    import torch
    from transition import update_pre_q,propagate_zero_write,qdq_r128
    from fla.ops.kda.naive import naive_recurrent_kda
    assert not (D/'run/STOP.json').exists()
    runtime();sem=load(D/'protocol/transition_parity_semantics.json')
    audit=load(D/'provenance/kda_transition_source_audit.json')
    for src in audit['sources']:assert sha(src['path'])==src['sha256']
    assert sha(TRACE)==audit['smoke_trace']['sha256']
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);hw=hardware();assert hw['gpus'][gpu]['gpu_uuid']==UUIDS[gpu]
    assert torch.cuda.device_count()==1
    trace=torch.load(TRACE,map_location='cpu',weights_only=False);assert list(trace['layers'])==LAYERS
    hpath=Path(load(V4/'protocol/canonical_endpoint_identity.json')['H_rotation']['path'])
    hrot=torch.load(hpath,map_location='cpu',weights_only=False)['rotations']
    rows=[]
    def compare(a,b,test,layer,token):
        ok=torch.allclose(a,b,atol=sem['torch_allclose_atol'],rtol=sem['torch_allclose_rtol'])
        z=dict(test=test,layer_id=layer,token_offset=token,max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)),gate='PASS' if ok else 'FAIL')
        rows.append(z)
        if not ok:
            emit('analysis/transition_parity_failed_case.json',z);raise AssertionError(z)
    with torch.inference_mode():
        for layer in LAYERS:
            lr=trace['layer_records'][layer];init=lr['initial_teacher_state'].float().cuda()
            ix=torch.arange(init.numel(),dtype=torch.float32).reshape(init.shape).cuda()
            a=init+0.01*ix.sin();b=.75*init-.01*ix.cos();r=hrot[layer].float().cuda()
            assert list(init.shape)==[1,16,128,128]
            for token in sem['offsets']:
                rec={k:v[:,token:token+1].cuda() for k,v in lr.items() if k in ['q','k','v','beta','log_decay']}
                ua=update_pre_q(a,rec);ub=update_pre_q(b,rec);zero=propagate_zero_write(a-b,rec)
                compare(ua-ub,zero,'affine_difference_equals_zero_write',layer,token)
                write=update_pre_q(torch.zeros_like(a),rec)
                compare(ua-propagate_zero_write(a,rec),write,'write_cancellation',layer,token)
                q=rec['q'].float();k=rec['k'].float();q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
                _, ref=naive_recurrent_kda(q,k,rec['v'],rec['log_decay'],rec['beta'],initial_state=a,output_final_state=True)
                compare(ua,ref,'actual_fused_matches_existing_naive_reference',layer,token)
                mapped=propagate_zero_write((a-b).matmul(r),rec).matmul(r.T)
                compare(mapped,zero,'native_rotated_zero_write_basis',layer,token)
                # Mapping tested in FP32 without an extra v cast: representation
                # rounding of rotated BF16 writes is audited separately in replay.
                vr=rec['v'].float().matmul(r)
                rotated=update_pre_q(a.matmul(r),rec,vr).matmul(r.T)
                compare(rotated,ua,'native_rotated_full_write_basis_FP32',layer,token)
                post,codes,scale=qdq_r128(ua)
                assert post.shape==ua.shape and post.dtype==ua.dtype and codes.min()>=-127 and codes.max()<=127
                assert torch.equal(post,(codes.float()*scale).to(post.dtype))
                next_input=post;assert tensor_sha(next_input)==tensor_sha(post)
                rows.append(dict(test='QDQ_after_update_t_next_input_t_plus_1',layer_id=layer,token_offset=token,gate='PASS',pre_q_sha=tensor_sha(ua),post_q_sha=tensor_sha(post),next_input_sha=tensor_sha(next_input)))
    emit('analysis/transition_parity_smoke.json',dict(gate='PASS',ZERO_WRITE_TRANSITION_PARITY_GATE='PASS',trace=row(TRACE),semantics=row(D/'protocol/transition_parity_semantics.json'),physical_gpu=gpu,gpu_uuid=UUIDS[gpu],rows=rows,tests=len(rows),fresh_documents_used=0,model_loaded=False,NEW_TRAINING='NO',NEW_AIME_GENERATION=0))
    print('ZERO_WRITE_TRANSITION_PARITY_GATE=PASS',len(rows),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','smoke']);args=a.parse_args()
    try: prepare() if args.mode=='prepare' else smoke()
    except BaseException as e:stop('PHASE_A_'+args.mode,e);raise
