"""Premeasurement protocol and complete identity inventory, no fresh forwards."""
from common import *
from replay_core import *
from surrogate_statistics import tests
import traceback

def main():
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    assert load(D/'provenance/engineering_parser_recovery_v2_completion.json')['gate']=='PASS'
    assert load(D/'protocol/frozen_documents.json')['gate']=='PASS'
    assert load(D/'protocol/frozen_rotation_bank.json')['count']==16
    for gpu in [0,1]:assert load(D/'runtime'/f'capture_canary_gpu{gpu}.json')['gate']=='PASS'
    reports=[load(D/'runtime'/f'feasibility_gpu{gpu}.json') for gpu in [0,1]]
    K=max(k for k in [4,8] if all(any(r['K']==k and r['stable'] for r in a['canaries']) for a in reports))
    signs=sign_schedule(K,32768);p=D/'protocol/probe_signs.pt';assert not p.exists();torch.save(signs,p);p.chmod(0o444)
    tests()
    config=dict(gate='PASS',task=TASK,probe_count=K,probe_sign_schedule=dict(algorithm='SHA256(ascii seed:one-based-token:zero-based-probe), first-byte low bit; one=>+1 else -1',seed=SIGN_SEED,artifact=row(p),shared_all_docs_and_rotations=True),horizons=HORIZONS,window_start=WINDOW_START,window='one-based inclusive [1024,h]; history propagates from token1',epsilon=EPS,layer_ids=LAYERS,ground_truth='mean_{tokens,layers} ||Squant@R.T-Sfp||^2/(||Sfp||^2+eps)',injection='own true post-update pre-Q FP32 y; eta=Q(y)-y; mean ||eta||^2/(||y||^2+eps)',historical_relative_injection='mean ||eta||/(||y||+eps); descriptive only',decay=dict(label='DECAY_ONLY_APPROXIMATION',d_eff_squared='exp(2*canonical_log_decay).mean(true-head,key)',tensor_layout='[layer-as-batch,token=1,true-head,key]',recurrence='A_t=d_eff_t^2*A_(t-1)+inj_energy_t; A_0=0; no erase'),propagation=dict(implementation='same actual FLA fused kernel with zero value; beta rank1 erase retained',recurrence='z_r,t=propagate_zero_write(z_r,t-1,record_t)+xi_r,t*eta_t',energy='mean_r ||z_r,t||^2/(||Sfp_t||^2+eps)',interpretation='signed estimator of propagated injection energy, not equal to J_ACC'),system_boundary='ON_POLICY_WITH_RESPECT_TO_RECURRENT_STATE_AND_QDQ UNDER_A_FROZEN_TEACHER_FORCED_TRANSITION_TRACE',value_basis='state key-first @ R; value float32 @ R then original BF16; recover @R.T',cross_term_residual=dict(definition='J_ACC-J_PROP',policy='DESCRIPTIVE_ONLY',caveat='finite Monte Carlo variability and BF16 rotated-write rounding can also contribute; not an isolated exact cross-term'),resources=dict(hardware='2xRTX4090 only',assignment='doc_rank%2',rotation_order='frozen bank manifest',trace_capture='once FP model exact teacher forcing',replay='FP model unloaded; all18layers/probes batched bitwise qualified',single_rollout=True),statistics=dict(spearman='average ranks, exact equality ties; constant panel undefined and candidate fails',concordance='concordant/120 over all rotation pairs; all ties non-success, categories separately reported',rules=dict(A='beats J_INJ strictly at least9/12',B='positive32768 at least3/4',C='median32768rho>=0.50',D='median32768concordance>=0.65'),selection='long-rho, all-rho, long-concordance, exact tie J_DECAY',no_discovery_confirmation_pooling=True,no_pvalues=True),NEW_TRAINING='NO',NEW_AIME_GENERATION=0,confirmation_authorization='only after discovery64/64rawfreeze and selected candidate PASS',FINAL_ACTION='STOP even if supported')
    emit('protocol/frozen_metrics_and_execution.json',config)
    external={}
    for f in load(V4/'runtime/readiness_source_freeze.json')['files']:
        assert sha(f['path'])==f['sha256'],('INHERITED_SOURCE_IDENTITY',f['path'])
        external[f['path']]=row(f['path'])
    for p in [PARENTS[2]/'scripts/phase2_final_hidden_512_smoke.py',V4/'scripts/v4_common.py',V4/'runtime/qualified_backward_selection.json',V4/'runtime/historical_fixed_forward_manifest.json',V3/'inputs/traces/wikitext2raw-train-000.pt']:
        external[str(p)]=row(p)
    # The package/source files are exact inherited bytes, not just version labels.
    source=load(D/'provenance/kda_transition_source_audit.json')
    def visit(x):
        if isinstance(x,dict):
            if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
                p=Path(x['path'])
                if p.is_file():assert sha(p)==x['sha256'];external[str(p)]=row(p)
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(source)
    parent=verify_parent();runtime();hardware()
    emit('provenance/prefreeze_parent_and_runtime_validation.json',dict(parent=parent,runtime=runtime(),hardware=hardware(),scientific_measurements=0))
    own=[row(p) for p in files(D) if '/runtime/capture_canary_gpu' not in str(p) or p.suffix=='.json' and '/chunks' not in str(p)]
    # Include inherited inputs/code and all new scientific objects. Do not include
    # self-referential manifest or future runtime outputs in this immutable list.
    emit('manifests/premeasurement_sha256.json',dict(gate='PASS',files=own+list(external.values()),measurement_count=0,external_parents_read_only=True))
    print('PREMEASUREMENT_FREEZE_PREPARATION=PASS; K=',K,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        emit('run/PREMEASUREMENT_FREEZE_STOP.json',dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),scientific_measurements=0,no_auto_retry=True));raise
