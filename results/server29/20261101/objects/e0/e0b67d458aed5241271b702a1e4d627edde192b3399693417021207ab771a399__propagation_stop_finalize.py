"""Seal the premeasurement deterministic-runtime stop; no further GPU work."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import propagation_recovery_closure as A
ROOT=A.ROOT

def main():
    A.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU finalize only')
    fault=A.read(ROOT/'analysis/phase_a_repaired_001.error.json')
    A.check('cumsum_cuda_kernel does not have a deterministic implementation' in fault['message'],'exact deterministic runtime failure')
    A.check((ROOT/'logs/phase_a_repaired_001.exitcode').read_text().strip()=='1','worker exitcode')
    A.check(not (ROOT/'traces/dev_phase_a_trace.pt').exists(),'unexpected completed trace')
    rec=A.read(ROOT/'provenance/anchor_recovery.json');params=A.read(ROOT/'provenance/rotation_parameterization_identity.json')
    A.check(params['ROTATION_PARAMETERIZATION_IDENTITY_GATE']=='PASS' and len(params['rows'])==72 and all(x['exact'] and x['reconstructed_sha256']==x['frozen_sha256'] for x in params['rows']),'parameter closure')
    for x in rec['anchors']:
        A.check(A.sha(x['matrix'])==x['matrix_sha256'] and A.sha(x['theta'])==x['theta_sha256'],'recovered identity')
    protected=A.read(ROOT/'provenance/parent_artifact_identity.json')['protected']
    for x in protected:A.check(A.sha(x['path'])==x['sha256'],'protected artifact drift '+x['path'])
    A.save('provenance/final_parent_revalidation.json',dict(status='PASS',files=protected,old_files_changed=False,old_verdicts_unchanged=True))
    not_run=dict(status='NOT_EVALUATED',reason='Strict deterministic CUDA cumsum blocked initial FP prefill before recurrent capture/parity panel',parity_panel_steps=0,semantic_or_numerical_retry=False)
    A.save('provenance/GDN_ZERO_WRITE_PARITY.json',dict(not_run,GDN_ZERO_WRITE_TRANSITION_PARITY_GATE='NOT_EVALUATED'))
    A.save('provenance/GDN_ERROR_RECURRENCE_PARITY.json',dict(not_run,GDN_DRIVEN_ERROR_RECURRENCE_GATE='NOT_EVALUATED'))
    A.save('provenance/native_basis_mapping.json',dict(not_run,NATIVE_BASIS_MAPPING_GATE='NOT_EVALUATED',analytic_mapping='Recovered archived physical_state and separate H/Delta orientation; numerical mapping smoke NOT_RUN'))
    A.save('provenance/transition_rotation_invariance.json',dict(not_run,PHYSICAL_TRANSITION_ROTATION_INVARIANCE='NOT_EVALUATED'))
    A.save('protocol/frozen_fresh_documents.json',dict(status='NOT_SELECTED',count=0,DOCUMENT_OVERLAP_AUDIT_GATE='NOT_EVALUATED',CONFIRMATION_SCIENTIFIC_RESULTS_READ_BEFORE_SELECTION='NO'))
    A.save('protocol/frozen_rotation_bank.json',dict(status='NOT_CREATED',ROTATION_BANK_COUNT=0,expected_count=12,exact_anchors_available=True,derived_rotations_created=0))
    A.save('protocol/metric_definitions.json',dict(status='USER_SPECIFICATIONS_RETAINED_NOT_FULL_PREMEASUREMENT_FROZEN',T_MAX=32768,horizons=[2048,8192,32768],window='inclusive1024..T',J_ACC='mean_l,t ||S_R_native-S_FP_native||_F^2/(||S_FP_native||_F^2+eps)',J_INJ='mean_l,t ||eta_native||_F^2/(||preQ_R_native||_F^2+eps)',J_PROP='mean_l,t mean_r||z_prop||_F^2/(||S_FP_native||_F^2+eps)',J_DECAY='same probes/signs/window/normalizer; actual per-head exp(g) component',epsilon='NOT_FROZEN',probe_count='NOT_FROZEN',discovery_gate='9/12 beats INJ;3/4 positive32K;median rho>=.50;median concordance>=.65',threshold_changed=False,candidate_metrics_computed=False))
    A.save('traces/trace_status.json',dict(status='NOT_CAPTURED',development_forward_attempts=1,completed_development_forwards=0,fp_prefill_failed=True,teacher_forced_recurrent_steps=0,fresh_document_trace_count=0))
    for name in ['discovery','confirmation']:
        A.save('analysis/'+name+'_summary.json',dict(status='NOT_RUN',complete='0/48',reason='premeasurement runtime stop',rotation_conditioned_metrics_computed=False))
    verdict=dict(EXPERIMENT=A.TASK,EVIDENCE_TYPE='PROSPECTIVE_CROSS_ROTATION_LONG_HORIZON_MECHANISM_VALIDATION',TASK_STATUS='STOP_STRICT_DETERMINISTIC_RUNTIME_UNAVAILABLE',NEW_TRAINING='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,
        QWEN_RUNTIME_IDENTITY_GATE='FAIL',GDN_RECURRENCE_SOURCE_IDENTITY_GATE='PASS',NATIVE_BASIS_MAPPING_GATE='NOT_EVALUATED',PHYSICAL_TRANSITION_ROTATION_INVARIANCE='NOT_EVALUATED',GDN_ZERO_WRITE_TRANSITION_PARITY_GATE='NOT_EVALUATED',GDN_DRIVEN_ERROR_RECURRENCE_GATE='NOT_EVALUATED',C128_QDQ_IDENTITY_GATE='PASS',FRESH_DOCUMENT_SELECTION_GATE='NOT_EVALUATED',DOCUMENT_OVERLAP_AUDIT_GATE='NOT_EVALUATED',ROTATION_PARAMETERIZATION_IDENTITY_GATE='PASS',ROTATION_BANK_COUNT=0,ROTATION_BANK_FREEZE_GATE='NOT_EVALUATED',ROTATION_BANK_ORTHOGONALITY_GATE='NOT_EVALUATED',ROTATION_BANK_UNIQUENESS_GATE='NOT_EVALUATED',PROPAGATION_MC_PROBE_COUNT='NOT_FROZEN',J_DECAY_AVAILABLE='YES',
        DISCOVERY_COMPLETE='0/48',DISCOVERY_JINJ_MEDIAN_RHO_2K='NA',DISCOVERY_JINJ_MEDIAN_RHO_8K='NA',DISCOVERY_JINJ_MEDIAN_RHO_32K='NA',DISCOVERY_JDECAY_MEDIAN_RHO_32K='NA',DISCOVERY_JPROP_MEDIAN_RHO_32K='NA',DISCOVERY_SELECTED_BEATS_JINJ_COUNT='NA',SELECTED_SURROGATE='NONE',DISCOVERY_SURROGATE_GATE='NOT_EVALUATED',CONFIRMATION_COMPLETE='0/48',CONFIRMATION_SELECTED_BEATS_JINJ_COUNT='NA',CONFIRMATION_32K_POSITIVE_RHO='NA',CONFIRMATION_32K_MEDIAN_RHO='NA',CONFIRMATION_32K_MEDIAN_CONCORDANCE='NA',QWEN_GDN_PROPAGATION_SURROGATE_CONFIRMATION='NOT_EVALUATED',CROSS_TERM_RESIDUAL='DESCRIPTIVE_ONLY_NOT_MEASURED',REASONING_IMPROVEMENT='NOT_EVALUATED',NEW_C8_TRAINING_AUTHORIZED='NO',PRIMARY_NEXT_STEP_CATEGORY='STOP',FINAL_ACTION='STOP')
    A.save('analysis/final_verdict.json',verdict)
    report='```text\n'+'\n'.join(k+' = '+str(v) for k,v in verdict.items())+'\n```\n\n'
    report+='## Outcome\n\nA800 access was restored in CPU-only mode. All four previously missing files were recovered through scp with exact SHA256 validation at source, local staging and target. The three frozen Delta exports and the three archived theta checkpoints are now retained in the new namespace. Original files, selections and previous Qwen verdicts were not modified.\n\n'
    report+='The original archived Cayley implementation reproduced all 72 selected layer matrices on GPU0 byte-exactly. Per-layer tensor SHAs match. Maximum orthogonality residual is '+str(max(x['orthogonality_max_abs'] for x in params['rows']))+' against the inherited 1e-4 gate. No inverse-Cayley parameter extraction, alternate parameterization or derived rotation was used.\n\n'
    report+='## Actual stop\n\nThe first development FP prefill failed in the unchanged Qwen `torch_chunk_gated_delta_rule`, source line291, `g = g.cumsum(dim=-1)`. PyTorch2.5.1+cu121 raises `cumsum_cuda_kernel does not have a deterministic implementation` under strict `torch.use_deterministic_algorithms(True)`. The model was loaded, but no complete forward or teacher-forced recurrent transition trace was produced. No parity-panel result or prospective scientific ranking was computed.\n\n'
    report+='Package/model/tokenizer/source identity checks PASS; the final runtime gate is FAIL specifically for strict deterministic execution feasibility, not a checkpoint integrity failure. No `warn_only=True`, nondeterministic exception, replacement cumsum, altered prefill path or automatic retry was applied. This runtime blockage is NOT a scientific rejection of propagation-aware metrics. An explicit separate deterministic-runtime compatibility decision is needed before resuming; preserve this stop evidence.\n\n'
    report+='## Source closure\n\n24 layer IDs and [1,32,128,128] state layout were verified against config and23 sealed parent captures; key16/value32/repeat2. Production FP32 recurrence decays, projects/erases, then applies beta-weighted key/value write. Canonical normalization precedes separate H and Delta transforms. Native mapping is recovered from the unchanged parent `physical_state` source (separate einsums, not fused H@Delta). C128 production quantizer source identity PASS; axis-2,128 coordinates,4096 groups, clamp_min(1e-12)/127, ties-to-even round,[-127,127] clip and actual dtype-preserving writeback. Numerical mapping, invariance, zero-write and driven-error gates remain NOT_EVALUATED. Source availability of a genuine per-head decay component is YES, but no DECAY/PROP score exists.\n\n'
    report+='## Engineering incident\n\nThe first attempted smoke was blocked before any model forward by an incorrectly transcribed65-character expected probe SHA. A CPU-only check showed the actual32-token SHA equals the historical64-character literal and the31 teacher tokens equal the sealed historical probe exactly. Attachment section42 permits premeasurement metadata fixes. Original script/policy/log/error remain preserved at commit dfcd2b8; an additive repaired script/policy and rationale were committed at8ef4ef4. Only the invalid metadata literal and versioned evidence paths changed; probe, operator, basis, QDQ and tolerance were not changed. The repaired attempt then encountered the strict cumsum runtime failure and was not retried.\n\n'
    report+='## Scope / integrity\n\nDiscovery0/48; Confirmation0/48. No fresh doc selected; no12-condition bank generated. No candidate correlations, rank concordances, horizon landscapes, residuals, reasoning endpoint, training, optimizer step, scorer call, free generation or AIME generation. No Ling outcome read. No old artifacts deleted or changed; parent106-file seal and original model/source/selection files revalidated. A800 ran no compute workload. All3090 GPUs were released naturally; no unrelated process was signalled. PREMEASUREMENT_COMMIT_SHA is NOT_CREATED: these are Phase A engineering commits, not a full Discovery authorization.\n\n'
    report+='## Q1-Q10\n\nQ1 instantaneous J_INJ ranking: NOT_EVALUATED.\n\nQ2 decay improvement: NOT_EVALUATED.\n\nQ3 full transition improvement beyond decay: NOT_EVALUATED.\n\nQ4 horizon scaling: NOT_EVALUATED.\n\nQ5 independent confirmation: NOT_RUN.\n\nQ6 H/C6R interpolation ordering: NOT_EVALUATED; bank not created.\n\nQ7 interaction residual/cancellation: NOT_MEASURED; no conclusion.\n\nQ8 physical transition invariance: NOT_EVALUATED; ideal-arithmetic reasoning is not a numerical gate.\n\nQ9 C8 justification: NO; no prospective confirmation and no C8 authorization.\n\nQ10 reasoning improvement: NO, not established.\n\n'
    report+='## Required-artifact completeness\n\nSource/parameter/recovery audits, prospective Phase A policies, both failed-attempt evidence sets, unavailable-gate records, stopped summaries, FINAL_REPORT and FINAL_SHA256SUMS are retained. No scientific raw CSV or trace was fabricated to fill missing measurements. The planned Discovery/Confirmation data artifacts are absent because the experiment stopped before measurement.\n'
    q=ROOT/'FINAL_REPORT.md'
    with q.open('x') as f:f.write(report);f.flush();os.fsync(f.fileno())
    q.chmod(0o444)
    # Compiled caches are engineering ephemera, not scientific output.
    m=ROOT/'FINAL_SHA256SUMS'
    with m.open('x') as f:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p!=m and '__pycache__' not in p.parts:f.write(A.sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    m.chmod(0o444)
    print(json.dumps(dict(status=verdict['TASK_STATUS'],root=str(ROOT),recovery='PASS',parameter_layers_exact=72,discovery='0/48',confirmation='0/48',FINAL_ACTION='STOP')),flush=True)

if __name__=='__main__':main()
