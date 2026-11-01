"""Frozen terminal reporting; never starts training, generation, or AIME."""
from common import *
import statistics

def main():
    selection=load(D/'analysis/discovery_selection.json')
    discovery=load(D/'analysis/discovery_ranking.json')
    selected=selection['SELECTED_PROPAGATION_SURROGATE']
    confirmation=load(D/'analysis/confirmation_ranking.json') if selected!='NONE' else None
    if selected=='NONE':assert not list((D/'raw/CONFIRMATION').glob('*.json'))
    cgate=confirmation['gates'][selected] if confirmation else None
    supported=cgate is not None and cgate['gate']=='PASS'
    verdict='SUPPORTED' if supported else 'NOT_SUPPORTED' if confirmation else 'NOT_EVALUATED'
    parent=verify_parent();rt=runtime();hardware()
    inv=load(D/'manifests/premeasurement_sha256.json')
    for f in inv['files']:assert sha(f['path'])==f['sha256'],f['path']
    for phase in ['DISCOVERY']+(['CONFIRMATION'] if confirmation else []):
        raw=load(D/'manifests'/(phase.lower()+'_raw_freeze.json'))
        for f in raw['files']:assert sha(f['path'])==f['sha256']
    def val(key):return cgate[key] if cgate else 'NOT_EVALUATED'
    header=dict(EXPERIMENT=TASK,EVIDENCE_TYPE='PROSPECTIVE_LONG_HORIZON_SURROGATE_VALIDATION',NEW_TRAINING='NO',NEW_AIME_GENERATION=0,PARENT_IMMUTABILITY_GATE=parent['gate'],DETERMINISTIC_RUNTIME_IDENTITY_GATE=rt['gate'],KDA_TRANSITION_SOURCE_IDENTITY_GATE='PASS',ZERO_WRITE_TRANSITION_PARITY_GATE='PASS',P1_P2_METRIC_IDENTITY_GATE='PASS',FRESH_DOCUMENT_SELECTION_GATE='PASS',OVERLAP_AUDIT_GATE='PASS',ROTATION_BANK_COUNT=16,ROTATION_BANK_FREEZE_GATE='PASS',ROTATION_BANK_ORTHOGONALITY_GATE='PASS',ROTATION_BANK_UNIQUENESS_GATE='PASS',PROPAGATION_MC_PROBE_COUNT=load(D/'protocol/frozen_metrics_and_execution.json')['probe_count'],DISCOVERY_COMPLETE='64/64',DISCOVERY_JINJ_MEDIAN_SPEARMAN=discovery['all_panel_median_spearman']['J_INJ'],DISCOVERY_JDECAY_MEDIAN_SPEARMAN=discovery['all_panel_median_spearman']['J_DECAY'],DISCOVERY_JPROP_MEDIAN_SPEARMAN=discovery['all_panel_median_spearman']['J_PROP'],SELECTED_PROPAGATION_SURROGATE=selected,DISCOVERY_SURROGATE_GATE=selection['DISCOVERY_SURROGATE_GATE'],CONFIRMATION_COMPLETE='64/64' if confirmation else '0/64',CONFIRMATION_SELECTED_BEATS_JINJ_COUNT=f"{val('beats_JINJ_count')}/12" if cgate else 'NOT_EVALUATED',CONFIRMATION_32768_POSITIVE_SPEARMAN=f"{val('positive_32768_count')}/4" if cgate else 'NOT_EVALUATED',CONFIRMATION_32768_MEDIAN_SPEARMAN=val('median_32768_spearman'),CONFIRMATION_32768_MEDIAN_CONCORDANCE=val('median_32768_concordance'),PROPAGATION_SURROGATE_CONFIRMATION=verdict,CROSS_TERM_RESIDUAL='DESCRIPTIVE_ONLY',NEW_TRAINING_AUTHORIZED='NO',AIME_AUTHORIZED='NO',PRIMARY_NEXT_STEP_CATEGORY='PREREGISTER_PROPAGATION_AWARE_MATCHED_TRAINING' if supported else 'STOP_PROPAGATION_OBJECTIVE_TRAINING',FINAL_ACTION='STOP',PREMEASUREMENT_COMMIT_SHA=load(D/'run/premeasurement_commit.json')['commit'])
    if supported:
        emit('proposals/LING_PROPAGATION_AWARE_MATCHED_TRAINING_V1.json',dict(proposal_only=True,execution_authorized=False,selected_surrogate=selected,control='L7R-style Local Functional +0.1 State',intervention='same Local Functional +0.1 State +lambda*selected surrogate',lambda_value='NOT_SELECTED; requires separate prospective calibration preregistration',matched=['train data','validation','seed','optimizer','LR','steps','layer learnability','INT8-R128','deterministic runtime'],only_objective_changes=True,reasoning_improvement='NOT_TESTED',surrogate_semantics='full-transition propagation-aware, not raw injection' if selected=='J_PROP' else 'decay-only lifetime approximation',FINAL_ACTION='STOP'))
    horizons=[]
    residual=[]
    for phase in ['DISCOVERY']+(['CONFIRMATION'] if confirmation else []):
        summary=discovery if phase=='DISCOVERY' else confirmation
        for h in [2048,8192,32768]:
            ps=[p for p in summary['panels'] if p['horizon']==h]
            rho={s:statistics.median([p[s]['spearman'] for p in ps]) if all(p[s]['spearman'] is not None for p in ps) else None for s in ['J_INJ','J_DECAY','J_PROP']}
            horizons.append(dict(phase=phase,horizon=h,median_spearman=rho,median_improvement_over_JINJ={s:statistics.median([p[s]['spearman']-p['J_INJ']['spearman'] for p in ps]) if all(p[s]['spearman'] is not None and p['J_INJ']['spearman'] is not None for p in ps) else None for s in ['J_DECAY','J_PROP']}))
        for f in load(D/'manifests'/(phase.lower()+'_raw_freeze.json'))['files']:
            x=load(f['path']);s=x['scores']['32768']
            residual.append(dict(phase=phase,document_id=x['document_id'],condition=x['condition_id'],J_ACC=s['J_ACC'],J_PROP=s['J_PROP'],residual=s['CROSS_TERM_RESIDUAL'],absolute_fraction_of_JACC=abs(s['CROSS_TERM_RESIDUAL'])/(s['J_ACC']+1e-12)))
    q={
        'Q1':'Raw J_INJ discrimination is reported as exact panel Spearman and concordance, not assumed weak. See frozen phase-specific tables.',
        'Q2':f"Discovery J_DECAY rules: {discovery['gates']['J_DECAY']}. Confirmation evidence is separate, when authorized.",
        'Q3':f"Discovery J_PROP rules: {discovery['gates']['J_PROP']}. It uses the full actual zero-write transition, not decay-only.",
        'Q4':'See median improvement by horizon below. This is descriptive; no post-hoc threshold, pooled gate, or significance claim.',
        'Q5':f'Independent confirmation: {verdict}. No discovery/confirmation pooling. '+('No candidate advanced, so confirmation was not measured.' if not confirmation else str(cgate)),
        'Q6':'Residual magnitudes and fractions are descriptive only. Finite signed-probe variance and rotated BF16 write rounding can contribute in addition to cross-time alignment/cancellation. The residual is not an exact isolated cross-term.',
        'Q7':'Supported for a separately preregistered objective-only matched training proposal; no training authorized.' if supported else 'Not enough supported independent evidence to advance propagation-objective training; STOP.',
        'Q8':'NO. No training, optimizer updates, AIME or reasoning endpoint was measured.'}
    report=dict(verdict=header,questions=q,horizon_description=horizons,cross_term_description=residual,parent_validation=parent,runtime_validation=rt,discovery_gates=discovery['gates'],confirmation_gate=cgate,SCIENTIFIC_OBJECTS_CHANGED_AFTER_DISCOVERY='NO')
    emit('reports/FINAL_REPORT.json',report)
    lines=['# Ling Propagation Surrogate Validation V1','', '```ini']+[f'{k} = {v}' for k,v in header.items()]+['```','','## System boundary','','ON_POLICY_WITH_RESPECT_TO_RECURRENT_STATE_AND_QDQ UNDER_A_FROZEN_TEACHER_FORCED_TRANSITION_TRACE. Not full-model autoregressive on-policy.','']
    for k,v in q.items():lines += ['## '+k,'',v,'']
    lines+=['## Frozen horizon descriptions','','```json',json.dumps(horizons,indent=2,allow_nan=False),'```','','## Evidence','','Raw per-document/per-condition metrics: `raw/{DISCOVERY,CONFIRMATION}/`. Phase-specific rankings and ties: `analysis/*_ranking.json`. Raw inventories precede all ranking aggregation: `manifests/*_raw_freeze.json`. Metric and conservative all-120-pair concordance definitions: `protocol/frozen_metrics_and_execution.json`. Full residual table: `reports/FINAL_REPORT.json`. Parent validation is byte-level over the recorded five original trees. No inference endpoint or AIME data was prepared.','', 'FINAL_ACTION = STOP','']
    p=D/'reports/FINAL_REPORT.md'
    with p.open('x') as f:f.write('\n'.join(lines))
    emit('provenance/COMPLETION.json',dict(gate='PASS',final_verdict=header,parent_immutability=parent,report=row(p),TRAINING_STARTED='NO',OPTIMIZER_STEPS=0,FREE_GENERATION=0,AIME_STARTED='NO',NEW_ROTATION_LEARNING='NO',FINAL_ACTION='STOP',time_unix=time.time()))
    inventory=[row(p) for folder in ['raw','analysis','reports','proposals','provenance'] for p in files(D/folder)]
    emit('manifests/final_sha256_inventory.json',dict(gate='PASS',files=inventory))
    print(json.dumps(header,indent=2),flush=True)

if __name__=='__main__':main()
