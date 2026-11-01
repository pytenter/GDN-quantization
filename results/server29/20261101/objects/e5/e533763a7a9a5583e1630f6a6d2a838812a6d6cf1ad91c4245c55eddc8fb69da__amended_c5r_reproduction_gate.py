#!/usr/bin/env python3
"""Non-AIME evidence only. No invented similarity score or missing historic Hidden."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
def main():
    evidence={}
    for rel in ('analysis/cross_objective_learning_curves.csv','analysis/shared16_comparison.csv','analysis/rotation_distances_descriptive.csv','analysis/shared16_capture_metrics.json','analysis/historical_reproduction_analysis.md'):
        evidence[rel]=sha(ROOT/rel)
    if any((ROOT/'aime26').rglob('*.jsonl')):raise RuntimeError('C5 reproduction route must precede new AIME')
    if not (ROOT/'training/C5R/summary.json').exists():
        import amended_training_entry as A
        if not A.isolation():raise RuntimeError('missing auxiliary training without isolation')
        result={'C5R_REPRODUCTION':'NOT_COMPLETED','C5R_AIME26_ROUTE':'NOT_REACHED','reason':'Preserved C5R-specific failure; shared validity separately reviewed. No C5R endpoint.'}
    else:
        summary=read(ROOT/'training/C5R/summary.json');selection=read(ROOT/'training/C5R/checkpoint_selection.json')['selected']
        evidence['training/C5R/summary.json']=sha(ROOT/'training/C5R/summary.json')
        evidence['training/C5R/checkpoint_selection.json']=sha(ROOT/'training/C5R/checkpoint_selection.json')
        with (ROOT/'analysis/cross_objective_learning_curves.csv').open() as f:curves=list(csv.DictReader(f))
        new=[r for r in curves if r['condition']=='C5R'];old=[r for r in curves if r['condition']=='historical_C5']
        with (ROOT/'analysis/shared16_comparison.csv').open() as f:panel={r['condition']:r for r in csv.DictReader(f)}
        if len(new)!=21 or len(old)!=20 or summary['status']!='PASS' or summary['updates']!=125 or summary['exposures']!=1000:raise RuntimeError('auxiliary evidence incomplete')
        for rows in (new,old):
            for r in rows:
                if not all(math.isfinite(float(r[k])) for k in ('state','functional')):raise RuntimeError('nonfinite evidence')
        # A single historical run supplies no repeatability/equivalence tolerance.
        # Exact structural comparability is evidence, not an arbitrary close/shift cut-off.
        result={'C5R_REPRODUCTION':'INCONCLUSIVE','C5R_AIME26_ROUTE':'SKIP_INCONCLUSIVE',
            'reason':'All eight evidence dimensions archived. No prospective historical repeatability envelope or justified qualitative close/material-shift decision exists; do not invent a weighted score or use AIME to resolve uncertainty.',
            'selected_update':selection['update'],'selected_exposure':selection['update']*8,
            'historical_deployed_selected_update':63,'historical_deployed_exposure':504,
            'historical_archived_state_reference':0.007916185,'historical_archived_functional_reference':0.049509294,
            'selected_shared16':panel['C5R'],'historical_C5_u063_on_clean_A800':panel['historical_C5_u063'],
            'historical_final_hidden_not_invented':True,'new_and_archived_validation_curves':{'new':new,'historical':old},
            'optimization_shape':{k:{'update0':float(new[0][k]),'minimum_scheduled':min(float(r[k]) for r in new[1:]),'last':float(new[-1][k]),'scheduled_decreases':sum(float(b[k])<float(a[k]) for a,b in zip(new,new[1:]))} for k in ('state','functional')},
            'numerical_stability':summary,'rotation_evidence':'analysis/rotation_distances_descriptive.csv; descriptive only'}
        review=ROOT/'analysis/C5R_NON_AIME_REPRODUCTION_REVIEW.json'
        if review.exists():
            r=read(review)
            if r.get('AIME_used') is not False or r.get('evidence_sha256')!=evidence or not r.get('rationale'):raise RuntimeError('review not supported by exact non-AIME evidence')
            category=r['C5R_REPRODUCTION']
            routes={'CLOSE_REPRODUCTION':'SKIP_NOT_NEEDED','MATERIAL_SHIFT':'RUN_DUE_TO_MATERIAL_SHIFT','INCONCLUSIVE':'SKIP_INCONCLUSIVE'}
            if category not in routes:raise RuntimeError('invalid reproduction classification')
            result.update(C5R_REPRODUCTION=category,C5R_AIME26_ROUTE=routes[category],reason=r['rationale'])
    result.update(status='PASS',evidence_sha256=evidence,no_AIME_used=True,amendment_reference='protocol/'+NAME+'.json',amendment_sha256=sha(ROOT/f'protocol/{NAME}.json'),mandatory_endpoint_conditions=['C6R','C7'])
    save(ROOT/'analysis/C5R_REPRODUCTION_GATE.json',result)
    with (ROOT/'analysis/C5R_REPRODUCTION_GATE.md').open('x') as f:
        f.write('# C5R non-AIME reproduction gate\n\nSelected update/exposure, State trajectory, Functional trajectory, selected State/Functional metrics, optimization shape, numerical stability and descriptive rotations are recorded below. Clean-A800 measurements of historical R are new comparable diagnostics, not invented historical Final-Hidden values. No assumption that skipped C5R equals historical 33/60.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    if result['C5R_AIME26_ROUTE']!='RUN_DUE_TO_MATERIAL_SHIFT':
        (ROOT/'aime26').mkdir(exist_ok=True)
        save(ROOT/'aime26/C5R_NOT_RUN.json',{'reason':result['reason'],'reproduction_status':result['C5R_REPRODUCTION'],'route':result['C5R_AIME26_ROUTE'],'amendment_reference':result['amendment_reference'],'C5R_accuracy':'NOT_RUN'})

if __name__=='__main__':main()
