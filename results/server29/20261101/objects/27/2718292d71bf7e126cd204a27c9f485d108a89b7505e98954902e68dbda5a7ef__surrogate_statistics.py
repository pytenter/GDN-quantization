"""Frozen ranking/advancement implementation; no inferential p-values."""
from common import *
import math,statistics,itertools
CANDIDATES=['J_DECAY','J_PROP']

def ranks(xs):
    ordered=sorted(range(len(xs)),key=lambda i:xs[i]);out=[0.]*len(xs);j=0
    while j<len(xs):
        k=j+1
        while k<len(xs) and xs[ordered[k]]==xs[ordered[j]]:k+=1
        for i in ordered[j:k]:out[i]=(j+1+k)/2
        j=k
    return out

def spearman(xs,ys):
    a=ranks(xs);b=ranks(ys);ma=statistics.mean(a);mb=statistics.mean(b)
    x=[v-ma for v in a];y=[v-mb for v in b]
    den=(sum(v*v for v in x)*sum(v*v for v in y))**.5
    return sum(u*v for u,v in zip(x,y))/den if den else None

def order(xs,ys):
    counts=dict(concordant=0,discordant=0,surrogate_tie_only=0,ground_truth_tie_only=0,both_tied=0)
    sign=lambda x:1 if x>0 else -1 if x<0 else 0
    for i,j in itertools.combinations(range(len(xs)),2):
        a=sign(xs[i]-xs[j]);b=sign(ys[i]-ys[j])
        key='both_tied' if not a and not b else 'surrogate_tie_only' if not a else 'ground_truth_tie_only' if not b else 'concordant' if a==b else 'discordant'
        counts[key]+=1
    total=sum(counts.values());assert total==120
    # Primary denominator is all 120 pairs, with ties as non-concordant.
    # This conservative rule is frozen before any scientific outcome.
    return dict(**counts,total_pairs=total,concordance=counts['concordant']/total)

def gate(panels,name):
    assert len(panels)==12
    allrho=[p[name]['spearman'] for p in panels]
    long=[p for p in panels if p['horizon']==32768];assert len(long)==4
    longrho=[p[name]['spearman'] for p in long]
    defined=all(v is not None for v in allrho)
    beats=sum(p[name]['spearman'] is not None and p['J_INJ']['spearman'] is not None and p[name]['spearman']>p['J_INJ']['spearman'] for p in panels)
    positive=sum(v is not None and v>0 for v in longrho)
    mr=statistics.median(longrho) if all(v is not None for v in longrho) else None
    mc=statistics.median([p[name]['order']['concordance'] for p in long])
    rules=dict(A=beats>=9,B=positive>=3,C=mr is not None and mr>=.5,D=mc>=.65)
    return dict(gate='PASS' if defined and all(rules.values()) else 'FAIL',all_panels_defined=defined,rules=rules,beats_JINJ_count=beats,positive_32768_count=positive,median_32768_spearman=mr,median_all_panel_spearman=statistics.median(allrho) if defined else None,median_32768_concordance=mc)

def freeze_raw(phase):
    docs=[d for d in load(D/'protocol/frozen_documents.json')['streams'] if d['phase']==phase]
    bank=load(D/'protocol/frozen_rotation_bank.json')
    paths=[D/'raw'/phase/(d['document_id']+'__'+c['condition_id']+'.json') for d in docs for c in bank['conditions']]
    assert len(paths)==64 and set(paths)==set((D/'raw'/phase).glob('*.json'))
    # Inventory is produced BEFORE any numerical raw result is loaded here.
    frozen=dict(gate='PASS',complete=64,phase=phase,files=[row(p) for p in paths],before_aggregation=True)
    p=D/'manifests'/(phase.lower()+'_raw_freeze.json');emit(p,frozen);p.chmod(0o444)
    return frozen

def analyze(phase):
    frozen=freeze_raw(phase);records=[]
    commit=load(D/'run/premeasurement_commit.json')['commit']
    docs={d['document_id']:d for d in load(D/'protocol/frozen_documents.json')['streams'] if d['phase']==phase}
    bank=load(D/'protocol/frozen_rotation_bank.json');conditions=bank['execution_order']
    rotation_sha={c['condition_id']:c['artifact']['sha256'] for c in bank['conditions']}
    for f in frozen['files']:
        assert sha(f['path'])==f['sha256'];x=load(f['path']);d=docs[x['document_id']]
        assert x['gate']=='PASS' and x['task']==TASK and x['phase']==phase
        assert x['physical_gpu']==d['physical_gpu'] and x['gpu_uuid']==d['gpu_uuid'] and x['selection_rank']==d['selection_rank']
        assert x['token_ids_sha256']==d['token_ids_sha256'] and x['rotation_sha256']==rotation_sha[x['condition_id']]
        assert x['premeasurement_commit']==commit and x['source_manifest_sha256']==sha(D/'manifests/premeasurement_sha256.json')
        assert x['tokens_consumed']==32768 and x['single_rollout_all_horizons'] and x['capture_replay_bitwise_parity']=='PASS'
        assert x['probe_count']==load(D/'protocol/frozen_metrics_and_execution.json')['probe_count'] and x['sign_schedule_sha256']==sha(D/'protocol/probe_signs.pt')
        assert sha(D/'run/traces'/x['document_id']/'trace_freeze.json')==x['trace_sha256']
        records.append(x)
    panels=[]
    for doc in docs:
        rr={x['condition_id']:x for x in records if x['document_id']==doc};assert set(rr)==set(conditions)
        for h in [2048,8192,32768]:
            ys=[rr[c]['scores'][str(h)]['J_ACC'] for c in conditions]
            assert all(math.isfinite(v) and v>=0 for v in ys)
            p=dict(document_id=doc,horizon=h,conditions=conditions,ground_truth=ys)
            for name in ['J_INJ']+CANDIDATES:
                xs=[rr[c]['scores'][str(h)][name] for c in conditions];assert all(math.isfinite(v) and v>=0 for v in xs)
                p[name]=dict(scores=xs,spearman=spearman(xs,ys),order=order(xs,ys))
            panels.append(p)
    gates={name:gate(panels,name) for name in CANDIDATES}
    summary=dict(phase=phase,complete=64,raw_freeze_sha256=sha(D/'manifests'/(phase.lower()+'_raw_freeze.json')),panels=panels,gates=gates,all_panel_median_spearman={name:statistics.median([p[name]['spearman'] for p in panels]) if all(p[name]['spearman'] is not None for p in panels) else None for name in ['J_INJ']+CANDIDATES})
    emit('analysis/'+phase.lower()+'_ranking.json',summary)
    if phase=='DISCOVERY':
        passed=[name for name in CANDIDATES if gates[name]['gate']=='PASS']
        def selection_key(name):
            x=gates[name];return (x['median_32768_spearman'],x['median_all_panel_spearman'],x['median_32768_concordance'],name=='J_DECAY')
        selected=max(passed,key=selection_key) if passed else 'NONE'
        emit('analysis/discovery_selection.json',dict(DISCOVERY_SURROGATE_GATE='PASS' if passed else 'FAIL',SELECTED_PROPAGATION_SURROGATE=selected,candidates=gates,rule='long-rho, all-rho, long-concordance, exact tie J_DECAY',raw_freeze_sha256=summary['raw_freeze_sha256'],ranking_sha256=sha(D/'analysis/discovery_ranking.json')))
    return summary

def tests():
    assert ranks([1,1,3,2])==[1.5,1.5,4.,3.]
    a=list(range(16));assert spearman(a,a)==1 and spearman(a,a[::-1])==-1
    assert spearman([0]*16,a) is None
    assert order(a,a)['concordance']==1 and order(a,a[::-1])['discordant']==120
    ties=order([0]*16,[0]*16);assert ties['both_tied']==120 and ties['concordance']==0
    panels=[dict(horizon=h,J_INJ=dict(spearman=.1),J_PROP=dict(spearman=.5,order=dict(concordance=.65))) for h in [2048,8192,32768] for d in range(4)]
    assert gate(panels,'J_PROP')['gate']=='PASS'
    for i in range(3):panels[i]['J_PROP']['spearman']=.1
    assert gate(panels,'J_PROP')['gate']=='PASS'
    panels[3]['J_PROP']['spearman']=.1;assert gate(panels,'J_PROP')['gate']=='FAIL'
    emit('runtime/statistical_unit_tests.json',dict(gate='PASS',synthetic_only=True,tie_ranking='average ranks exact equality',constant_panel='undefined and fails candidate',concordance='concordant/120; ties reported separately and do not count',thresholds_unchanged=True))

if __name__=='__main__':tests()
