#!/usr/bin/env python3
"""Compact descriptive Shared16 closure, protected-byte revalidation and one local commit. STOP."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import traceback
import posttrain_closure_audit as A

ROOT=A.ROOT
PANEL=('Hadamard','historical_C5_u063','historical_C5_u113','historical_C6_u125','C5R','C6R','C7')
METRICS=('state','functional','final_hidden')
REPORT='reports/QWEN_A800_POSTTRAIN_SHARED16_REPRODUCTION_CLOSURE_V1.md'

def table(rel,rows):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ranks(xs):
    sorted_indices=sorted(range(len(xs)),key=lambda i:xs[i]);out=[0.]*len(xs);i=0
    while i<len(xs):
        j=i+1
        while j<len(xs) and xs[sorted_indices[j]]==xs[sorted_indices[i]]:j+=1
        for k in range(i,j):out[sorted_indices[k]]=(i+j+1)/2
        i=j
    return out

def correlation(xs,ys):
    mx=statistics.mean(xs);my=statistics.mean(ys)
    dx=[x-mx for x in xs];dy=[y-my for y in ys]
    denominator=math.sqrt(sum(x*x for x in dx)*sum(y*y for y in dy))
    return sum(x*y for x,y in zip(dx,dy))/denominator if denominator else None

def sign(x):return (x>0)-(x<0)

def main():
    for rel in ('analysis/posttrain_hash_failure_audit.json','analysis/posttrain_runner_scientific_path_identity.json','provenance/posttrain_effective_provenance.json','analysis/frozen_rotation_export_audit.json','logs/posttrain_closure_runner.completion.json'):
        A.require(A.read(rel)['status']=='PASS','required gate '+rel)
    detail=A.read('analysis/shared16_capture_metrics.json');A.require(set(detail)==set(PANEL),'seven-arm panel coverage')
    prereg=A.read('protocol/preregistration.json');cache=A.read('teacher/cache_files.json')
    ids=prereg['validation']['document_ids'];positions={r['document_id']:r['positions'] for r in cache['files']}
    documents=[];captures=[];summary={};lookup={}
    for condition in PANEL:
        d=detail[condition];A.require(d['documents']==16 and d['captures']==128,'Shared16 count')
        A.require([r['document_id'] for r in d['per_document']]==ids,'Shared16 document order')
        lookup[condition]={}
        for doc in d['per_document']:
            name=doc['document_id'];A.require([r['position'] for r in doc['captures']]==positions[name] and len(doc['captures'])==8,'capture order/coverage')
            for i,c in enumerate(doc['captures']):
                A.require(all(math.isfinite(c[k]) and c[k]>=0 for k in METRICS),'nonfinite/negative relative-MSE')
                captures.append(dict(condition=condition,document_id=name,capture_index=i,position=c['position'],**{k:c[k] for k in METRICS}))
            mean={k:sum(c[k] for c in doc['captures'])/8 for k in METRICS}
            documents.append(dict(condition=condition,document_id=name,captures=8,**mean));lookup[condition][name]=mean
        summary[condition]={}
        for k in METRICS:
            xs=[lookup[condition][n][k] for n in ids];mean=sum(xs)/16
            A.require(mean==d[k],'overall metric aggregation differs from original evaluator')
            summary[condition][k]={'mean':mean,'document_median':statistics.median(xs),'document_min':min(xs),'document_max':max(xs)}
    A.require(len(documents)==112 and len(captures)==896,'compact output coverage')
    table('analysis/posttrain_shared16_per_document.csv',documents)
    table('analysis/posttrain_shared16_per_capture.csv',captures)
    # The canonical pipeline already defines shared16_comparison.csv/capture_metrics.json; retain and reference them.
    A.save('analysis/posttrain_shared16_summary.json',{'status':'PASS','POSTTRAIN_SHARED16_GATE':'PASS','conditions':summary,'documents_per_condition':16,'captures_per_document':8,'total_compact_capture_records':896,'canonical_summary':'analysis/shared16_comparison.csv','canonical_capture_records':'analysis/shared16_capture_metrics.json','teacher_forced':True,'new_AIME_generation':0,'metric_definition':'unchanged registered State, local Functional, final-normalized lm_head-input relative-MSE; equal 16 docs x 8 captures; epsilon1e-12','not_added':['KL','NLL','margin']})
    pair={}
    for k in METRICS:
        old=[lookup['historical_C5_u063'][n][k]-lookup['Hadamard'][n][k] for n in ids]
        new=[lookup['C5R'][n][k]-lookup['Hadamard'][n][k] for n in ids]
        rows=[{'document_id':n,'historical_C5_minus_H':o,'C5R_minus_H':v,'C5R_minus_historical_C5':v-o,'same_direction':sign(o)==sign(v)} for n,o,v in zip(ids,old,new)]
        pair[k]={'historical_relative_to_H_mean_delta':statistics.mean(old),'C5R_relative_to_H_mean_delta':statistics.mean(new),'historical_relative_to_H_fraction':detail['historical_C5_u063'][k]/detail['Hadamard'][k]-1,'C5R_relative_to_H_fraction':detail['C5R'][k]/detail['Hadamard'][k]-1,'sign_concordance_count':sum(r['same_direction'] for r in rows),'historical_improved_documents':sum(o<0 for o in old),'C5R_improved_documents':sum(v<0 for v in new),'Pearson_delta_association':correlation(old,new),'Spearman_delta_association':correlation(ranks(old),ranks(new)),'paired_shift_median':statistics.median(v-o for o,v in zip(old,new)),'paired_shift_min':min(v-o for o,v in zip(old,new)),'paired_shift_max':max(v-o for o,v in zip(old,new)),'documents':rows,'largest_absolute_paired_differences_descriptive_only':sorted(rows,key=lambda r:abs(r['C5R_minus_historical_C5']),reverse=True)[:4],'no_outliers_removed':True}
    oldroot=Path('/root/autodl-tmp/GDN-formal-c5-c6-h64-v1/experiments/qwen_gdn/recurrent_training/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1')
    history=json.loads((oldroot/'analysis/C5_validation_summary.json').read_text())
    hselected=next(c for c in history['candidates'] if c['update']==history['selected_update'])
    oldp=json.loads((oldroot/'preregistration.json').read_text());cfg=A.read('protocol/c5r_config.json')
    A.require(prereg['common']==oldp['training']['common'] and prereg['rotation']==oldp['rotation'],'protocol lineage drift')
    A.require(cfg['lr']==oldp['training']['C5']['lr'] and cfg['optimizer']=='Adam' and cfg['horizon']==64,'C5R protocol alignment')
    # Frozen prospective framework explicitly has no mechanical equivalence regime thresholds.
    A.require('no new numerical thresholds' in prereg['reproduction_interpretation'],'reproduction framework unexpectedly changed')
    with (ROOT/'analysis/rotation_distances_descriptive.csv').open() as f:geometry=list(csv.DictReader(f))
    geometry_summary={}
    for candidate,ref in [('C5R','historical_C5_u063'),('C5R','historical_C5_u113'),('C6R','historical_C6_u125')]:
        vals=[float(r['delta_frobenius_distance'])/float(r['reference_norm']) for r in geometry if r['candidate']==candidate and r['reference']==ref]
        A.require(len(vals)==24,'descriptive rotation coverage')
        geometry_summary[candidate+'_vs_'+ref]={'relative_Frobenius_min':min(vals),'relative_Frobenius_median':statistics.median(vals),'relative_Frobenius_max':max(vals),'descriptive_only':True}
    selection=A.read('analysis/posttrain_selected_checkpoint_identity.json')['conditions']
    rep={'status':'PASS','C5R_REPRODUCTION':'INCONCLUSIVE','C5R_AIME':'SKIP','reason':'No prospective mechanical criterion/equivalence envelope for objective direction/magnitude or 16-document association was frozen. Same-runtime evidence is complete; do not invent post-hoc 2%, 5% or correlation thresholds.','missing_prospective_criterion':['allowable objective-amplitude difference','required per-document sign/rank concordance','repeatability/equivalence envelope'],'Layer_A_protocol_reproduction':{'status':'PASS','common_config_exact':True,'data_order_seed_H64_captures_optimizer_exact':True,'C5R_LR_matches_historical':True,'Hadamard_init_State_objective_QDQ_writeback':'unchanged frozen implementations audited; training readiness PASS'},'Layer_B_same_runtime_objectives':{c:{k:detail[c][k] for k in METRICS} for c in ('Hadamard','historical_C5_u063','C5R')},'Layer_C_document_level_patterns':pair,'Layer_D_geometry_descriptive_only':geometry_summary,'matrix_distance_alone_did_not_determine_verdict':True,'historical_selected_update_exposure':[63,504],'C5R_selected_update_exposure':[selection['C5R']['update'],selection['C5R']['exposure']],'historical_OLD_RUNTIME_background':{'state':hselected['validation']['state'],'functional':hselected['validation']['functional'],'final_hidden':'NOT_MEASURED_IN_OLD_SUMMARY','sha256':A.sha(oldroot/'analysis/C5_validation_summary.json')},'historical_clean_A800_measurements_new_not_invented_old_hidden':True,'no_AIME_used':True,'all_conditions_still_NOT_STARTED_for_AIME':True}
    A.save('analysis/c5r_reproduction_analysis.json',rep)
    comparison={k:{'C6R':detail['C6R'][k],'C7':detail['C7'][k],'C7_minus_C6R':detail['C7'][k]-detail['C6R'][k],'fractional_change':detail['C7'][k]/detail['C6R'][k]-1,'per_document_paired_delta':[{'document_id':n,'delta':lookup['C7'][n][k]-lookup['C6R'][n][k]} for n in ids]} for k in METRICS}
    A.save('analysis/c6r_c7_shared16_comparison.json',{'status':'PASS','C6R_role':'PRIMARY_MATCHED_CONTROL','C7_role':'PRIMARY_NEW_OBJECTIVE','metrics':comparison,'interpretation_boundary':'Shared16 fidelity is not reasoning accuracy. C7 vs C6R Canonical60 remains unrun and requires separate authorization.','AIME_started':False})
    pre=A.read('provenance/preclosure_training_artifact_inventory.json');checked=[]
    for i,row in enumerate(pre['files']):
        actual=A.identity(row['path'],row['sha256'])
        A.require(actual==row,'protected size or mtime changed: '+row['path']);checked.append(actual)
        if i%30==0:print(json.dumps({'event':'protected_revalidation','files':i+1}),flush=True)
    A.save('analysis/postclosure_training_artifact_revalidation.json',{'status':'PASS','TRAINING_ARTIFACT_BYTE_IDENTITY':'PASS','files':checked,'count':len(checked),'all_sizes_hashes_mtimes_unchanged':True,'preinventory_sha256':A.sha(ROOT/'provenance/preclosure_training_artifact_inventory.json'),'no_existing_export_rewritten':True})
    gate={'EXPERIMENT':ROOT.name,'CLOSURE':A.TASK,'C5R_TRAINING':'COMPLETE / PASS','C6R_TRAINING':'COMPLETE / PASS','C7_TRAINING':'COMPLETE / PASS','TRAINING_RERUN':'NO','HASH_FAILURE_AUDIT':'PASS','HASH_CONSTANT_RECORD_ERROR':'SUPPORTED','ARTIFACT_MUTATION_SUPPORTED':'NO','POSTTRAIN_RUNNER_SCIENTIFIC_PATH_IDENTITY_GATE':'PASS','POSTTRAIN_EFFECTIVE_PROVENANCE_GATE':'PASS','POSTTRAIN_SHARED16_GATE':'PASS','QWEN_FINAL_HIDDEN_INDEXING_GATE':'PASS','C5R_REPRODUCTION':'INCONCLUSIVE','C5R_AIME':'SKIP','C6R_FROZEN_ROTATION_EXPORT':'PASS','C7_FROZEN_ROTATION_EXPORT':'PASS','C6R_EXPORT_PARITY_GATE':'PASS','C7_EXPORT_PARITY_GATE':'PASS','TRAINING_ARTIFACT_BYTE_IDENTITY':'PASS','3090_DEPLOYMENT_STARTED':'NO','C5R_AIME_STARTED':'NO','C6R_AIME_STARTED':'NO','C7_AIME_STARTED':'NO','NEW_AIME_GENERATION':0,'PUSH_PERFORMED':'NO'}
    report=['```text']+[f'{k} = {v}' for k,v in gate.items()]+['```','','# Shared16 post-training closure','','All rows below are direct same-runtime trusted A800 clean-teacher measurements, not AIME. Frozen seven-condition panel preserved.','','| Condition | State | Functional | Final Hidden |','|---|---:|---:|---:|']
    report += [f"| {c} | {detail[c]['state']:.12g} | {detail[c]['functional']:.12g} | {detail[c]['final_hidden']:.12g} |" for c in PANEL]
    report += ['','## Historical background (different runtime, not direct comparison)','',f"Historical C5 selected u63/e504: State {hselected['validation']['state']:.12g}, Functional {hselected['validation']['functional']:.12g}. Old Final Hidden was not recorded.",'','## C5R reproduction','','C5R selected u113/e904; historical u63/e504. Layer A protocol reproduction PASS. Layers B/C supply full paired objective and document evidence; Layer D is descriptive only. Matrix distance alone did not determine verdict.','',rep['reason'],'','| Metric | Historical C5 vs H | C5R vs H | Sign agreement /16 | Pearson | Spearman |','|---|---:|---:|---:|---:|---:|']
    for k,r in pair.items():report.append(f"| {k} | {r['historical_relative_to_H_fraction']:+.6%} | {r['C5R_relative_to_H_fraction']:+.6%} | {r['sign_concordance_count']} | {r['Pearson_delta_association']} | {r['Spearman_delta_association']} |")
    report += ['','All 16 documents retained. Per-capture/per-document CSVs contain 896/112 rows; all values finite. Largest paired differences, exact document IDs and complete signed distributions are recorded in c5r_reproduction_analysis.json. No outliers removed and no thresholds invented.','',json.dumps(geometry_summary,indent=2),'','## C6R / C7 and exports','','C6R is the primary matched control, C7 the primary new objective. Compare their three Shared16 fidelities only; no reasoning superiority claim. The selected existing 24x128x128 FP32 exports passed per-layer SHA, frozen orthogonality threshold and exact A800 checkpoint-loaded matrix parity. No export rewritten, no selection changed.','','## Hash failure and scientific path','','All three archived EXPECTED constants were malformed 65-character hex strings. GNU sha256sum/Python hashlib agree with independent 3090 copies and preexisting identity records. Old failing source and logs remain byte-identical. Append-only amendment fixes EXPECTED resolution only. The original archived main, payload, model loader, teacher cache, evaluator, QDQ, writeback and metrics are used unchanged; audit output names differ to preserve failure records.','',f"Protected inventory: {len(checked)} files / {sum(x['size'] for x in checked)} bytes, exact hash/size/mtime revalidated. Teachers were not rebuilt.",'','## Provenance / scope / commit','','Branch: '+pre['pre_task_branch']+'. Pre-task HEAD: '+A.HEAD+'. One local closure commit is created only after schema/hash/diff review; its SHA is in logs/posttrain_closure_commit.json (excluded from pre-commit hash manifest to avoid self-reference). No push.','',f"Host/GPU/driver: {pre['hostname']} / {' / '.join(pre['gpu'][:3])}. Python {pre['python']}; Torch {pre['versions']['torch']}, CUDA12.1. Fresh direct CPU checkpoint and all 429 full GPU state items verified before any Shared16 computation.",'','Reproduction commands: scripts/posttrain_closure_pipeline.sh (one-shot; existing evidence blocks repetition). Sources, audits and raw compact evidence are indexed by manifests/posttrain_closure_artifact_sha256.txt.','','STOP: no training/backward/optimizer, no transfer of new R to3090, no deployment, no AIME, no scoring. Even PASS does not authorize endpoints.']
    p=ROOT/REPORT;p.parent.mkdir(parents=True,exist_ok=True);p.open('x').write('\n'.join(report)+'\n')
    A.save('analysis/posttrain_closure_gates.json',dict(gate,status='PASS',stop='WAIT_FOR_SEPARATE_USER_AUTHORIZATION'))
    # Parse ALL current JSON and validate all newly produced metric CSV schemas before freezing.
    for p in ROOT.rglob('*.json'):json.loads(p.read_text())
    for rel,count,fields in [('analysis/posttrain_shared16_per_document.csv',112,('condition','document_id','captures',*METRICS)),('analysis/posttrain_shared16_per_capture.csv',896,('condition','document_id','capture_index','position',*METRICS))]:
        with (ROOT/rel).open() as f:w=csv.DictReader(f);A.require(tuple(w.fieldnames)==fields,'CSV schema');A.require(len(list(w))==count,'CSV count')
    # Complete closure hashes, including every protected file. Exclude only this manifest and open operational log.
    paths={Path(r['path']) for r in checked}
    paths.update(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('posttrain_closure_pipeline.log','posttrain_closure_artifact_sha256.txt'))
    manifest=ROOT/'manifests/posttrain_closure_artifact_sha256.txt';manifest.parent.mkdir(exist_ok=True)
    with manifest.open('x') as f:
        for p in sorted(paths):f.write(A.sha(p)+'  '+str(p)+'\n')
    for line in manifest.read_text().splitlines():
        digest,path=line.split('  ',1);A.require(A.sha(path)==digest,'hash manifest revalidation')
    A.require(A.command(['git','rev-parse','HEAD'])==A.HEAD,'unexpected intervening commit')
    # Only closure-specific files. Prior untracked amended controllers remain untouched and unstaged.
    prefix=str(ROOT.relative_to(A.REPO));files=[prefix+'/scripts/'+s for s in A.NEW_SCRIPTS]
    files.append(prefix+'/scripts/amended_a800_posttraining.py')  # Exact archived failed source is a closure dependency, not edited.
    files += [str(p.relative_to(A.REPO)) for p in ROOT.rglob('*') if p.is_file() and ('posttrain_' in p.name or p.name in ('c5r_reproduction_analysis.json','c6r_c7_shared16_comparison.json','frozen_rotation_export_audit.json','QWEN_A800_POSTTRAIN_SHARED16_REPRODUCTION_CLOSURE_V1.md')) and p.suffix not in ('.log','.pyc')]
    files += [str(p.relative_to(A.REPO)) for p in (ROOT/'provenance/posttrain_reference_evidence').rglob('*') if p.is_file()]
    files += [prefix+'/'+r for r in ('analysis/shared16_comparison.csv','analysis/shared16_capture_metrics.json','analysis/rotation_distances_descriptive.csv','analysis/historical_reproduction_analysis.md','deployment/selected_artifacts_pre_endpoint.json')]
    files += [prefix+'/'+r for r in ('provenance/preclosure_training_artifact_inventory.json','analysis/postclosure_training_artifact_revalidation.json')]
    files=sorted(set(files));A.require(not A.command(['git','diff','--cached','--name-only']),'preexisting staged changes')
    subprocess.run(['git','add','-f','--',*files],cwd=A.REPO,check=True)
    diff=A.command(['git','diff','--cached','--check']);stat=A.command(['git','diff','--cached','--stat'])
    staged=A.command(['git','diff','--cached','--name-only']).splitlines();A.require(set(staged)==set(files),'unexpected staged scope')
    A.require(not A.command(['git','diff','--name-only']),'unexpected tracked unstaged mutation')
    subprocess.run(['git','commit','-m','Close A800 post-training Shared16 provenance and reproduction; stop before endpoints'],cwd=A.REPO,check=True)
    A.save('logs/posttrain_closure_commit.json',{'status':'PASS','branch':A.command(['git','branch','--show-current']),'pre_task_HEAD':A.HEAD,'new_commit_SHA':A.command(['git','rev-parse','HEAD']),'git_status':A.command(['git','status','--short']),'diff_check':diff,'diff_stat':stat,'staged_scope':staged,'artifact_root':str(ROOT),'push_performed':False,'stop':True})
    print(json.dumps({'event':'closure_complete','gates':gate,'commit':A.command(['git','rev-parse','HEAD'])}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        A.save('logs/posttrain_closure_finalize.error.json',{'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'do_not_retry':True});raise
