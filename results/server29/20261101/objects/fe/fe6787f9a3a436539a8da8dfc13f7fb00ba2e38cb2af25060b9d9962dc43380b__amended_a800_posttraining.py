#!/usr/bin/env python3
"""Fixed pre-endpoint common diagnostics. No training or selection changes."""
import csv
import json
import math
from pathlib import Path
import time
import traceback
import a800_matched as E

ROOT=E.ROOT
ALL_ARMS=E.ARMS
if not (ROOT/'training/C5R/summary.json').exists():
    import amended_training_entry as A
    if not A.isolation():raise RuntimeError('auxiliary failure not isolated')
    E.ARMS=('C6R','C7')
EXPECTED={
    'C5_u063':'0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d7996c86f857c9',
    'C5_u113':'b625e99dd7f57a87040e517d263555e5366de6bba1678d221ec3cba287fafe023',
    'C6_u125':'67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b2894426990a7b09f39c'}


def table(path,rows,fields):
    E.once(path)
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def payload(path,digest):
    if E.S(path)!=digest:raise RuntimeError('fixed rotation source hash mismatch: '+str(path))
    x=E.torch.load(path,map_location='cpu',weights_only=False)
    if tuple(x['layer_ids'])!=E.core.GDN_LAYERS or set(x['matrices'])!={str(l) for l in E.core.GDN_LAYERS}:
        raise RuntimeError('fixed rotation layer mapping mismatch')
    for l,v in x['matrices'].items():
        if v.dtype!=E.torch.float32 or v.shape!=(128,128) or not bool(E.torch.isfinite(v).all()):raise RuntimeError('fixed matrix structure')
    return x


def main():
    E.once(ROOT/'logs/posttraining.completion.json');E.once(ROOT/'logs/posttraining.error.json')
    for arm in E.ARMS:E.require(f'training/{arm}/summary.json');E.require(f'logs/train_{arm}.completion.json')
    # Fixed sources, fixed seven-arm Shared16 panel. C5u113 is explicitly noncanonical.
    manifests={};banks={};curves=[];histories={}
    for arm in E.ARMS:
        selected=json.loads((ROOT/f'training/{arm}/checkpoint_selection.json').read_text())['selected']
        manifests[arm]=selected
        x=payload(Path(selected['frozen_delta_path']),selected['frozen_delta_sha256'])
        banks[arm]=x['matrices']
        zero=json.loads((ROOT/f'analysis/{arm.lower()}_update0_validation.json').read_text())['validation']
        curves.append(dict(condition=arm,update=0,exposure=0,**{k:zero[k] for k in E.METRICS}))
        for line in (ROOT/f'training/{arm}/train_log.jsonl').read_text().splitlines():
            r=json.loads(line)
            if 'validation' in r:curves.append(dict(condition=arm,update=r['update'],exposure=r['exposure'],**{k:r['validation'][k] for k in E.METRICS}))
    for condition in ('C5','C6'):
        path=E.M.OLD/f'analysis/{condition}_validation_summary.json'
        old=json.loads(path.read_text());histories[condition]={'sha256':E.S(path),'selected_update':old['selected_update'],'selected_primary':old['selected_primary'],'primary_metric':old['primary_metric']}
        for c in old['candidates']:
            curves.append({'condition':'historical_'+condition,'update':c['update'],'exposure':c['target_exposures'],
                'state':c['validation']['state'],'functional':c['validation']['functional'],'final_hidden':''})
    table(ROOT/'analysis/cross_objective_learning_curves.csv',curves,['condition','update','exposure','state','functional','final_hidden'])
    for name,digest in EXPECTED.items():
        path=ROOT/f'provenance/historical_rotations/{name}.pt';banks['historical_'+name]=payload(path,digest)['matrices']
    model,tokenizer,sources,device=E.load('posttraining')
    panel=('Hadamard','historical_C5_u063','historical_C5_u113','historical_C6_u125','C5R','C6R','C7')
    panel=tuple(c for c in panel if c!='C5R' or 'C5R' in E.ARMS)
    comparisons=[];detail={};distances=[]
    for condition in panel:
        bank=(E.fresh_bank(device) if condition=='Hadamard' else E.runtime.FrozenMatrixRotationBank(E.core.GDN_LAYERS,banks[condition]).to(device))
        hook=E.M.HiddenCapture(model)
        with E.core.RecurrentExperimentPatch(model,bank,E.core.hadamard(device)) as patch:
            values=E.evaluate(model,tokenizer,bank,patch,hook)
        hook.close();del bank;E.M.clean()
        comparisons.append(dict(condition=condition,**{k:values[k] for k in E.METRICS}));detail[condition]=values
        if condition in E.ARMS:
            for k in E.METRICS:
                if values[k]!=manifests[condition]['validation'][k]:raise RuntimeError('selected frozen materialization changes Shared16 metric: '+condition+'/'+k)
        print(json.dumps({'event':'fixed_shared16_complete','condition':condition,'metrics':{k:values[k] for k in E.METRICS}}),flush=True)
    table(ROOT/'analysis/shared16_comparison.csv',comparisons,['condition']+list(E.METRICS))
    E.J(ROOT/'analysis/shared16_capture_metrics.json',detail)
    for arm,historical in [('C5R','historical_C5_u063'),('C5R','historical_C5_u113'),('C6R','historical_C6_u125')]:
        if arm not in banks:continue
        for layer in E.core.GDN_LAYERS:
            a=banks[arm][str(layer)];b=banks[historical][str(layer)]
            distances.append({'candidate':arm,'reference':historical,'layer':layer,'delta_frobenius_distance':float((a-b).norm()),
                'reference_norm':float(b.norm()),'descriptive_only':True})
    table(ROOT/'analysis/rotation_distances_descriptive.csv',distances,list(distances[0]))
    E.guard(model,sources,'posttraining','AFTER_FIXED_SEVEN_SHARED16_CONDITIONS')
    text=['# Pre-endpoint clean-room reproduction evidence','',
        'Historical C5/C6 remain valid deployed artifacts and observed endpoints. The primary matched comparison is C7 vs C6R.',
        'Historical candidate metrics and all new scheduled metrics are in cross_objective_learning_curves.csv; all seven conditions are evaluated on the same clean A800 teachers in shared16_comparison.csv.',
        'C5 u113 uses the previously archived noncanonical research export, not a newly rematerialized theta. C5u63/C6u125 use archived canonical deployed byte artifacts.',
        'Rotation distance is descriptive only. No numerical threshold was invented to call a reproduction close or a material shift.',
        'Qualitative reproduction classification: INCONCLUSIVE_PENDING_FINAL_INTERPRETATION. Raw validation evidence is frozen before AIME; this is not an endpoint-based selection gate.',
        'No shift is automatically attributed to old silent corruption.', '']
    for old,arm in [('C5','C5R'),('C6','C6R')]:
        if arm not in manifests:continue
        text.append(f"{old} historical selected update {histories[old]['selected_update']}; {arm} selected update {manifests[arm]['update']}; primary metric {E.PRIMARY[arm]}.")
    path=ROOT/'analysis/historical_reproduction_analysis.md';E.once(path);path.write_text('\n'.join(text)+'\n')
    frozen={arm:{'rotation_path':m['frozen_delta_path'],'file_sha256':m['frozen_delta_sha256'],'condition':arm,
        'per_layer_sha256':m['per_layer_sha256'],'matrix_dtype':'torch.float32','matrix_shape':[128,128],
        'layer_ids':list(E.core.GDN_LAYERS),'selected_update':m['update'],'primary_validation':m['primary_value'],
        'loading_rule':'load frozen rotation directly; do not reconstruct R from theta on inference host',
        'training_host':'A800','historical_endpoint_tuning':False} for arm,m in manifests.items()}
    E.J(ROOT/'deployment/selected_artifacts_pre_endpoint.json',{'status':'PASS','conditions':frozen,'shared16':comparisons,
        'posttraining_script_sha256':E.S(Path(__file__)),'source_history':histories,'no_AIME_used':True})
    E.J(ROOT/'logs/posttraining.completion.json',{'status':'PASS','fixed_panel':list(panel),'selected_R_byte_artifacts_frozen':True,'new_training_updates':0})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        E.J(ROOT/'logs/posttraining.error.json',{'status':'FAIL','error':str(exc),'traceback':traceback.format_exc(),'do_not_retry':True});raise
