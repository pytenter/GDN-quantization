#!/usr/bin/env python3
"""One matched engine, three objectives; immutable canonical helpers are imported."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import time
import traceback

import a800_qualify as Q
import trusted_materializer as TM
M = Q.parent()
F, core, torch, runtime = M.F, M.core, M.torch, M.runtime
ROOT = Q.ROOT
ARMS = ('C5R', 'C6R', 'C7')
METRICS = ('state', 'functional', 'final_hidden')
PRIMARY = {'C5R': 'state', 'C6R': 'functional', 'C7': 'final_hidden'}
LR = {'C5R': .003, 'C6R': .001, 'C7': .001}
J, S, T = F.atomic_json, F.file_sha256, F.atomic_torch
RESERVE = 4 * 1024**3


def once(path):
    if path.exists(): raise RuntimeError('existing evidence; no automatic retry: ' + str(path))


def trusted():
    freeze = json.loads((ROOT / 'provenance/trusted_environment_freeze.json').read_text())
    if freeze['A800_TRAINING_ENVIRONMENT_TRUSTED'] != 'YES': raise RuntimeError('environment not trusted')
    for name, digest in freeze['code_hashes'].items():
        if S(ROOT / name) != digest: raise RuntimeError('frozen implementation changed: ' + name)
    for row in freeze['model_files']:
        if S(Path(row['path'])) != row['sha256']: raise RuntimeError('model file changed')
    versions = {n: Q.importlib.metadata.version(n) for n in freeze['environment']['versions']}
    if versions != freeze['environment']['versions']: raise RuntimeError('packages changed after trust freeze')
    import transformers
    if transformers.__version__ != freeze['environment']['transformers']: raise RuntimeError('Transformers changed')
    return freeze


def protocol():
    p = json.loads((ROOT / 'protocol/preregistration.json').read_text())
    if S(Path(__file__)) != p['engine_sha256']: raise RuntimeError('unified engine drift')
    for path, digest in p['frozen_inputs'].items():
        if S(Path(path)) != digest: raise RuntimeError('historical source/config drift: ' + path)
    return p


def prepare():
    freeze = trusted()
    once(ROOT / 'protocol/preregistration.json')
    old = json.loads((M.OLD / 'preregistration.json').read_text())
    prior = M.protocol()
    common = old['training']['common']
    assert common['horizon'] == 64 and common['tokens_per_document'] == 1024
    assert common['optimizer_updates'] == 125 and common['target_exposures'] == 1000
    assert common['captures_per_document'] == 8 and common['weight_decay'] == 0
    assert common['gradient_clip_global_norm'] == 1 and old['training']['C5']['lr'] == LR['C5R']
    assert old['training']['C6']['lr'] == LR['C6R']
    assert S(core.CORPUS) == common['corpus_sha256']
    rows = core.corpus_rows('TRAIN'); validation = core.corpus_rows('VALIDATION')
    assert len(rows) == 64 and len(validation) == 16
    assert [r['document_id'] for r in validation] == old['validation']['document_ids']
    rng = random.Random(0)
    order = [rows[rng.randrange(len(rows))]['document_id'] for _ in range(125)]
    assert order == common['document_order']
    inputs = {r['path']: r['sha256'] for r in prior['frozen_inputs'].values()}
    inputs[str(M.OLD / 'preregistration.json')] = S(M.OLD / 'preregistration.json')
    inputs[str(core.CORPUS)] = S(core.CORPUS)
    for name in old['frozen_config_hashes']:
        inputs[str(M.OLD / 'configs' / name)] = old['frozen_config_hashes'][name]
    for arm in ARMS:
        cfg = dict(common, arm=arm, lr=LR[arm], optimizer='Adam', selection_metric=PRIMARY[arm],
            objective={'C5R': 'historical captured_losses()[2]', 'C6R': 'historical captured_losses()[3] = Functional + 0.1 State',
                       'C7': 'CAYLEY.relative_mse(final_normalized_lm_head_input, clean_FP_hidden) ONLY'}[arm],
            epsilon='sum(error^2)/clamp_min(sum(reference^2),1e-12)', initialization=old['rotation']['initialization'],
            capture_weight='1/8', validation_weight='equal 16 documents and 8 captures each')
        J(ROOT / f'protocol/{arm.lower()}_config.json', cfg)
    p = {'task': ROOT.name, 'training_order': list(ARMS), 'engine_sha256': S(Path(__file__)),
        'common': common, 'validation': old['validation'], 'rotation': old['rotation'], 'frozen_inputs': inputs,
        'trusted_environment_sha256': S(ROOT / 'provenance/trusted_environment_freeze.json'),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parents[1], text=True).strip(),
        'only_C6R_C7_intended_difference': 'objective', 'teacher_parity_panel': 'first 3 TRAIN documents x 8 captures, all 24 layers',
        'update0_panel': 'first TRAIN document, tokens 0..63, all 24 GDN layers, fresh theta0 per arm',
        'readiness': 'fresh gradient 64 tokens then separate theta0 three exact 1024-token/8-capture/H64 updates per arm',
        'memory_creep_threshold_bytes': F.MEMORY_CREEP_BYTES,
        'immutable_weight_checks': ['fresh CPU direct safetensors and complete GPU', 'before update1', 'after update1', 'after update63', 'after update125'],
        'primary_comparison': ['C7', 'C6R'], 'paired_family': [['C7','C6R'],['C7','H'],['C6R','H'],['C5R','H'],['C6R','historical_C6'],['C5R','historical_C5'],['C7','historical_C6'],['C7','FP']],
        'selection_eligible': 'same historical 20 scheduled candidates, update0 reference only, minimum selected metric, exact ties earlier',
        'GSM8K': 'NOT_STARTED_ARCHIVED_SPLIT_UNTOUCHED', 'FULL_DENSITY_FULL60': 'CANCELLED_BY_USER',
        'retry_scientific_failure': False, 'historical_endpoints': 'valid archived observations, not relabeled invalid',
        'reproduction_interpretation': 'predefined common validation and learning curves, descriptive rotation distance; no new numerical thresholds'}
    J(ROOT / 'protocol/preregistration.json', p)
    J(ROOT / 'protocol/data_provenance.json', {'corpus_sha256': S(core.CORPUS), 'TRAIN_ids': [r['document_id'] for r in rows],
        'VALIDATION_ids': [r['document_id'] for r in validation], 'AIME_overlap': 'NO inherited verified corpus audit',
        'historical_teacher_caches': 'NOT_USED', 'tokenizer_identity': freeze['model_files']})


def load(stage):
    trusted(); protocol(); M.seed()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Q.MODEL), trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(str(Q.MODEL), torch_dtype=torch.bfloat16,
        device_map=None, trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('unexpected canonical load inventory')
    Q.verify_cpu_checkpoint(model, f'qualification/{stage}_cpu_checkpoint.json')
    sources = {n: x['tensor'].detach() for n, x in TM.state_items(model).items()}
    model.to('cuda:0'); torch.cuda.synchronize(); core.freeze_model(model)
    guard(model, sources, stage, 'FRESH_FULL_GPU_INTEGRITY')
    return model, tokenizer, sources, model.get_input_embeddings().weight.device


def guard(model, sources, stage, point):
    path = ROOT / f'qualification/{stage}_immutable_guards.jsonl'
    def emit(name, kind, where, result):
        if not result['exact_equal'] or not result['finite']:
            J(ROOT / f'qualification/{stage}_mutation.json', {'name': name, 'point': point, 'result': result})
    hashes = TM.verify_model(model, sources, emit, point)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise RuntimeError('frozen model acquired gradient')
    F.append_jsonl(path, {'status':'PASS','point':point,'pid':os.getpid(),'tensor_count':len(hashes),'hashes':hashes})


def token_hash(ids):
    return hashlib.sha256(b''.join(int(t).to_bytes(8, 'little') for t in ids)).hexdigest()


def unified_teacher(model, patch, ids, positions):
    hidden = []
    def head(module, args):
        if patch.mode == 'teacher' and patch.capture:
            hidden.append(args[0].detach().cpu().contiguous().clone())
    handle = model.lm_head.register_forward_pre_hook(head)
    try: values = core.collect_teacher_targets(model, patch, ids, positions)
    finally: handle.remove()
    if len(hidden) != len(positions): raise RuntimeError('unified hidden capture coverage')
    for pos, h in zip(positions, hidden):
        if h.shape != (1, 1, 4096) or h.dtype != torch.bfloat16: raise RuntimeError('Final Hidden shape/dtype')
        values[pos]['hidden'] = h
    return values


def tensors(values):
    for pos, families in values.items():
        for family in ('prev', 'out'):
            for layer, value in sorted(families[family].items()): yield pos, family, layer, value
        yield pos, 'hidden', -1, families['hidden']


def teacher():
    once(ROOT / 'teacher/teacher_cache_parity.json')
    model, tokenizer, sources, device = load('teacher')
    bank = F.fresh_bank(device); rows = core.corpus_rows('TRAIN') + core.corpus_rows('VALIDATION')
    destinations = [ROOT / 'teacher/targets_data', Path('/root/qwen_a800_teacher') / ROOT.name]
    for d in destinations: d.mkdir(parents=True, exist_ok=True)
    files, manifest, parity = [], [], []
    remaining = [shutil.disk_usage(d).free - RESERVE for d in destinations]
    with core.RecurrentExperimentPatch(model, bank, core.hadamard(device)) as patch:
        for i, row in enumerate(rows):
            ids, positions = core.tokenize(tokenizer, row)
            if len(ids) != 1024 or len(positions) != 8: raise RuntimeError('teacher document/capture size')
            values = unified_teacher(model, patch, ids, positions)
            tensor_bytes = sum(v.numel()*v.element_size() for _,_,_,v in tensors(values))
            if i == 0:
                estimated = (tensor_bytes + (1 << 20)) * len(rows)
                if estimated > sum(remaining): raise RuntimeError('insufficient combined disks for exact clean teacher cache')
                J(ROOT / 'teacher/storage_plan.json', {'destinations':[str(d) for d in destinations], 'reserve_per_volume_bytes':RESERVE,
                    'available_after_reserve':remaining, 'estimated_total_bytes':estimated, 'uncompressed_exact_targets':True,
                    'volumes': [os.stat(d).st_dev for d in destinations], 'source_science_deleted':False})
                if os.stat(destinations[0]).st_dev == os.stat(destinations[1]).st_dev:
                    raise RuntimeError('planned independent teacher volumes are same filesystem')
            choice = max(range(2), key=lambda j: remaining[j]); path = destinations[choice] / (row['document_id']+'.pt')
            if tensor_bytes + (1<<20) > remaining[choice]: raise RuntimeError('teacher disk reserve')
            once(path)
            payload = {'document_id':row['document_id'],'token_ids_sha256':token_hash(ids),'positions':positions,'targets':values,
                'old_vGPU_cache_used':False,'metadata':{f'{pos}/{family}/{layer}':runtime.tensor_sha256(v) for pos,family,layer,v in tensors(values)}}
            T(path, payload); remaining[choice] -= path.stat().st_size
            files.append({'document_id':row['document_id'],'path':str(path),'sha256':S(path),'bytes':path.stat().st_size,'positions':positions})
            for pos,family,layer,v in tensors(values):
                if not bool(torch.isfinite(v).all()): raise FloatingPointError('nonfinite teacher target')
                manifest.append({'document_id':row['document_id'],'capture_index':positions.index(pos),'token_position':pos,'family':family,
                    'layer':layer,'shape':json.dumps(list(v.shape)),'dtype':str(v.dtype),'sha256':payload['metadata'][f'{pos}/{family}/{layer}'],
                    'finite':True,'cache_file':str(path)})
            if i < 3:
                loaded = torch.load(path,map_location='cpu',weights_only=False)['targets']
                # Independently call unchanged historical collectors, including separate hidden collector.
                reference = core.collect_teacher_targets(model,patch,ids,positions)
                hook = M.HiddenCapture(model)
                try: h = M.hidden_targets(model,patch,hook,ids,positions)
                finally: hook.close()
                for pos in positions:
                    for family in ('prev','out'):
                        equal = all(torch.equal(loaded[pos][family][l], reference[pos][family][l]) for l in core.GDN_LAYERS)
                        parity.append({'document_id':row['document_id'],'position':pos,'family':family,'layers':24,'exact_equal':equal})
                        if not equal: raise RuntimeError('online/cached State or Functional parity failed')
                    equal = torch.equal(loaded[pos]['hidden'],h[pos])
                    parity.append({'document_id':row['document_id'],'position':pos,'family':'hidden','exact_equal':equal})
                    if not equal: raise RuntimeError('online/cached Final Hidden parity failed')
                del loaded,reference,h
            del values,payload; M.clean(patch)
            print(json.dumps({'event':'clean_teacher_document','document_id':row['document_id'],'completed':i+1,'total':80}),flush=True)
    if len(files) != 80 or len(manifest) != 640*49: raise RuntimeError('teacher manifest coverage')
    with (ROOT / 'teacher/teacher_target_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader();w.writerows(manifest)
    J(ROOT / 'teacher/cache_files.json', {'status':'PASS','files':files,'documents':80,'captures':640})
    guard(model,sources,'teacher','AFTER_ALL_CLEAN_TEACHERS')
    J(ROOT / 'teacher/teacher_cache_parity.json', {'status':'PASS','CLEAN_TEACHER_TARGET_PARITY':'PASS','parity':parity,
        'captures_per_family':24,'all_state_functional_layers':24,'unified_pass':True,'direct_file_readback':True})


def cached(row, ids, positions):
    registry=json.loads((ROOT / 'teacher/cache_files.json').read_text())
    item=next(r for r in registry['files'] if r['document_id']==row['document_id'])
    path=Path(item['path'])
    if S(path)!=item['sha256']: raise RuntimeError('teacher file hash drift')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['token_ids_sha256']!=token_hash(ids) or value['positions']!=positions: raise RuntimeError('cache trace drift')
    for pos,family,layer,v in tensors(value['targets']):
        if runtime.tensor_sha256(v)!=value['metadata'][f'{pos}/{family}/{layer}']:raise RuntimeError('teacher tensor hash drift')
    return value['targets']


def fresh_bank(device):
    bank=F.fresh_bank(device)
    F.verify_initialization(bank,json.loads((M.OLD / 'preregistration.json').read_text()))
    return bank


def train_document(model,tokenizer,bank,patch,hook,opt,device,row,update,arm,smoke=False):
    ids,positions=core.tokenize(tokenizer,row)
    if smoke:
        ids=ids[:64];positions=[63];targets=unified_teacher(model,patch,ids,positions)
    else:
        if len(ids)!=1024 or len(positions)!=8:raise RuntimeError('formal trace size mismatch')
        targets=cached(row,ids,positions)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats(device);opt.zero_grad(set_to_none=True)
    cache=None;losses=[];captures=[];boundaries=[];qbefore=patch.qdq_audit['calls']
    for i,token in enumerate(ids):
        capture=i in positions
        if capture:core.install_teacher_target(patch,targets[i],device)
        hook.active=capture
        cache=core.forward_student(model,patch,token,cache,capture)
        if capture:
            state,functional,c5,c6=patch.captured_losses()
            hidden=core.CAYLEY.relative_mse(hook.value,targets[i]['hidden'].to(device))
            selected={'C5R':c5,'C6R':c6,'C7':hidden}[arm]
            vals={'state':float(state.detach()),'functional':float(functional.detach()),'final_hidden':float(hidden.detach())}
            if any(not math.isfinite(v) for v in vals.values()):raise FloatingPointError('nonfinite captured loss')
            losses.append(selected/len(positions));captures.append(dict(position=i,optimized_loss=float(selected.detach()),**vals))
            del state,functional,c5,c6,hidden,selected;hook.clear()
        if (i+1)%64==0 or i+1==len(ids):
            if losses:
                total=sum(losses);total.backward();losses.clear();del total
            boundaries.append({'tokens':i+1,'after_backward_allocated':torch.cuda.memory_allocated(device),'reserved':torch.cuda.memory_reserved(device)})
            patch.clear_capture();hook.clear();core.detach_cache(cache)
        if (i+1)%128==0:print(json.dumps({'event':'training_token','arm':arm,'update':update,'tokens':i+1}),flush=True)
    if patch.qdq_audit['calls']-qbefore!=24*len(ids):raise RuntimeError('QDQ coverage mismatch')
    params=list(bank.parameters())
    if len(params)!=24 or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('invalid theta gradient')
    norms={str(l):float(bank.layer(l).theta.grad.norm()) for l in core.GDN_LAYERS}
    if sum(v>0 for v in norms.values())<2:raise RuntimeError('insufficient nonzero layer gradient')
    before=[p.detach().clone() for p in params]
    gradient=float(torch.nn.utils.clip_grad_norm_(params,1.0))
    if not math.isfinite(gradient) or gradient>F.EXPLODING_GRAD_NORM:raise FloatingPointError('exploding gradient')
    opt.step();after_step=torch.cuda.memory_allocated(device)
    movements=[float((p.detach()-b).norm()) for p,b in zip(params,before)]
    if not any(v>0 for v in movements):raise RuntimeError('theta did not move')
    del cache,targets,before;M.clean(patch)
    orth=bank.runtime_gate(threshold=core.ORTHOGONALITY_THRESHOLD)
    if orth['status']!='PASS':raise RuntimeError('rotation orthogonality failed')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('base model not frozen')
    return {'arm':arm,'update':update,'exposure':update*len(positions),'document_id':row['document_id'],
        'captures':captures,'training_primary_loss':sum(c[PRIMARY[arm]] for c in captures)/len(positions),
        'training_optimized_loss':sum(c['optimized_loss'] for c in captures)/len(positions),
        'gradient_norm':gradient,'per_layer_gradient_norm':norms,'theta_movement':movements,'orthogonality':orth,
        'boundaries':boundaries,'after_optimizer_allocated':after_step,'memory':F.memory_snapshot(device,'post_update_cleanup'),
        'runtime_seconds':time.monotonic()-start,'QDQ_calls':24*len(ids),'H':64,'numerical_state_reset_at_boundaries':False}


def evaluate(model,tokenizer,bank,patch,hook):
    docs=[];device=model.get_input_embeddings().weight.device
    with torch.no_grad():
        for row in core.corpus_rows('VALIDATION'):
            ids,positions=core.tokenize(tokenizer,row);targets=cached(row,ids,positions);cache=None;captures=[]
            for i,t in enumerate(ids):
                cap=i in positions
                if cap:core.install_teacher_target(patch,targets[i],device)
                hook.active=cap;cache=core.forward_student(model,patch,t,cache,cap)
                if cap:
                    s,f,_,_=patch.captured_losses();h=core.CAYLEY.relative_mse(hook.value,targets[i]['hidden'].to(device))
                    captures.append({'position':i,'state':float(s),'functional':float(f),'final_hidden':float(h)})
                    del s,f,h;hook.clear();patch.clear_capture()
                if (i+1)%64==0:core.detach_cache(cache)
            docs.append({'document_id':row['document_id'],'captures':captures})
            del cache,targets;M.clean(patch)
            print(json.dumps({'event':'validation_document','document_id':row['document_id']}),flush=True)
    values={k:sum(sum(c[k] for c in d['captures'])/8 for d in docs)/16 for k in METRICS}
    if len(docs)!=16 or any(not math.isfinite(v) for v in values.values()):raise FloatingPointError('invalid Shared16 validation')
    return dict(values,documents=16,captures=128,per_document=docs)


def update0():
    once(ROOT / 'readiness/update0_forward_parity.json')
    model,tokenizer,sources,device=load('update0');ids,_=core.tokenize(tokenizer,core.corpus_rows('TRAIN')[0]);ids=ids[:64]
    results=[];index_audit=[]
    norm=model.model.norm if hasattr(model.model,'norm') else model.model.language_model.norm
    norm_path=next(n for n,m in model.named_modules() if m is norm)
    for arm in ARMS:
        bank=fresh_bank(device);records=[];cache=None;norm_value={};out_proj=[]
        nh=norm.register_forward_hook(lambda m,a,o:norm_value.update(value=o))
        hook=M.HiddenCapture(model);head_rows=[];logits_rows=[]
        hh=model.lm_head.register_forward_pre_hook(lambda m,a:head_rows.append(runtime.tensor_sha256(a[0])))
        lh=model.lm_head.register_forward_hook(lambda m,a,o:logits_rows.append(runtime.tensor_sha256(o)))
        original=core.CAYLEY.qwen_c128_ste
        layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
        oh=[layers[l].linear_attn.out_proj.register_forward_hook(
            lambda m,a,o,l=l:out_proj.append({'layer':l,'hash':runtime.tensor_sha256(o)})) for l in core.GDN_LAYERS]
        with core.RecurrentExperimentPatch(model,bank,core.hadamard(device)) as patch:
            def observe(x):
                q=original(x);records.append({'layer':patch.current_layer,'input':runtime.tensor_sha256(x),'qcodes':runtime.tensor_sha256(q.codes),'writeback':runtime.tensor_sha256(q.dequant)})
                return q
            core.CAYLEY.qwen_c128_ste=observe
            try:
                with torch.no_grad():
                    for i,t in enumerate(ids):
                        hook.active=True;cache=core.forward_student(model,patch,t,cache,False)
                        if not torch.equal(hook.value,norm_value['value']):raise RuntimeError('final norm/lm_head indexing differs')
                        index_audit.append({'arm':arm,'input_token_index':i,'hidden_position':i,'logit_position':i,'next_token_target_index':i+1,'norm_module':norm_path,'shape':list(hook.value.shape),'dtype':str(hook.value.dtype),'exact_lm_head_input':True})
                        norm_value.clear();hook.clear()
                    capture={'arm':arm,'qdq':records,'hidden':head_rows,'logits':logits_rows,'out_proj':out_proj,'cache':M.cache_hashes(cache)}
            finally:
                core.CAYLEY.qwen_c128_ste=original;nh.remove();hh.remove();lh.remove();hook.close()
                for handle in oh:handle.remove()
        if len(records)!=24*64 or len(out_proj)!=24*64:raise RuntimeError('update0 layer/token coverage')
        results.append(capture);del cache,bank;M.clean()
    reference={k:v for k,v in results[0].items() if k!='arm'}
    if any({k:v for k,v in r.items() if k!='arm'}!=reference for r in results):raise RuntimeError('three-arm update0 exact forward parity failed')
    J(ROOT / 'readiness/final_hidden_indexing_audit.json',{'status':'PASS','FINAL_HIDDEN_INDEXING_AUDIT':'PASS','records':index_audit,'final_hidden':'final normalized lm_head input','loss':'representation only, no shifted LM label loss'})
    J(ROOT / 'readiness/update0_forward_parity.json',{'status':'PASS','C5R_C6R_C7_UPDATE0_FORWARD_PARITY':'PASS','records':results,'tokens':64,'all_24_layers':True})
    # All three full Shared16 baselines are prospective: before ANY formal update.
    validations=[]
    for arm in ARMS:
        bank=fresh_bank(device);hook=M.HiddenCapture(model)
        with core.RecurrentExperimentPatch(model,bank,core.hadamard(device)) as patch:
            value=evaluate(model,tokenizer,bank,patch,hook)
        J(ROOT / f'analysis/{arm.lower()}_update0_validation.json',{'status':'PASS','validation':value,'reference_only':True,
            'prospective_before_all_arms_training':True,'theta0':True})
        validations.append(value);hook.close();del bank;M.clean()
    if any(v!=validations[0] for v in validations):raise RuntimeError('prospective full Shared16 update0 parity')
    J(ROOT / 'readiness/update0_validation_parity.json',{'status':'PASS','all_metrics_all_128_captures_exact':True,'arms':list(ARMS)})
    guard(model,sources,'update0','AFTER_THREE_FORWARD_PANELS')


def readiness(arm):
    once(ROOT / f'readiness/{arm.lower()}_h64_readiness.json')
    require('teacher/teacher_cache_parity.json');require('readiness/update0_forward_parity.json')
    require('readiness/update0_validation_parity.json')
    model,tokenizer,sources,device=load('readiness_'+arm)
    bank=fresh_bank(device);hook=M.HiddenCapture(model);opt=torch.optim.Adam(bank.parameters(),lr=LR[arm],weight_decay=0)
    with core.RecurrentExperimentPatch(model,bank,core.hadamard(device)) as patch:
        record=train_document(model,tokenizer,bank,patch,hook,opt,device,core.corpus_rows('TRAIN')[0],1,arm,True)
    guard(model,sources,'readiness_'+arm,'AFTER_GRADIENT_SMOKE')
    J(ROOT / f'readiness/{arm.lower()}_gradient_smoke.json',{'status':'PASS',arm+'_GRADIENT_SMOKE':'PASS','record':record,'only_theta_updates':True})
    hook.close();del bank,opt;M.clean()
    bank=fresh_bank(device);hook=M.HiddenCapture(model);opt=torch.optim.Adam(bank.parameters(),lr=LR[arm],weight_decay=0)
    rows={r['document_id']:r for r in core.corpus_rows('TRAIN')};order=protocol()['common']['document_order'][:3];docs=[]
    with core.RecurrentExperimentPatch(model,bank,core.hadamard(device)) as patch:
        for i,name in enumerate(order,1):docs.append(train_document(model,tokenizer,bank,patch,hook,opt,device,rows[name],i,arm))
    span=max(d['memory']['allocated_bytes'] for d in docs)-min(d['memory']['allocated_bytes'] for d in docs)
    if span>F.MEMORY_CREEP_BYTES:raise RuntimeError('exact H64 memory creep exceeds inherited threshold')
    guard(model,sources,'readiness_'+arm,'AFTER_THREE_EXACT_H64_UPDATES')
    J(ROOT / f'readiness/{arm.lower()}_h64_readiness.json',{'status':'PASS',arm+'_H64_READY':'PASS','documents':docs,'H':64,'sequence_length':1024,'captures_per_document':8,'post_cleanup_memory_span':span,'activation_checkpointing':False,'fresh_formal_start_required':True})
    hook.close()


def require(relative):
    if json.loads((ROOT / relative).read_text())['status']!='PASS':raise RuntimeError('required gate not PASS: '+relative)


def candidate(bank,arm,update,validation):
    base=ROOT / 'training' / arm;prefix=f'update_{update:03d}_exposure_{update*8:04d}'
    theta=base / 'checkpoints' / (prefix+'.pt');delta=base / 'frozen_rotations' / (prefix+'.pt')
    once(theta);once(delta)
    T(theta,{'condition':arm,'bank':F.cpu_state_dict(bank),'update':update,'exposure':update*8,'validation':validation,'preregistration_sha256':S(ROOT/'protocol/preregistration.json')})
    matrices={str(l):bank.layer(l).matrix().detach().float().cpu().contiguous().clone() for l in core.GDN_LAYERS}
    T(delta,{'format_version':1,'experiment':ROOT.name,'condition':arm,'layer_ids':tuple(core.GDN_LAYERS),'matrices':matrices,
        'matrix_dtype':'torch.float32','matrix_shape':[128,128],'matrix_meaning':'training-A800 materialized Delta, H separately',
        'orientation':'q/k.float() @ H128 @ Delta; recovery Delta then H128 on key axis','theta_checkpoint_sha256':S(theta),
        'export_host':socket.gethostname(),'update':update,'exposure':update*8})
    record={'arm':arm,'update':update,'exposure':update*8,'theta_path':str(theta),'theta_sha256':S(theta),'frozen_delta_path':str(delta),
        'frozen_delta_sha256':S(delta),'per_layer_sha256':{l:runtime.tensor_sha256(v) for l,v in matrices.items()},'validation':validation,
        'primary_selection_metric':PRIMARY[arm],'primary_value':validation[PRIMARY[arm]],
        'orthogonality':max(float((v.T@v-torch.eye(128)).abs().max()) for v in matrices.values())}
    if record['orthogonality']>core.ORTHOGONALITY_THRESHOLD:raise RuntimeError('candidate export orthogonality failed')
    J(base / 'frozen_rotations' / (prefix+'.manifest.json'),record)
    F.append_jsonl(base / 'candidate_manifest.jsonl',record)
    return record


def train(arm):
    p=protocol();base=ROOT / 'training' / arm;once(base / 'train_log.jsonl')
    require('teacher/teacher_cache_parity.json');require('readiness/update0_forward_parity.json')
    require('readiness/update0_validation_parity.json')
    for a in ARMS:
        require(f'readiness/{a.lower()}_gradient_smoke.json');require(f'readiness/{a.lower()}_h64_readiness.json')
    for a in ARMS[:ARMS.index(arm)]:require(f'training/{a}/summary.json')
    model,tokenizer,sources,device=load('training_'+arm)
    bank=fresh_bank(device);hook=M.HiddenCapture(model);opt=torch.optim.Adam(bank.parameters(),lr=LR[arm],weight_decay=0)
    rows={r['document_id']:r for r in core.corpus_rows('TRAIN')};candidates=[];allocated=[];start=time.monotonic()
    with core.RecurrentExperimentPatch(model,bank,core.hadamard(device)) as patch:
        zero=json.loads((ROOT / f'analysis/{arm.lower()}_update0_validation.json').read_text())['validation']
        initial=candidate(bank,arm,0,zero)
        guard(model,sources,'training_'+arm,'BEFORE_UPDATE1')
        for update,name in enumerate(p['common']['document_order'],1):
            record=train_document(model,tokenizer,bank,patch,hook,opt,device,rows[name],update,arm)
            allocated.append(record['memory']['allocated_bytes'])
            if max(allocated)-min(allocated)>F.MEMORY_CREEP_BYTES:raise RuntimeError('formal memory creep')
            if update in (1,63,125):guard(model,sources,'training_'+arm,'AFTER_UPDATE_'+str(update))
            if update in p['validation']['actual_validation_updates']:
                val=evaluate(model,tokenizer,bank,patch,hook);candidates.append(candidate(bank,arm,update,val));record['validation']=val
            F.append_jsonl(base / 'train_log.jsonl',record)
            print(json.dumps({'event':'formal_update_complete','arm':arm,'update':update,'exposure':update*8,'primary_training_loss':record['training_primary_loss']}),flush=True)
    if len(candidates)!=20:raise RuntimeError('candidate coverage differs from frozen cadence')
    selected=min(candidates,key=lambda r:(r['primary_value'],r['update']))
    J(base / 'checkpoint_selection.json',{'status':'PASS','selected':selected,'candidate_count':20,'rule':p['selection_eligible'],'AIME_used':False})
    J(base / 'summary.json',{'status':'PASS','updates':125,'exposures':1000,'H':64,'selected_update':selected['update'],
        'FROZEN_MODEL_INTEGRITY':'PASS','retry_count':0,'runtime_seconds':time.monotonic()-start})
    hook.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','teacher','update0','readiness','train'],required=True)
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args();key=args.phase+('_'+args.arm if args.arm else '')
    for d in ('protocol','teacher','readiness','training','analysis','logs'): (ROOT / d).mkdir(parents=True,exist_ok=True)
    once(ROOT / f'logs/{key}.completion.json');once(ROOT / f'logs/{key}.error.json')
    start=time.monotonic()
    try:
        if args.phase in ('readiness','train'):globals()[args.phase](args.arm)
        else:globals()[args.phase]()
        J(ROOT / f'logs/{key}.completion.json',{'status':'PASS','runtime_seconds':time.monotonic()-start})
    except BaseException as exc:
        error={'status':'FAIL','phase':args.phase,'arm':args.arm,'error':str(exc),'traceback':traceback.format_exc(),'do_not_retry':True,'protocol_changed':False}
        if torch.cuda.is_available():error.update(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())
        J(ROOT / f'logs/{key}.error.json',error);raise


if __name__=='__main__':main()
