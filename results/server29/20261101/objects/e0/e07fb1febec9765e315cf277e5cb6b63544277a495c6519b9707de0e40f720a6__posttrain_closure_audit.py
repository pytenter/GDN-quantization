#!/usr/bin/env python3
"""Append-only CPU provenance audit; never imports or executes training/model code."""
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
TASK = 'QWEN_GDN_A800_POSTTRAIN_SHARED16_REPRODUCTION_CLOSURE_V1'
HEAD = 'ad1b9fc19bd5fd5cc9014b7473f31edce22976a3'
FREEZE = 'd44267ee6bc18acce6985ef18f9e3ce434c1a4cf6cae870bd9300e2cc4ae6816'
PREREG = '009279fb84dd6d45f73bdfc27629a6d28de8d1a6dbfab64b65c83e9c4de5f3c8'
ENGINE = '0f85e40100070c051a01d05125341c1863b5c9808247803a725b1b218f54412d'
OLD_RUNNER = 'fe6787f9a3a436539a8da8dfc13f7fb00ba2e38cb2af25060b9d9962dc43380b'
KNOWN_DIRTY = {
 'EXECUTION_HANDOFF.md':'072e72743b4902b0aa96266d7939ee1da4f9c80047633f9c623cc3ce63e2277d',
 'scripts/a800_inventory.py':'51e6ef673af7229473b5250e8101bb8cd83377e2b06a8e53d1e8e13b893b67f4',
 'scripts/amended_a800_controller.py':'58c59d3612dea44c08cd80be0d846a5d542fe676e04fd44931725918fdef62ef',
 'scripts/amended_a800_inventory.py':'bf5e49ff6228948000e6dd7e13ba9fccc930a964e10785df6c0cd1403b8100ff',
 'scripts/amended_a800_posttraining.py':OLD_RUNNER,
 'scripts/amended_c5r_reproduction_gate.py':'e533763a7a9a5583e1630f6a6d2a838812a6d6cf1ad91c4245c55eddc8fb69da',
 'scripts/amended_control_audit.py':'573a5d03172517a2c666757b417e37f8519ab0fa83269883dd085f40015822b5',
 'scripts/amended_training_entry.py':'b2d7484c8662e452e80ae055edb7488c9b7edf3dd0be8c32643bc207e16be61d',
 'scripts/amendment_record.py':'f9c167ec87decdafd6ad71b231830191efb00dd48d276dbf5a595c5c6435bb84'}
NEW_SCRIPTS = ('posttrain_closure_audit.py','posttrain_closure_runner.py','posttrain_closure_finalize.py','posttrain_closure_pipeline.sh')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(rel):return json.loads((ROOT/rel).read_text())
def save(rel,value):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def require(cond,message):
    if not cond:raise RuntimeError(message)
def command(args):return subprocess.check_output(args,cwd=REPO,text=True).strip()
def identity(path,digest=None):
    p=Path(path);s=p.stat();h=sha(p)
    require(digest is None or h==digest,'frozen identity mismatch: '+str(p))
    require(p.stat().st_mtime_ns==s.st_mtime_ns and p.stat().st_size==s.st_size,'changed during hash: '+str(p))
    return {'path':str(p),'size':s.st_size,'sha256':h,'mtime_ns':s.st_mtime_ns}

def preaudit():
    require(command(['git','rev-parse','HEAD'])==HEAD,'pre-task HEAD changed')
    dirty=command(['git','status','--porcelain','--untracked-files=all']).splitlines()
    explained=[];prefix=str(ROOT.relative_to(REPO))+'/'
    for line in dirty:
        require(line.startswith('?? '+prefix),'unexplained tracked dirty state: '+line)
        rel=line[3+len(prefix):]
        if rel in KNOWN_DIRTY:
            identity(ROOT/rel,KNOWN_DIRTY[rel]);reason='prior authorized controller/evidence; exact local source hash matched'
        else:
            require(rel in ['scripts/'+s for s in NEW_SCRIPTS],'unexplained untracked file: '+rel)
            reason='new closure-only source uploaded for this task'
        explained.append({'git_line':line,'reason':reason,'sha256':sha(ROOT/rel)})
    require(sha(ROOT/'provenance/trusted_environment_freeze.json')==FREEZE,'trust freeze drift')
    require(sha(ROOT/'protocol/preregistration.json')==PREREG,'registration drift')
    freeze=read('provenance/trusted_environment_freeze.json');p=read('protocol/preregistration.json')
    host=read('provenance/a800_host.json')
    gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used','--format=csv,noheader']).split(', ')
    require(len(gpu)==4 and gpu[0]==host['gpu_model'] and gpu[1]==host['gpu_uuid'] and gpu[2]==host['driver'],'GPU/driver qualification identity changed')
    require(gpu[3]=='0 MiB','GPU not idle')
    require(sys.version_info[:3]==(3,10,18),'Python identity changed')
    versions={k:importlib.metadata.version(k) for k in freeze['environment']['versions']}
    require(versions==freeze['environment']['versions'],'package identity changed')
    require(importlib.metadata.version('transformers')==freeze['environment']['transformers'],'Transformers changed')
    for rel,d in freeze['code_hashes'].items():identity(ROOT/rel,d)
    for path,d in p['frozen_inputs'].items():identity(path,d)
    require(sha(ROOT/'scripts/a800_matched.py')==ENGINE,'engine changed')
    amendment=read('protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json')
    prior_append=[]
    for rel,d in amendment['artifact_hashes'].items():
        if rel=='qualification/update0_immutable_guards.jsonl':
            # The amendment was recorded during update0. Its last PASS guard was legitimately appended at phase end.
            raw=(ROOT/rel).read_bytes();parts=raw.splitlines(keepends=True)
            require(any(hashlib.sha256(b''.join(parts[:i])).hexdigest()==d for i in range(1,len(parts)+1)),'old in-flight guard prefix changed')
            rows=[json.loads(x) for x in parts]
            require(all(x['status']=='PASS' and x['hashes']==rows[0]['hashes'] for x in rows),'append guard integrity failed')
            require(rows[-1]['point']=='AFTER_THREE_FORWARD_PANELS','update0 final guard missing')
            prior_append.append({'path':rel,'amendment_inflight_prefix_sha256':d,'final_sha256':sha(ROOT/rel),'explanation':'authorized update0 naturally completed after amendment; exact old prefix retained, final guard same full model hashes'})
        else:identity(ROOT/rel,d)
    cache=read('teacher/cache_files.json');require(cache['status']=='PASS' and len(cache['files'])==80,'teacher coverage')
    selected={}
    for arm in ('C5R','C6R','C7'):
        s=read(f'training/{arm}/summary.json');selection=read(f'training/{arm}/checkpoint_selection.json')
        require(s['status']=='PASS' and s['updates']==125 and s['exposures']==1000 and s['retry_count']==0,'training summary failed')
        require(selection['status']=='PASS' and selection['candidate_count']==20,'selection frozen gate')
        z=selection['selected'];require(z['update']==s['selected_update'] and z['exposure']==z['update']*8,'selected identity')
        identity(z['theta_path'],z['theta_sha256']);identity(z['frozen_delta_path'],z['frozen_delta_sha256'])
        candidates=[json.loads(l) for l in (ROOT/f'training/{arm}/candidate_manifest.jsonl').read_text().splitlines()]
        require(len(candidates)==21 and z in candidates and [x['update'] for x in candidates[1:]]==p['validation']['actual_validation_updates'],'candidate schedule identity')
        for c in candidates:
            identity(c['theta_path'],c['theta_sha256']);identity(c['frozen_delta_path'],c['frozen_delta_sha256'])
        selected[arm]=dict(z,selection_rule=selection['rule'],reselection=False)
    # Snapshot ALL existing evidence before any runner identity amendment. Includes failures and all teachers.
    paths={x for x in ROOT.rglob('*') if x.is_file() and '__pycache__' not in x.parts and '.git' not in x.parts}
    paths.update(Path(x['path']) for x in cache['files']);paths.update(Path(x['path']) for x in freeze['model_files'])
    paths.update(Path(x) for x in p['frozen_inputs'])
    rows=[];cache_expected={x['path']:x['sha256'] for x in cache['files']};models={x['path']:x['sha256'] for x in freeze['model_files']}
    for i,path in enumerate(sorted(paths)):
        # Operational progress log is the only currently open file.
        if path.name=='posttrain_closure_pipeline.log':continue
        row=identity(path,cache_expected.get(str(path),models.get(str(path))));rows.append(row)
        if i%20==0:print(json.dumps({'event':'preinventory','files':i+1,'path':str(path)}),flush=True)
    save('provenance/preclosure_training_artifact_inventory.json',{'status':'PASS','files':rows,'count':len(rows),'bytes':sum(x['size'] for x in rows),'coverage':'all preexisting experiment evidence, external 80 teachers, frozen model files and registered inputs','pre_task_head':HEAD,'pre_task_branch':command(['git','branch','--show-current']),'dirty_explained':explained,'hostname':platform.node(),'gpu':gpu,'python':sys.version,'versions':versions,'torch_cuda_runtime':'12.1 pinned wheel; actual checked before evaluation','teacher_cache_rebuild':False,'mtime_ns_recorded':True})
    save('analysis/posttrain_selected_checkpoint_identity.json',{'status':'PASS','conditions':selected,'selection_changed':False})
    # Digest resolution is derived from independently archived identity records, not current bytes.
    refs=read('provenance/posttrain_reference_evidence/formal_config.json')
    require(sha(ROOT/'provenance/posttrain_reference_evidence/formal_config.json')=='557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836','independent canonical config drift')
    export=read('provenance/posttrain_reference_evidence/noncanonical_candidate_export.json')
    require(sha(ROOT/'provenance/posttrain_reference_evidence/noncanonical_candidate_export.json')=='9713c31c26df43bc8d516f01f6ecb5a0c8e2e4ad26f16b0189a6884705bd02bf','noncanonical independent export identity changed')
    expected={'C5_u063':refs['conditions']['C5']['rotation_sha256'],'C6_u125':refs['conditions']['C6']['rotation_sha256'],'C5_u113':next(x['frozen_rotation_sha256'] for x in export['candidates'] if x['candidate']=='C5_U113')}
    observation=read('provenance/posttrain_reference_evidence/independent_copy_observation.json')
    old=ROOT/'scripts/amended_a800_posttraining.py';identity(old,OLD_RUNNER)
    tree=ast.parse(old.read_text());bad=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED' for t in x.targets))
    records=[]
    for name,d in expected.items():
        path=ROOT/f'provenance/historical_rotations/{name}.pt';actual=sha(path);second=command(['sha256sum',str(path)]).split()[0];obs=observation['copies'][name]
        require(re.fullmatch('[0-9a-f]{64}',d) is not None and actual==second==d==obs['hashlib']==obs['sha256sum'],'historical hash identity unresolved: '+name)
        require(len(bad[name])==65 and re.fullmatch('[0-9a-f]{64}',bad[name]) is None,'unexpected bad constant structure')
        records.append({'artifact_role':name,'artifact_path':str(path),'constant_name':"EXPECTED['"+name+"']",'source_file':str(old),'function':'module EXPECTED; main -> payload','old_expected_digest':bad[name],'expected_syntax_valid_SHA256':False,'actual_digest':actual,'sha256sum':second,'evidence_backed_correct_digest':d,'independent_copy':obs,'record_error':'transcription: malformed 65-hex source literal; valid archived bytes match prior independent manifests and copies','artifact_mutation_supported':'NO','old_record_preserved':True})
    error=identity(ROOT/'logs/posttraining.error.json');failure=read('logs/posttraining.error.json')
    save('analysis/posttrain_hash_failure_audit.json',{'status':'PASS','HASH_CONSTANT_RECORD_ERROR':'SUPPORTED','ARTIFACT_MUTATION_SUPPORTED':'NO','HASH_IDENTITY_AMBIGUOUS':'NO','records':records,'original_failure':failure,'original_failure_log':error,'failed_command':sys.executable+' '+str(old),'expected_origin':'original scripts/a800_posttraining.py literals inherited by amended runner; frozen original also preserved','first_failure':'C5_u063 before model loading; C5_u113 and C6_u125 additional dormant malformed literals','training_artifacts_separate':'mismatch pertains to historical archived controls, not the newly trained selected R; all 63 candidate pairs rehashed against saved manifests','independent_evidence_metadata_only_transfer':True})
    save('provenance/posttrain_hash_constant_amendment_v1.json',{'amendment_id':TASK+'_HASH_IDENTITY_ONLY_V1','failed_gate':'fixed rotation source hash identity','status':'PASS','records':records,'resolved_EXPECTED':expected,'old_record_preserved':True,'artifact_mutation_supported':'NO','new_runner':'scripts/posttrain_closure_runner.py','scope':'hash identity and audit artifact writing only; no training/deployment/AIME'})
    before=json.dumps(bad,indent=2,sort_keys=True).splitlines(keepends=True)
    after=json.dumps(expected,indent=2,sort_keys=True).splitlines(keepends=True)
    with (ROOT/'analysis/posttrain_identity_resolution.diff').open('x') as f:
        f.write('Original archived source is not edited. Exact runtime EXPECTED override below.\n')
        f.writelines(difflib.unified_diff(before,after,fromfile='archived EXPECTED',tofile='evidence-backed EXPECTED'))
    with (ROOT/'analysis/posttrain_runner_source_addition.diff').open('x') as f:
        f.writelines(difflib.unified_diff([], (ROOT/'scripts/posttrain_closure_runner.py').read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='scripts/posttrain_closure_runner.py'))
    save('analysis/posttrain_cpu_prerequisites.json',{'status':'PASS','PRETASK_GIT_GATE':'PASS','gpu':gpu,'host':platform.node(),'freeze_sha256':FREEZE,'registration_sha256':PREREG,'teacher_documents':80,'teacher_cache_identity':'PASS','teacher_cache_parity':read('teacher/teacher_cache_parity.json')['status'],'final_hidden_indexing_sha256':sha(ROOT/'readiness/final_hidden_indexing_audit.json'),'qualification_fp_reference_sha256':sha(ROOT/'qualification/fp_runtime_stability.json'),'historical_inflight_guard_append':prior_append,'model_files_identity':'PASS','model_GPU_integrity':'PENDING_FRESH_LOAD','Shared16_manifest':p['validation'],'no_new_selection':True,'new_AIME_generation':0,'3090_deployment_started':False})
    print('POSTTRAIN_CPU_PREREQUISITES=PASS',flush=True)

if __name__=='__main__':
    try:preaudit()
    except BaseException as e:
        save('logs/posttrain_closure_audit.error.json',{'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'do_not_retry':True});raise
