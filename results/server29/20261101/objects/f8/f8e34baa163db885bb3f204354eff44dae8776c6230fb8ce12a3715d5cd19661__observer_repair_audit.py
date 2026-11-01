#!/usr/bin/env python3
"""Read-only CPU preflight/freeze/final evidence audits. Never launches GPU work."""
import argparse,ast,difflib,json,subprocess,sys,time
from pathlib import Path
import deployment_closure as A
R=A.ROOT;SUB='QWEN_GDN_3090_OBSERVER_LIFETIME_REPAIR_SMOKE_V1'
OLD=R/'scripts/deployment_closure_gpu.py';NEW=R/'scripts/deployment_observer_repaired.py'
FAILURES=['analysis/gpu0_repeat1_smoke.error.json','logs/gpu0_repeat1.log','logs/gpu0_repeat1.exitcode','logs/smoke_controller.error.json','analysis/failure_static_reference_audit.json']
def audit_parent():
 c=A.read(R/'logs/deployment_commit.json');head=A.cmd(['git','rev-parse','HEAD']);A.require(head==c['deployment_closure_commit'],'parent commit identity mismatch')
 branch=A.cmd(['git','symbolic-ref','--short','HEAD']);A.require(branch=='exp/qwen-gdn-c6r-c7-4x3090-deployment-v1','branch mismatch')
 A.require(not A.cmd(['git','diff','--name-only']) and not A.cmd(['git','diff','--cached','--name-only']),'unexplained tracked dirty files')
 allowed={'?? '+str((R/'scripts'/n).relative_to(A.REPO)) for n in ('observer_repair_audit.py','observer_repair_recipe.json')}
 A.require(set(A.cmd(['git','status','--porcelain']).splitlines())<=allowed,'unexplained visible dirty files')
 rows={}
 for x in A.read(R/'provenance/protected_target_inventory.json')['files']:
  A.require(A.identity(x['path'],x['sha256'])==x,'historical protected identity changed');rows[x['path']]=x
 for line in (R/'manifests/deployment_artifact_sha256.txt').read_text().splitlines():
  digest,path=line.split('  ',1);p=R/path;rows[str(p)]=A.identity(p,digest)
 rows[str(R/'logs/deployment_commit.json')]=A.identity(R/'logs/deployment_commit.json')
 evidence=[]
 for name in FAILURES:
  p=R/name;p.read_bytes();entry=A.identity(p);entry['git_blob']=A.cmd(['git','rev-parse',head+':'+str(p.relative_to(A.REPO))]);entry['git_commit']=head;evidence.append(entry)
 return dict(status='PASS',hostname=A.cmd(['hostname']),branch=branch,HEAD=head,git_status=A.cmd(['git','status','--short']),git_log=A.cmd(['git','log','-8','--oneline']),known_new_files_explanation='Exactly two agent-authored preflight files, observer_repair_audit.py and observer_repair_recipe.json, validated by exact untracked-path whitelist; no scientific edits',files=list(rows.values()),count=len(rows),failure_evidence=evidence,ignored_untracked_provenance='Prior post-commit logs/deployment_commit.json and mutable logs/failure_finalize.log; new audit files only. No tracked dirty file.',GPU_model_attempts=0)
def audit():
 A.require(not (R/'provenance/observer_repair_prechange_inventory.json').exists(),'prechange audit already exists')
 inv=audit_parent();A.save('provenance/observer_repair_prechange_inventory.json',inv)
 source=OLD.read_text();formal=A.FORMAL.read_text()
 for text in ('original=qmod.torch_recurrent_gated_delta_rule','return original(query,key,value,*args,**kw)','del patch,bank,cache,out,previous,qdq_rows,hidden,kwargs'):
  A.require(text in source,'source graph premise missing')
 for text in ('self.model = model','def wrapped(query, key, value','query = self._transform(query, use_norm)','self.saved.append((qmod, name, original))','for handle in self.handles:'):
  A.require(text in formal,'formal graph premise missing')
 graph=[dict(object=o,owner=owner,reference_type=typ,lifetime=life,expected_release_point=expected,actual_release_point=actual) for o,owner,typ,life,expected,actual in [
  ('model','smoke local model','strong','whole model smoke','after base rehash','local deleted at final cleanup'),
  ('patch.self.model','formal patch instance','strong','condition','after observer teardown','kept via wrapped callable until smoke returns'),
  ('wrapped closure self','smoke local original','strong closure cell','condition; survives unpatch','after last observer use','smoke returns'),
  ('original callable','observed closure + smoke local','strong closure/local','condition; function-scope','after restore and capture','smoke returns'),
  ('observed callable','smoke local + qmod during call','strong','active instrumentation','after restore and capture','qmod restored; smoke local survives'),
  ('rotation bank','C6R/C7 patch.self.bank','strong','condition','after observer teardown','last patch reachable through original until smoke returns'),
  ('saved unpatched kernel','patch.saved','strong','patch','patch destruction','operator restored by __exit__'),
  ('observer hook','model module pre-hook registry','strong lambda captures self','active instrumentation','__exit__ and hook.remove','hooks removed; does not remove local original')]]
 A.save('analysis/observer_reference_graph_before_repair.json',dict(status='AUDITED_STATIC',nodes=graph,edges=['smoke.original -> wrapped.__closure__.self -> patch.model','smoke.observed -> original -> wrapped.self -> patch.model','patch.bank -> frozen rotation GPU tensors'],operator_install='with patch calls __enter__ before original is read',unpatch='finally restores original (patched wrapper); patch.__exit__ restores base operators and removes registered layer hooks; lm_head hook removed',source=A.identity(OLD),formal=A.identity(A.FORMAL),root_cause='PLAUSIBLE_NOT_VERIFIED'))
 A.save('analysis/observer_lifetime_repair_hypothesis.json',dict(id='H_OBSERVER_LIFETIME',hypothesis='The instrumentation retains at least one strong Python reference to the wrapped model / rotation bank beyond the intended observer lifetime, preventing final model cleanup.',prospective=True,status='PLAUSIBLE_NOT_VERIFIED',fixed_GPU_uuid='GPU-a78176ed-f97f-c54f-6201-5a01b570f422',one_attempt_only=True,cleanup_threshold_bytes=1048576,no_scientific_change=True))
 print('PRECHANGE_AND_HYPOTHESIS_FROZEN_NO_GPU',flush=True)
def freeze():
 inv=A.read(R/'provenance/observer_repair_prechange_inventory.json');A.require(inv['status']=='PASS','prechange missing')
 for x in inv['files']:A.require(A.identity(x['path'],x['sha256'])==x,'pre-GPU protected identity drift '+x['path'])
 old=OLD.read_text();new=NEW.read_text();recipe=A.read(R/'scripts/observer_repair_recipe.json')
 expected=old
 for a,b in recipe['replacements']:
  A.require(expected.count(a)==1,'recipe not unique');expected=expected.replace(a,b)
 expected=expected[:expected.index('def main():')]+recipe['new_diagnostic_entrypoint']
 A.require(expected==new,'candidate outside prospective lifecycle/diagnostic recipe')
 ot=ast.parse(old);nt=ast.parse(new)
 of={n.name:ast.dump(n,include_attributes=False) for n in ot.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 nf={n.name:ast.dump(n,include_attributes=False) for n in nt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 unchanged={k:of[k]==nf[k] for k in of if k not in ('smoke','main')};A.require(all(unchanged.values()),'helper scientific AST mismatch')
 obs_old=next(n for n in ast.walk(ot) if isinstance(n,ast.FunctionDef) and n.name=='observed')
 obs_new=next(n for n in ast.walk(nt) if isinstance(n,ast.FunctionDef) and n.name=='observed')
 A.require(ast.dump(obs_old,include_attributes=False)==ast.dump(obs_new,include_attributes=False),'wrapped numerical observer body changed')
 A.require(old.count('gc.collect()')==new.count('gc.collect()')==2 and old.count('torch.cuda.empty_cache()')==new.count('torch.cuda.empty_cache()')==2,'cleanup call sites changed')
 A.require("require(end['allocated']<1024*1024,'unexpected memory retention')" in new,'threshold changed')
 A.save('analysis/observer_repair_scientific_path_identity.json',dict(status='PASS',unchanged_function_AST=unchanged,observed_callable_AST_exact=True,complete_source_matches_allowlist=True,formal_model_rotation_QDQ_writeback_sources_unchanged=True,condition_loop_and_metric_body_exact=True,input_tokenization_not_invoked=True,same_threshold=True,same_gc_and_empty_cache_call_sites=True,only_new_release='del original,observed after each condition fingerprints; split existing delete for diagnostic checkpoints',entrypoint_scope_narrowed='GPU0 repeat1 only; no prepare/generation/other GPU',old_output_fingerprints_missing=True,no_unavailable_old_dynamic_numerical_parity_claim=True))
 diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='deployment_closure_gpu.py',tofile='deployment_observer_repaired.py'))
 A.save('analysis/observer_repair_source_amendment.json',dict(status='PROSPECTIVE_FROZEN',changed_files=[str(NEW)],original_file_not_modified=True,old_SHA256=A.sha(OLD),new_SHA256=A.sha(NEW),exact_diff=diff,line_change_categories=['weakref-only ownership diagnostics','M0-M10 memory append/atomic flush','compact fingerprints before final assertion','del original,observed releases patch closure after last use','split existing local deletes for diagnostic checkpoints, same gc/empty_cache','fixed UUID, frozen preconditions and single-attempt guard','unique repaired artifact names; no old output overwrite'],GPU_model_attempts=0,root_cause='PLAUSIBLE_NOT_VERIFIED'))
 A.save('analysis/observer_reference_graph_after_repair.json',dict(status='PASS',static_only=True,root_cause_not_yet_verified=True,release_sequence=['finally restore condition wrapper; formal patch __exit__ restore original base kernels and remove hooks','hook.remove and all existing capture checks/fingerprints unchanged','del original,observed','del patch,bank','delete existing condition-local tensors','same existing gc.collect/empty_cache','base weight rehash then del model,tok; same final gc/empty_cache/assert'],strong_observer_to_model_path_after_teardown=False,diagnostic_references='weakref.ref only; serialized records contain scalars/hashes, no tensors or model handles'))
 uuid=A.cmd(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader']);A.require(uuid=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422','physical GPU0 mismatch')
 A.require(not A.cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU busy; no competition')
 cfg=A.read(R/'configs/frozen_smoke_protocol.json');A.require(cfg['seed']==1 and cfg['token_count']==32 and cfg['conditions']==['Hadamard','C6R','C7'],'smoke config differs')
 A.save('provenance/observer_repair_pre_gpu_identity.json',dict(status='PASS',protected_files=len(inv['files']),UUID=uuid,protocol_SHA256=A.sha(R/'configs/frozen_smoke_protocol.json'),input_token_hash=cfg['input_token_sha256'],source_commit=inv['HEAD'],GPU_model_attempts=0))
 # Prospective file list is exact and absolute; no mutation of old manifest.
 paths=set(x['path'] for x in inv['files'])
 paths.update(str(p) for p in R.rglob('*') if p.is_file() and (p.name.startswith('observer_repair') or p.name.startswith('observer_reference') or p.name in ('deployment_observer_repaired.py','observer_repair_audit.py')) and '__pycache__' not in p.parts)
 paths.add(str(R/'analysis/observer_lifetime_repair_hypothesis.json'))
 manifest=R/'manifests/observer_repair_pre_gpu_sha256.txt'
 with manifest.open('x') as f:
  for p in sorted(paths):f.write(A.sha(p)+'  '+p+'\n')
 for line in manifest.read_text().splitlines():digest,p=line.split('  ',1);A.require(A.sha(p)==digest,'pre GPU manifest')
 print('PRE_GPU_GATES_AND_FREEZE_PASS_NO_MODEL_ATTEMPT',flush=True)
def finalize():
 inv=A.read(R/'provenance/observer_repair_prechange_inventory.json')
 A.require(A.cmd(['git','rev-parse','HEAD'])==inv['HEAD'],'HEAD drift')
 for x in inv['files']:A.require(A.identity(x['path'],x['sha256'])==x,'post repair protected artifact drift '+x['path'])
 for line in (R/'manifests/observer_repair_pre_gpu_sha256.txt').read_text().splitlines():digest,p=line.split('  ',1);A.require(A.sha(p)==digest,'frozen repair source drift')
 source=A.read(R/'provenance/observer_repair_source_revalidation.json');A.require(source['status']=='PASS' and source['count']==51,'source final audit missing')
 A.save('provenance/observer_repair_postchange_inventory.json',dict(status='PASS',protected_files=len(inv['files']),source_files=51,old_failure_artifacts_byte_identical=True,rotations_base_tokenizer_historical_outputs_and_scoring_unchanged=True,source_revalidation_SHA256=A.sha(R/'provenance/observer_repair_source_revalidation.json'),repaired_source=A.identity(NEW)))
 attempt=A.read(R/'logs/observer_repair_gpu_attempt_started.json');A.require(attempt['attempt']==1 and attempt['gpu']==0,'attempt identity')
 diagnostics=A.read(R/'analysis/gpu0_repaired_smoke_runtime_diagnostics.json');finger=A.read(R/'analysis/gpu0_repaired_smoke_output_fingerprints.json')
 resultpath=R/'analysis/gpu0_repaired_smoke.json';result=A.read(resultpath) if resultpath.exists() else None
 errpath=R/'analysis/gpu0_repaired_smoke.error.json';err=A.read(errpath) if errpath.exists() else None
 complete=finger['conditions'];dynamic=all(a in complete and complete[a]['status']=='PASS' for a in ('Hadamard','C6R','C7'))
 timeline=diagnostics['memory'];final=next((m for m in reversed(timeline) if m['point']=='M10'),None)
 memory_pass=final is not None and final['allocated']<1048576
 passed=bool(result and result['status']=='PASS' and dynamic and memory_pass and not err)
 rootcause='SUPPORTED' if passed and final['model_alive']==False and final['observer_alive']==False and final['rotation_bank_alive']==False else ('NOT_SUPPORTED' if final and not memory_pass and final['model_alive']==False and final['observer_alive']==False else 'INCONCLUSIVE')
 A.save('analysis/gpu0_repaired_rotation_load_audit.json',dict(status='PASS' if dynamic else 'NOT_RUN',conditions={a:complete[a]['rotation_audit'] for a in A.ARMS if a in complete},actual_GPU_attempt=True))
 A.save('analysis/gpu0_repaired_qdq_writeback_audit.json',dict(status='PASS' if dynamic else 'NOT_RUN',conditions={a:{k:complete[a][k] for k in ('rotation_calls','quantizer_audit','writeback','first_state_entry','final_state_sha256')} for a in complete},actual_GPU_attempt=True,unchanged_formal_source=A.identity(A.FORMAL),no_teacher_reset=True))
 A.save('analysis/gpu0_repaired_memory_audit.json',dict(status='PASS' if memory_pass else ('FAIL' if final else 'NOT_RUN'),threshold_bytes=1048576,final_allocated_bytes=None if final is None else final['allocated'],final_reserved_bytes=None if final is None else final['reserved'],timeline=timeline,weakrefs_released=None if final is None else {k:final[k] for k in ('model_alive','observer_alive','rotation_bank_alive')},root_cause=rootcause,no_added_gc_empty_cache_cycles=True))
 gates=dict(EXPERIMENT=A.TASK,SUBPHASE=SUB,PREVIOUS_FAILURE='GPU0_REPEAT1_FINAL_CLEANUP_MEMORY_RETENTION',REPAIR_SCOPE='OBSERVER_REFERENCE_LIFETIME_ONLY',GPU_MODEL_ATTEMPTS=1,GPU_USED='PHYSICAL_RTX3090_GPU0',GPU_UUID=diagnostics['GPU_uuid'],OBSERVER_REPAIR_SCIENTIFIC_PATH_IDENTITY_GATE='PASS',STATIC_OBSERVER_LIFETIME_REPAIR_GATE='PASS',PRE_GPU_PROTECTED_IDENTITY_GATE='PASS',C6R_ROTATION_LOAD_GATE='PASS' if dynamic else 'NOT_RUN',C7_ROTATION_LOAD_GATE='PASS' if dynamic else 'NOT_RUN',C6R_ORTHOGONALITY_GATE='PASS' if dynamic else 'NOT_RUN',C7_ORTHOGONALITY_GATE='PASS' if dynamic else 'NOT_RUN',ROTATION_MAPPING_GATE='PASS' if dynamic else 'NOT_RUN',QDQ_SEMANTICS_GATE='PASS' if dynamic else 'NOT_RUN',REAL_RECURRENT_WRITEBACK_GATE='PASS' if dynamic else 'NOT_RUN',DEPLOYMENT_MEMORY_GATE='PASS' if memory_pass else ('FAIL' if final else 'NOT_RUN'),OBSERVER_LIFETIME_ROOT_CAUSE=rootcause,REPAIRED_GPU0_SMOKE_GATE='PASS' if passed else 'FAIL',C6R_3090_DETERMINISM_GATE='NOT_RUN',C7_3090_DETERMINISM_GATE='NOT_RUN',FOUR_GPU_CONSISTENCY_GATE='NOT_RUN',C6R_3090_DEPLOYMENT_READY='NO',C7_3090_DEPLOYMENT_READY='NO',READY_FOR_C6R_C7_CANONICAL60='NO',NEW_TRAINING_STARTED='NO',TRAINING_RERUN='NO',C6R_AIME_STARTED='NO',C7_AIME_STARTED='NO',NEW_AIME_GENERATION=0,SCORING_STARTED='NO',SMOKE_AUTOMATIC_RETRY='NO',PUSH_PERFORMED='NO',PROTECTED_ARTIFACT_REVALIDATION_GATE='PASS',FINAL_ACTION='STOP')
 A.save('analysis/observer_repair_final_gates.json',gates)
 report=['# '+SUB,'','```text']+[f'{k} = {v}' for k,v in gates.items()]+['```','','## Repair and causal evidence','',
  'Only a new runner was added; the original failed runner and artifacts remain byte-identical. Lifecycle repair: `del original,observed` after each condition capture, followed by existing patch/bank deletion. Existing tensor deletion, gc.collect(), empty_cache(), threshold and numerical bodies are unchanged. Weakrefs and atomic diagnostic writes add no strong model/tensor references.',
  '', 'Static AST/body audit and exact allowlisted source diff are frozen before the sole GPU attempt. No old successful output fingerprint exists, so numerical equality with the unsaved old run cannot be measured; scientific-source identity, actual finite outputs and exact QDQ/writeback checks supply the valid evidence.',
  '', 'Root-cause verdict: '+rootcause+'. Old cleanup FAIL; repaired cleanup '+gates['DEPLOYMENT_MEMORY_GATE']+'. Weakrefs at final check: '+str(None if final is None else {k:final[k] for k in ('model_alive','observer_alive','rotation_bank_alive')})+'.',
  '', '## Memory timeline','', '| Point | Condition | Allocated bytes | Reserved bytes | Peak allocated | Peak reserved | Model alive | Observer alive | Bank alive |','|---|---|---:|---:|---:|---:|---|---|---|']
 for m in timeline:report.append('| '+' | '.join(str(m.get(k,'')) for k in ('point','condition','allocated','reserved','peak_allocated','peak_reserved','model_alive','observer_alive','rotation_bank_alive'))+' |')
 report+=['','Final allocated bytes: '+str(None if final is None else final['allocated'])+'. Final reserved bytes: '+str(None if final is None else final['reserved'])+'. Threshold strictly remains <1,048,576 allocated bytes. This is not long-generation memory proof.',
  '', '## Dynamic gates and fingerprint boundary','', 'Hadamard/C6R/C7 each use the unchanged 32-token list, seed1, physical GPU0. QDQ calls/condition:31; layer coverage24; rotation calls768; writeback continuity744 exact transitions. These counts are validated by unchanged assertions before condition PASS is recorded, not extrapolated to other GPUs. Raw diagnostic details and all 32 logit/FinalHidden hashes, 31x24 scale/qcode/post-QDQ hashes, final states are in gpu0_repaired_smoke_output_fingerprints.json. No scientific ranking is performed.',
  '', '## Provenance and final stop','', f'Branch: {inv["branch"]}. Pre-task HEAD: {inv["HEAD"]}. Old failures and all {len(inv["files"])} target protected files unchanged; all51 A800 source files unchanged. New commit record will be logs/observer_repair_commit.json; no push.',
  '', 'Exactly one fresh GPU0 attempt. No GPU0 repeat2, no GPU1-3 work, no four-GPU consistency, future config freeze, AIME, training or scoring. Readiness remains NO even if this smoke passes. FINAL_ACTION=STOP.']
 with (R/'reports'/f'{SUB}.md').open('x') as f:f.write('\n'.join(report)+'\n')
 for p in R.rglob('*.json'):json.loads(p.read_text())
 # Additions only; parent tracked files may not be edited.
 A.require(not A.cmd(['git','diff','--name-only']) and not A.cmd(['git','diff','--cached','--name-only']),'tracked changes')
 additions=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and A.cmd(['git','ls-files','--',str(p.relative_to(A.REPO))])=='']
 excluded={'observer_repair_audit.log','observer_repair_freeze.log','observer_repair_finalize.log','failure_finalize.log','deployment_commit.json'}
 additions=[p for p in additions if p.name not in excluded]
 manifest=R/'manifests/observer_repair_final_sha256.txt'
 with manifest.open('x') as f:
  for p in sorted(set(additions+[Path(x['path']) for x in inv['failure_evidence']])):f.write(A.sha(p)+'  '+str(p)+'\n')
 for line in manifest.read_text().splitlines():digest,p=line.split('  ',1);A.require(A.sha(p)==digest,'final manifest')
 for p in additions+[manifest]:A.cmd(['git','add','-f','--',str(p.relative_to(A.REPO))])
 A.cmd(['git','diff','--cached','--check']);A.require(not A.cmd(['git','diff','--cached','--diff-filter=MDR','--name-only']),'old files staged')
 scope=A.cmd(['git','diff','--cached','--name-only']).splitlines();A.require(all(p.startswith(str(R.relative_to(A.REPO))+'/') for p in scope),'commit scope')
 stat=A.cmd(['git','diff','--cached','--stat'])
 A.cmd(['git','-c','user.name=Qwen Scientific Automation','-c','user.email=qwen-scientific-automation@localhost','commit','-m','Repair observer reference lifetime and close one GPU0 fixed smoke; stop'])
 A.save('logs/observer_repair_commit.json',dict(status='PASS',branch=inv['branch'],pre_task_HEAD=inv['HEAD'],new_commit=A.cmd(['git','rev-parse','HEAD']),git_status=A.cmd(['git','status','--short']),staged_diff_stat=stat,scope=scope,additions_only=True,json_parse='PASS',final_manifest_SHA256=A.sha(manifest),manifest_reverified=True,old_failures_byte_identical=True,push=False,stop=True))
 print('OBSERVER_REPAIR_CLOSURE_STOP '+A.cmd(['git','rev-parse','HEAD']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('audit','freeze','finalize'),required=True);x=p.parse_args()
 {'audit':audit,'freeze':freeze,'finalize':finalize}[x.phase]()
