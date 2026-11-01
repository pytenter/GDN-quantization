"""Narrow audit-only adapter and write-once exact V3 recovery orchestration.

Original scientific module, setup guards, numerical core and history remain
unchanged. Only the individually audited track_b function is rebound, with an
independently sealed source hash. No model work is repeated for valid units.
"""
import argparse
import ast
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import recovery_audit as A

N=A.N;S=A.S;ROOT=A.PARENT;REC=A.REC
RUNTIME=Path('/data/zypan/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime')


def event(kind,**data):
    with (RUNTIME/'recovery_002_events.jsonl').open('a') as f:
        f.write(json.dumps({'utc':N.now(),'event':kind,**data},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def bind_audit_repair():
    proof=A.patch_proof()
    frozen=N.read(REC/'RECOVERY_PREMEASUREMENT_FREEZE.json')
    for path,wanted in frozen['source_hashes'].items():N.check(N.sha(path)==wanted,'recovery source freeze '+path)
    source=REC/'scripts/scale_science_audit_repaired.py'
    tree=ast.parse(source.read_bytes().decode('utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='track_b')
    # setup/Harness/teacher/metrics retain their exact original code and guards.
    # The sole repaired function carries the real corrected source filename.
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),S.__dict__)
    return proof


def child(phase,gpu,doc,**options):
    command=[sys.executable,str(Path(__file__)),phase,'--gpu',str(gpu),'--doc',str(doc)]
    for k,v in options.items():
        if v:command.append('--'+k)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    path=RUNTIME/f'recovery_002_{phase}_doc{doc}_gpu{gpu}.log'
    log=path.open('x')
    worker=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
    event('CHILD_STARTED',phase=phase,doc=doc,gpu=gpu,pid=worker.pid,log=str(path))
    return worker


def dry_run():
    N.check(N.read(REC/'CPU_REGRESSION.json')['status']=='PASS','CPU regression completed')
    N.check(N.read(REC/'POSTSTOP_REUSE_AUDIT.json')['status']=='PASS','original closure audit')
    table=N.read(REC/'FORMAL_UNIT_LEDGER_BEFORE_RESUME.json');spec=N.read(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json')
    resources=N.resources();N.check(not resources['processes_csv'],'all GPUs idle before resume plan')
    config=N.read('/data/zypan/modelscope_models/Qwen3.5-9B/config.json')['text_config']
    attention=config['layer_types'].count('full_attention')
    kv_per_token=2*attention*config['num_key_value_heads']*config['head_dim']*2
    gdn_bytes=len(S.LAYERS)*32*128*128*4
    # Actual source has one complete FP model cache, one FP replay state, and
    # four students. Two complete checkpoints/document are retained unchanged.
    one_checkpoint=kv_per_token*8256+6*gdn_bytes
    target_sizes=[(ROOT/f'track_B/doc_{d:02d}/original_teacher_targets.pt').stat().st_size for d in [1,2,4]]
    missing_b_estimate=max(target_sizes)
    drives_estimate=4*64*1024*1024
    serializer_and_reports_allowance=2*1024**3
    peak_additional=8*one_checkpoint+missing_b_estimate+drives_estimate+serializer_and_reports_allowance
    N.check(peak_additional<=resources['disk_available_bytes'],'safe unchanged capture/checkpoint storage estimate')
    packages={}
    for name in ['torch','triton','transformers','flash-linear-attention','fla-core']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='NOT_INSTALLED_AS_DISTRIBUTION'
    import torch
    reusable=[u['unit_id'] for u in table['units'] if u['reuse_eligible']]
    pending=[u['unit_id'] for u in table['units'] if not u['reuse_eligible']]
    N.check(len(reusable)==12 and len(pending)==36 and set(reusable).isdisjoint(pending),'exact unit partition')
    plan={'status':'PASS','utc':N.now(),'dry_run_only':True,'total_expected_scientific_units':48,
          'reusable_units':reusable,'units_requiring_execution':pending,'units_blocked':[],
          'execution_bundles':{'TrackB':[{'doc':3,'gpu':2,'conditions':list(S.CONDITIONS)}],
                               'TrackA':[{'doc':d,'gpu':d-1,'conditions':list(S.CONDITIONS)} for d in range(1,5)],
                               'MatchedLong':[{'doc':d,'gpu':d-1,'conditions':list(S.CONDITIONS)} for d in range(1,5)]},
          'engineering_prerequisite':'unchanged original matched_history prefix256 Doc1/GPU0; one qualification only',
          'B_engineering_and_numerical_and_observer':'REUSE_NO_RERUN',
          'resource_snapshot':resources,'existing_artifact_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
          'storage':{'KV_bytes_per_token':kv_per_token,'estimated_complete_anchor_bytes_upper_geometry':one_checkpoint,
                     'two_anchors_four_documents_bytes':8*one_checkpoint,'missing_B_teacher_bytes_estimate':missing_b_estimate,
                     'short_drive_bytes_estimate':drives_estimate,'serialization_and_report_allowance_bytes':serializer_and_reports_allowance,
                     'peak_additional_bytes_estimate':peak_additional,'available_bytes':resources['disk_available_bytes'],
                     'headroom_bytes_after_estimate':resources['disk_available_bytes']-peak_additional,
                     'actual_checkpoint_and8192_VRAM_gate':'original source checks remain mandatory; estimate not a waiver'},
          'environment':{'packages':packages,'CUDA':torch.version.cuda,'driver':N.cmd(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),
                         'CUBLAS_WORKSPACE_CONFIG':':4096:8','strict_deterministic':True,'warn_only':False,'TF32':False},
          'no_fixed_GPUh_or_VRAM_percentage_cap':True,
          'command':f'env CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 {sys.executable} {REC}/scripts/resume_v3.py controller',
          'idempotence':'exclusive lock; finished execution verified/no-op; started unfinished attempt cannot relaunch; immutable result and log writes'}
    A.save('RESUME_PLAN.json',plan)
    print(json.dumps({'DRY_RUN':'PASS','reuse_units':12,'pending_units':36,'pending_B_document':3,
                      'additional_storage_GiB':peak_additional/1024**3,'headroom_GiB':plan['storage']['headroom_bytes_after_estimate']/1024**3}),flush=True)


def freeze():
    N.check(N.read(REC/'RESUME_PLAN.json')['status']=='PASS','dry-run prerequisite')
    N.check(N.read(REC/'MODEL_CHECKPOINT_IDENTITY_REUSE.json')['status']=='PASS','exact existing checkpoint identity')
    files=[ROOT/'scripts/numerical_recovery.py',ROOT/'scripts/scale_science.py',ROOT/'scripts/matched_history.py',ROOT/'scripts/finalize_discovery.py',
           REC/'scripts/recovery_audit.py',REC/'scripts/resume_v3.py',REC/'scripts/scale_science_audit_repaired.py',
           ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json',ROOT/'NEW_NUMERICAL_CONTRACT.json',ROOT/'QUALIFICATION_POLICY.json',
           REC/'CPU_REGRESSION.json',REC/'MINIMAL_PATCH_AUDIT.json',REC/'POSTSTOP_REUSE_AUDIT.json',REC/'RESUME_PLAN.json',
           REC/'MODEL_CHECKPOINT_IDENTITY_REUSE.json',REC/'scripts/model_identity_reuse.py']
    A.save('RECOVERY_PREMEASUREMENT_FREEZE.json',{'status':'PASS','utc':N.now(),'source_hashes':{str(p):N.sha(p) for p in files},
           'old_source_revision':A.OLD_COMMIT,'repair_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO),
           'scientific_protocol_changed':False,'qualified_source_difference':'one read-only text decode expression; no production tensor computation changed'})
    print('RECOVERY_PREMEASUREMENT_FREEZE_PASS',flush=True)


def formal_b_panel():
    allowed=[A.OLD_COMMIT,N.read(REC/'RECOVERY_PREMEASUREMENT_FREEZE.json')['repair_commit']]
    rows=[];receipts=[]
    for doc in range(1,5):
        path=ROOT/f'track_B/doc_{doc:02d}/result.json';data=A.verify_result(doc,path,allowed)
        if doc in [1,2,4]:
            expected=N.read(REC/'POSTSTOP_REUSE_AUDIT.json')['manifest_entries'][str(path.relative_to(ROOT))]
            N.check(N.sha(path)==expected,'reused formal receipt unchanged')
        else:
            ended=N.read(REC/'execution/track_b_doc3_COMPLETED.json')
            N.check(ended['return_status']=='SUCCESS' and ended['result_sha256']==N.sha(path),'recovery Doc3 explicit end evidence')
        rows.extend(data['scores']);receipts.append({'doc':doc,'sha256':N.sha(path),'source_commit':data['source_commit'],'action':'REUSE' if doc!=3 else 'EXECUTED_ZERO_FORWARD_RECOVERY'})
    keys=[(r['document_index'],r['condition'],r['H']) for r in rows]
    N.check(len(keys)==len(set(keys))==32,'full exact32 B panel')
    if not (ROOT/'TRACK_B_SHORT_SCORES.csv').exists():
        S.csv_write('TRACK_B_SHORT_SCORES.csv',rows)
        S.csv_write('ORIGINAL_C5_C6_BASELINES.csv',[{k:r[k] for k in ['document_index','condition','H','B0','B1','B2','teacher_target_sha256','document_sha256','token_sha256']} for r in rows])
        N.save('TRACK_B_SHORT_SCORE_FREEZE.json',{'status':'PASS','utc':N.now(),'rows':32,'sha256':N.sha(ROOT/'TRACK_B_SHORT_SCORES.csv'),
            'baseline_sha256':N.sha(ROOT/'ORIGINAL_C5_C6_BASELINES.csv'),'spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
            'long_G_read':False,'recovery_attempt':'002','full_panel_integrity':'PASS','receipts':receipts})
    A.save('TRACK_B_PANEL_INTEGRITY.json',{'status':'PASS','rows':32,'independent_documents':4,'unique_conditions_horizons':True,
        'all_atomic_receipts_verified':True,'qualifying_scores_mixed_into_formal':False,'receipts':receipts,
        'frozen_scores_sha256':N.sha(ROOT/'TRACK_B_SHORT_SCORES.csv')})
    event('TRACK_B_COMPLETE',receipts=receipts)


def history_module():
    import matched_history as H
    return H


def science_worker(phase,gpu,doc,engineering):
    bind_audit_repair()
    N.check(not (REC/f'execution/{phase}_doc{doc}_STARTED.json').exists(),'no duplicate attempt unit start')
    A.save(f'execution/{phase}_doc{doc}_STARTED.json',{'utc':N.now(),'pid':os.getpid(),'GPU':gpu,'GPU_UUID':S.UUIDS[gpu],
           'phase':phase,'doc':doc,'repair_commit':N.read(REC/'RECOVERY_PREMEASUREMENT_FREEZE.json')['repair_commit']})
    if phase=='track_b':
        N.check(doc==3 and gpu==2 and not engineering,'only zero-forward Doc3 formal B recovery')
        N.check(not (ROOT/'track_B/doc_03/result.json').exists() and not (ROOT/'track_B/doc_03/original_teacher_targets.pt').exists(),'no unqualified partial Doc3 overwrite')
        S.track_b(gpu,doc,False)
        result=ROOT/'track_B/doc_03/result.json'
    elif phase=='history_engineering':
        history_module().prefix(gpu,doc,True);result=ROOT/f'history_qualification/doc_{doc:02d}/qualification.json'
    elif phase=='prefix':
        history_module().prefix(gpu,doc,False);result=ROOT/f'track_A/doc_{doc:02d}/short_result.json'
    else:
        history_module().continuation(gpu,doc);result=ROOT/f'long/doc_{doc:02d}/result.json'
    A.save(f'execution/{phase}_doc{doc}_COMPLETED.json',{'utc':N.now(),'return_status':'SUCCESS','result':str(result.relative_to(ROOT)),
          'result_sha256':N.sha(result),'GPU_UUID':S.UUIDS[gpu],'pid':os.getpid()})


def wait_exact(workers,phase):
    codes=[w.wait() for w in workers]
    A.save(f'execution/{phase}_WORKER_EXITS.json',{'utc':N.now(),'phase':phase,'exitcodes':codes,'pids':[w.pid for w in workers]})
    N.check(codes==[0]*len(workers),phase+' new hard failure; no automatic scientific retry')
    N.check(not N.resources()['processes_csv'],phase+' GPU release before dependent stage')


def final_delivery():
    import finalize_discovery as F
    F.science_analysis()
    ledger=F.resource_ledger();A.save('RESOURCE_LEDGER.json',ledger)
    units=N.read(REC/'FORMAL_UNIT_LEDGER_BEFORE_RESUME.json')
    provenance=[]
    for stage,rel in [('B','track_B/doc_{d:02d}/result.json'),('A','track_A/doc_{d:02d}/short_result.json'),('LONG','long/doc_{d:02d}/result.json')]:
        for doc in range(1,5):
            path=ROOT/rel.format(d=doc);data=N.read(path)
            N.check(data['status']=='PASS','complete final worker receipt')
            provenance.append({'stage':stage,'doc':doc,'artifact':str(path.relative_to(ROOT)),'sha256':N.sha(path),
                'source_commit':data['source_commit'],'reuse':stage=='B' and doc!=3,'GPU':doc-1,'GPU_UUID':S.UUIDS[doc-1],
                'model_rotation_quantizer_identity_sha256':N.sha(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json'),
                'numerical_contract_sha256':N.sha(ROOT/'NEW_NUMERICAL_CONTRACT.json'),'metric_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json')})
    A.save('EXECUTION_PROVENANCE.json',{'status':'PASS','receipts':provenance,'scientific_units':48,'formal_qualification_separate':True})
    analysis=N.read(ROOT/'INCREMENTAL_INFORMATION_ANALYSIS.json')
    statuses={'NUMERICAL_CLOSURE_STATUS':'PASS','HISTORICAL_BITWISE_PARITY':'FAIL_PRESERVED','OBSERVER_PARITY':'PASS',
              'TRACK_B_STATUS':'COMPLETE','TRACK_A_STATUS':'COMPLETE','LONG_HORIZON_STATUS':'COMPLETE_MATCHED_NEW_PANEL',
              'SURROGATE_DISCOVERY_STATUS':'DESCRIPTIVE_DISCOVERY_COMPLETE_NOT_CONFIRMED'}
    A.save('SCIENTIFIC_VERDICT.json',{'status':statuses,'independent_n':4,'confirmation':'NOT_CONFIRMED_DISCOVERY_ONLY',
        'future_functional_observability':'NOT_TESTED; E_future is state error, not full-model KL/AIME',
        'Track_A_functional_baseline':'NOT_COMPARABLE_FROZEN_DRIVE','old_G_used':False,
        'training':False,'new_rotation':False,'AIME':False,'generation':False,'independent_KL':False,'gradient_optimization':False})
    report=['# V3 精确恢复与 Discovery 报告','',json.dumps(statuses,ensure_ascii=False,indent=2),'',
        '原 CRLF 文本审核误报已作单表达式修复。原 STOP、原报告、原 manifest 与 POSTSTOP 全部保留。',
        '复用 Doc1/2/4 的正式 Track B，仅执行零 forward 的 Doc3；数值 1110 对象、三进程复现与 observer 资格均复用。',
        '只有完整 Track B 门禁通过后执行真实8192 anchor、short全局冻结、fresh-process resume parity和同runtime32K长期面板。',
        '四篇已知文档仅是 Discovery；没有独立 confirmation，不能把旋转对、token、layer 当独立样本。',
        'Track B 的 B0/B1/B2 使用原始 teacher/helper/out_proj；Track A 的 B1/B2 不合法可比，保持 NOT_COMPARABLE。',
        'E_future 是不重叠未来 recurrent-state 损伤，不是 future functional KL 或推理正确率。','',
        '## 实际分数','']
    for name in ['TRACK_B_SHORT_SCORES.csv','TRACK_A_8192_SHORT_SCORES.csv','MATCHED_LONG_G.csv']:
        rows=F.read_csv(name)
        keys=['document_index','condition','H','S1','S2','D1','D2','B0','B1','B2'] if 'SHORT' in name else ['document_index','condition','J8192','J32768','G','E_future']
        report+=['### '+name,'','| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
        report+=['| '+' | '.join(r.get(k,'') or 'NOT_COMPARABLE' for k in keys)+' |' for r in rows]+['']
    report+=['## 逐文档排名预测（n=4，描述性）','','| Track | H | 指标 | 标签 | Doc1–4 Spearman | 均值 | 配对一致率均值 |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in analysis['document_level_summaries']:
        report+=['| '+f"{r['track']} | {r['H']} | {r['metric']} | {r['label']} | {json.dumps(r['per_document_spearman'])} | {r['mean_document_spearman']} | {r['mean_document_concordance']}"+' |']
    report+=['','S1 与 B2/S2/D1/D2 的分歧及原始胜负方向见 INCREMENTAL_INFORMATION_ANALYSIS.json；没有追加筛选规则或统计阈值。',
        '目前只能评估长程状态排序关联，不能由局部状态信号确认未来 functional observability。下一步如有必要，只建议独立文档、outcome-blind协议的functional验证；本轮不执行。','',
        f"记录实际 GPUh={ledger['recorded_completed_phase_GPU_hours']:.6f}；峰值 allocated={ledger['peak_allocated_bytes']/1024**3:.3f}GiB；reserved={ledger['peak_reserved_bytes']/1024**3:.3f}GiB。",'']
    with (REC/'FINAL_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report))
    seal_final(statuses)


def seal_final(statuses):
    # Copy only already closed external logs; final seal is never self-mutating.
    dest=REC/'closed_logs';dest.mkdir(exist_ok=True)
    for path in sorted(RUNTIME.glob('recovery_002_*.log')):
        if path.name=='recovery_002_controller.log':continue
        shutil.copyfile(path,dest/path.name)
    historical=N.read(REC/'POSTSTOP_REUSE_AUDIT.json')['manifest_entries']
    for rel,wanted in historical.items():N.check(N.sha(ROOT/rel)==wanted,'historical evidence byte unchanged '+rel)
    A.save('COMPLETE.json',{'status':'COMPLETE','utc':N.now(),'stage_status':statuses,'historical_evidence_unchanged':True,
        'controller_log':'external operational log excluded; structured completion and child logs sealed',
        'science_units':48,'new_attempt':'002','source_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO)})
    manifest=REC/'FINAL_IMMUTABLE_SHA256SUMS.txt'
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=manifest and p.name not in ['ACTIVE_LOCK'])
    with manifest.open('x',encoding='utf-8') as f:f.write(''.join(N.sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    for line in manifest.read_text().splitlines():
        digest,rel=line.split('  ',1);N.check(N.sha(ROOT/rel)==digest,'final immutable complete artifact manifest')
    print(json.dumps({'FINAL':'COMPLETE','manifest_sha256':N.sha(manifest),'report':str(REC/'FINAL_REPORT.md')}),flush=True)


def controller():
    lock=(REC/'ACTIVE_LOCK').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('ALREADY_RUNNING_NO_DUPLICATE_LAUNCH',flush=True);return
    if (REC/'COMPLETE.json').exists():print('ALREADY_COMPLETE_NO_DUPLICATE_LAUNCH',flush=True);return
    N.check(not (REC/'CONTROLLER_STARTED.json').exists(),'unfinished attempt already started; no blind relaunch')
    bind_audit_repair();N.check(not N.resources()['processes_csv'],'no GPU competition after old controller exited')
    A.save('CONTROLLER_STARTED.json',{'utc':N.now(),'pid':os.getpid(),'source_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO)})
    event('EXACT_RECOVERY_STARTED',reused_B_docs=[1,2,4],executed_B_doc=3)
    wait_exact([child('track_b',2,3)],'TRACK_B_RECOVERY')
    formal_b_panel()
    H=history_module()
    H.prepare()
    A.save('HISTORY_FREEZE_RECEIPT.json',{'status':'PASS','utc':N.now(),'history_spec_sha256':N.sha(ROOT/'HISTORY_PREMEASUREMENT_SPEC.json'),
        'history_code_sha256':N.sha(ROOT/'scripts/matched_history.py'),'science_spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
        'repair_commit':N.read(REC/'RECOVERY_PREMEASUREMENT_FREEZE.json')['repair_commit']})
    wait_exact([child('history_engineering',0,1)],'HISTORY_ENGINEERING')
    event('TRACK_A_STARTED')
    wait_exact([child('prefix',d-1,d) for d in range(1,5)],'TRACK_A_PREFIX')
    rows=[]
    for doc in range(1,5):
        data=N.read(ROOT/f'track_A/doc_{doc:02d}/short_result.json')
        N.check(data['status']=='PASS' and data['QDQ_calls_to8256']=={c:8256 for c in S.CONDITIONS},'real persistent8192/8256 coverage')
        for artifact in data['anchors']:
            N.check(N.sha(ROOT/artifact['path'])==artifact['sha256'],'real retained anchor tensors')
        rows.extend(data['scores'])
    N.check(len(rows)==32 and len({(r['document_index'],r['condition'],r['H']) for r in rows})==32,'exact TrackA full panel')
    S.csv_write('TRACK_A_8192_SHORT_SCORES.csv',rows)
    files=['TRACK_B_SHORT_SCORES.csv','ORIGINAL_C5_C6_BASELINES.csv','TRACK_A_8192_SHORT_SCORES.csv','SCIENCE_PREMEASUREMENT_SPEC.json','HISTORY_PREMEASUREMENT_SPEC.json','NEW_NUMERICAL_CONTRACT.json']
    N.save('ALL_SHORT_SCORES_FREEZE.json',{'utc':N.now(),'status':'PASS','hashes':{rel:N.sha(ROOT/rel) for rel in files},
        'matched_long_G_read':False,'old_G_pairing':'NOT_AUTHORIZED','independent_documents':4,'recovery_attempt':'002'})
    for doc in range(1,5):N.check(N.read(ROOT/f'track_A/doc_{doc:02d}/LONG_RESOURCE_PREFLIGHT.json')['status']=='PASS','original actual32K memory gate')
    event('ALL_SHORT_SCORES_FROZEN_BEFORE_LONG_LABELS',sha256=N.sha(ROOT/'ALL_SHORT_SCORES_FREEZE.json'))
    wait_exact([child('continuation',d-1,d) for d in range(1,5)],'MATCHED_LONG')
    rows=[]
    for doc in range(1,5):
        data=N.read(ROOT/f'long/doc_{doc:02d}/result.json')
        N.check(data['status']=='PASS' and data['actual_total_scientific_tokens']==32768 and data['QDQ_calls']=={c:32768 for c in S.CONDITIONS},'matched full long execution')
        N.check(N.read(ROOT/f'long/doc_{doc:02d}/RESUME_PARITY.json')['status']=='PASS','fresh complete-cache resume parity')
        rows.extend(data['scores'])
    N.check(len(rows)==len({(r['document_index'],r['condition']) for r in rows})==16,'unique16 long matched units')
    S.csv_write('MATCHED_LONG_G.csv',rows)
    S.csv_write('FUTURE_ONLY_ERROR.csv',[{k:r[k] for k in ['document_index','condition','E_future','future_tokens','document_sha256','token_sha256']} for r in rows])
    N.save('MATCHED_PANEL_COMPLETE.json',{'status':'PASS','documents':4,'conditions':4,'Track_A_rows':32,'long_rows':16,'old_G_used_for_pairing':False,
        'new_full_model_KL_executed':False,'generation':False,'training':False,'recovery_attempt':'002'})
    event('FINAL_ANALYSIS_STARTED')
    final_delivery()


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dry_run','freeze','controller','track_b','history_engineering','prefix','continuation'])
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--doc',type=int,default=1);p.add_argument('--engineering',action='store_true');args=p.parse_args()
    try:
        if args.phase=='dry_run':dry_run()
        elif args.phase=='freeze':freeze()
        elif args.phase=='controller':controller()
        else:science_worker(args.phase,args.gpu,args.doc,args.engineering)
    except Exception as exc:
        rel=f'execution/{args.phase}_doc{args.doc}_STOP.json'
        if not (REC/rel).exists():A.save(rel,{'status':'FAIL_AND_STOP','utc':N.now(),'phase':args.phase,'doc':args.doc,
            'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_scientific_retry':False})
        event('FAIL_AND_STOP',phase=args.phase,doc=args.doc,message=str(exc));raise


if __name__=='__main__':main()
