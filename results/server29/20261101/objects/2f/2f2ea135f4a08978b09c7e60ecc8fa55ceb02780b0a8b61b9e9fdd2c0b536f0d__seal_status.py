"""Create reviewable delivery snapshots; missing measurements remain missing."""
import csv
import fcntl
import io
import json
import os
import shutil
import time
from pathlib import Path
import functional_worker as w

D = Path(__file__).resolve().parent
DELIVERY = D/'delivery'; DELIVERY.mkdir(exist_ok=True)


def save(name,value):
    target = DELIVERY/name; tmp = target.with_name(target.name+'.partial')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    os.replace(tmp,target)


def write_csv(name, rows, fields):
    stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    (DELIVERY/name).write_text(stream.getvalue())


def main():
    spec = w.read(D/'FROZEN_SPEC.json')
    stop = w.read(D/'STOP.json') if (D/'STOP.json').exists() else None
    gates_path = D/'QUALIFICATION_GATES_COMPLETED.json'
    gates = w.read(gates_path) if gates_path.exists() else w.read(D/'QUALIFICATION_GATES.json')
    gates = dict(gates)
    gates['secondary_definitions'] = w.read(D/'SECONDARY_SPEC.json')['status']
    gates['secondary_spec_sha256'] = w.sha(D/'SECONDARY_SPEC.json')
    references = []; results = []; documents = []; jobs = []
    eps = spec['resolution']['threshold']
    for entry in spec['documents']:
        doc = entry['document']; r = {}
        for condition in ('FP',entry['P1_predicted_winner'],entry['L7_predicted_winner']):
            path = D/'formal'/f'{doc}__{condition}__formal.json'
            if path.exists():
                r[condition] = w.read(path)
        fp = r.get('FP')
        references.append(dict(document=doc,status='PASS' if fp else 'NOT_RUN_OR_INCOMPLETE',
            primary_spec_sha256=w.sha(D/'FROZEN_SPEC.json'),token_sha256=entry['token_file']['sha256'],
            model_identity=spec['model'],device=spec['execution']['gpu_uuid'],runtime=spec['runtime'],
            sample_positions=spec['primary']['positions'],expected_sample_count=384,actual_sample_count=384 if fp else None,
            vocabulary_dimension=157184,logits_dtype='torch.float32',sampled_logits_content_sha256=fp['sampled_logits_content_sha256'] if fp else None,
            reference_result_sha256=w.sha(D/'formal'/f'{doc}__FP__formal.json') if fp else None,
            artifacts=fp['references'] if fp else [],full_32768_dense_logits_saved=False))
        for condition in (entry['P1_predicted_winner'],entry['L7_predicted_winner']):
            v = r.get(condition)
            results.append(dict(document=doc,rotation=condition,status='PASS' if v else 'NOT_RUN_OR_INCOMPLETE',
                primary_mean_KL=v['primary_mean_KL'] if v else None,sample_count=384 if v else None,
                interval_start=8192,interval_end=32768,vocabulary_dimension=157184,
                all_finite=v['all_finite'] if v else None,
                checkpoint_parity=v['formal_observer_uninterrupted_parity_PASS'] if v else None))
        p1 = r.get(entry['P1_predicted_winner']); l7 = r.get(entry['L7_predicted_winner'])
        complete = p1 is not None and l7 is not None and fp is not None
        delta = p1['primary_mean_KL']-l7['primary_mean_KL'] if complete else None
        winner = ('UNRESOLVED' if abs(delta)<=eps else 'P1' if delta<0 else 'L7') if complete else 'NOT_EVALUATED'
        secondary = []
        if complete:
            for pos in ('8192','16383','24575','32767'):
                ps=p1['state_diagnostics'][pos];ls=l7['state_diagnostics'][pos]
                mse_delta=ps['accumulated_error_mse']-ls['accumulated_error_mse']
                state_winner='TIE' if mse_delta==0 else 'P1' if mse_delta<0 else 'L7'
                secondary.append(dict(position=int(pos),P1_V4_state=ps,L7_V4_state=ls,
                    state_MSE_winner=state_winner,functional_winner=winner,
                    direction_agrees=None if winner=='UNRESOLVED' or state_winner=='TIE' else state_winner==winner))
        documents.append(dict(document=doc,status='COMPLETE' if complete else 'INCOMPLETE_NOT_EVALUATED',
            P1_rotation=entry['P1_predicted_winner'],L7_rotation=entry['L7_predicted_winner'],
            P1_mean_Future_KL=p1['primary_mean_KL'] if p1 else None,
            L7_mean_Future_KL=l7['primary_mean_KL'] if l7 else None,Delta=delta,
            numerical_resolution=eps,prediction_winner=winner,
            V4_state_diagnostic_comparison=secondary,
            quantization_diagnostics={k:r[k]['quantization_diagnostics'] for k in (entry['P1_predicted_winner'],entry['L7_predicted_winner']) if k in r}))
    if all(d['status']=='COMPLETE' for d in documents):
        if any(d['prediction_winner']=='UNRESOLVED' for d in documents): overall='UNRESOLVED_SMALL_PILOT'
        else:
            wins=sum(d['prediction_winner']=='P1' for d in documents)
            overall={2:'DIRECTIONALLY_SUPPORTIVE_SMALL_PILOT',1:'MIXED',0:'P1_NOT_SUPPORTED'}[wins]
    else: overall='INCOMPLETE_NOT_EVALUATED'
    state='STOP' if stop else 'COMPLETE' if overall!='INCOMPLETE_NOT_EVALUATED' else 'IN_PROGRESS'
    for folder,pattern in ((D/'runs','*__parent.json'),(D/'runs','*__resumed_parity.json'),(D/'formal','*__formal.json')):
        for path in sorted(folder.glob(pattern)):
            v=w.read(path)
            jobs.append(dict(path=str(path),sha256=w.sha(path),document=v['document'],condition=v['condition'],
                stage=v.get('stage','formal'),status=v['status'],elapsed_seconds=v['elapsed_seconds'],
                gpu_hours_to_canonical_completion=v['elapsed_seconds']/3600,
                peak_allocated=max(x.get('peak_allocated',0) for x in v.get('memory',[])),
                peak_reserved=max(x.get('peak_reserved',0) for x in v.get('memory',[]))))
    partial=[]
    for folder in (D/'runs',D/'formal'):
        for path in folder.glob('*.STOP.json'):
            v=w.read(path);elapsed=v['stopped_epoch']-v.get('started_epoch',v['stopped_epoch'])
            partial.append(dict(path=str(path),status='STOP',elapsed_seconds=elapsed,gpu_hours=elapsed/3600))
    active=[]
    for folder in (D/'runs',D/'formal'):
        for path in folder.glob('*.STARTED.json'):
            base=path.name.removesuffix('.STARTED.json')
            if (folder/(base+'.json')).exists() or (folder/(base+'.STOP.json')).exists(): continue
            v=w.read(path);start=v.get('started_epoch',v.get('started',time.time()))
            active.append(dict(tag=base,pid=v['pid'],status='IN_PROGRESS',
                elapsed_wall_seconds=time.time()-start,wall_hours_to_snapshot=(time.time()-start)/3600))
    ledger=dict(status=state,updated_epoch=time.time(),single_physical_GPU6=True,
        GPU_uuid=spec['execution']['gpu_uuid'],hardware='24GB RTX3090',jobs=jobs,failed_jobs=partial,
        measured_completed_worker_hours=sum(x['gpu_hours_to_canonical_completion'] for x in jobs),
        failed_worker_hours=sum(x['gpu_hours'] for x in partial),
        active_jobs=active,active_worker_wall_hours_to_snapshot=sum(x['wall_hours_to_snapshot'] for x in active),
        GPU_hours_method='Worker elapsed wall time through canonical completion, including startup/loading; GPU context teardown interval is not included. CPU-only qualification-controller time and waits excluded.',
        peak_allocated_during_recorded_token_execution=max([x['peak_allocated'] for x in jobs]+[0]),
        peak_reserved_during_recorded_token_execution=max([x['peak_reserved'] for x in jobs]+[0]),
        disk_free_bytes=shutil.disk_usage(D).free,
        primary_students_complete=sum(r['status']=='PASS' for r in results),
        FP_references_complete=sum(r['status']=='PASS' for r in references),stop=stop)
    save('RESOURCE_LEDGER.json',ledger)
    save('FP_REFERENCE_MANIFEST.json',dict(status=state,references=references,streaming=True))
    parity_path=D/'CHECKPOINT_FORK_PARITY_COMPLETED.json'
    save('CHECKPOINT_FORK_PARITY.json',w.read(parity_path) if parity_path.exists() else
        dict(status='IN_PROGRESS' if not stop else 'INCOMPLETE_STOP',required_pairs=6,completed_pairs=[
            dict(document=w.read(p)['document'],condition=w.read(p)['condition'],status='LOCAL_PAIR_RESULTS_EXIST')
            for p in (D/'runs').glob('*__resumed_parity.json')]))
    save('DOCUMENT_VERDICT.json',dict(status=state,overall=overall,documents=documents,
        independent_document_count=2,statistical_confirmation=False,no_training_executed=True,
        E_future='NOT_FROZEN_NOT_COMPUTED',Hidden_KL='NOT_FROZEN_NOT_COMPUTED'))
    write_csv('FUNCTIONAL_KL_RESULTS.csv',results,list(results[0]))
    rows=[{k:d[k] for k in ('document','status','P1_rotation','L7_rotation','P1_mean_Future_KL','L7_mean_Future_KL','Delta','numerical_resolution','prediction_winner')} for d in documents]
    write_csv('P1_L7_COMPARISON.csv',rows,list(rows[0]))
    frozen=dict(spec);frozen['secondary']=w.read(D/'SECONDARY_SPEC.json')
    frozen['qualification_primary_spec_sha256']=w.sha(D/'FROZEN_SPEC.json')
    frozen['status']='FROZEN_COMPLETE_DESIGN';save('FROZEN_SPEC.json',frozen)
    shutil.copyfile(D/'EXECUTION_PLAN.md',DELIVERY/'EXECUTION_PLAN.md')
    save('QUALIFICATION_GATES.json',gates)
    report=[f'# P1 与 L7 完整模型 Future-KL 验证\n\n状态：{state}；总体判定：{overall}。',
        '\nPrimary 是冻结的 384 个时间位置、完整 157184 维词表 FP32 KL(FP||student)，sum/384。采样从8192到32704；模型执行到32768。',
        '\n|文档|P1 rotation KL|L7 rotation KL|Delta=P1−L7|胜者|\n|---|---:|---:|---:|---|']
    for d in documents:
        def fmt(v): return '未测得' if v is None else f'{v:.12g}'
        report.append(f"|{d['document']}|{fmt(d['P1_mean_Future_KL'])}|{fmt(d['L7_mean_Future_KL'])}|{fmt(d['Delta'])}|{d['prediction_winner']}|")
    report.append(f'\n预冻结数值分辨率 {eps:.12g}；abs(Delta) 不超过此值时 UNRESOLVED。它是经验算术分辨率，不是统计显著性。')
    if overall=='INCOMPLETE_NOT_EVALUATED':
        report.append('\n1. P1 与 L7 谁更准确：目前不能判断，正式完整 student 轨迹尚未全部测得。\n2. 两篇文档结果见表。\n3. Future-KL 差异尚未全部测得；未测值不填零。\n4. State damage 与 functional 一致性：正式配对完成前不能判断。\n5. P1 是否提供 L7 之外信息：不能从未完成实验判断。\n6. 目前没有本轮证据支持进入 P1 gradient trainability。\n7. 下一步不自动进入 L8；不训练。')
    else:
        supportive=overall=='DIRECTIONALLY_SUPPORTIVE_SMALL_PILOT'
        report.append('\nP1/L7 预测准确性由表中胜者判定。仅两篇文档，不作统计确认或普遍有效主张。')
        report.append('\nState damage 只使用 V4 原有 per-position L2/MSE，在8192、16383、24575、32767比较；不新增时间聚合 proxy。详细数值保存在 DOCUMENT_VERDICT.json。')
        for d in documents:
            report.append('\n'+d['document']+' 的 state MSE 方向：'+', '.join(f"t={s['position']}:{s['state_MSE_winner']}" for s in d['V4_state_diagnostic_comparison'])+'；完整模型 KL 方向：'+d['prediction_winner']+'。')
        report.append('\nP1 对 L7 之外的信息：'+('本次冻结的相反预测配对中，P1 的排序更贴近真实 Future-KL，构成有限的方向性证据；不能证明一般条件增量信息。' if supportive else '本轮不提供一致的、可推广的 P1 增量信息证据。'))
        report.append('\nP1 gradient trainability：'+('值得另行设计可微性与梯度一致性验证；本轮没有训练有效性证据。' if supportive else '先扩展预注册的 functional 验证和机制分析，不由本轮直接进入训练。'))
        report.append('\n下一步不需要自动进入 L8。L8 是独立训练问题，需另行设计；本轮无 optimizer、Cayley、AIME 或自由生成。')
    report.append('\nE_future 和 Hidden KL：NOT_FROZEN_NOT_COMPUTED，依据用户本轮澄清；未新增未来 functional proxy。')
    report.append('\n量化诊断仅记录续跑区间[8192,32768)的原 scale arithmetic、真实 saturation/clipping 和原 J_INJ QDQ 误差；不虚构未保存的前8192 scale历史。V4 未导出独立 layer-wise state-error metric，因此未另造公式。')
    report.append(f"\n资源：completed worker {ledger['measured_completed_worker_hours']:.6f} GPU-context wall-hours（含启动/加载，精确 teardown 不计）；student完成 {ledger['primary_students_complete']}/4，FP reference完成 {ledger['FP_references_complete']}/2。")
    if stop: report.append('\n停止原因：'+stop['reason']+'。门禁失败后没有继续科学运行或更改设计。')
    report.append('\nV4 三缓存预估容量 STOP 原样保留。本轮使用单缓存串行执行，并重新验证新增的 checkpoint 存储／恢复接口；原数值引擎未修改。')
    (DELIVERY/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n')
    names=sorted(p for p in DELIVERY.iterdir() if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS.txt' and not p.name.endswith('.partial'))
    (DELIVERY/'SHA256SUMS.txt').write_text(''.join(f'{w.sha(p)}  {p.name}\n' for p in names))
    print(json.dumps(dict(status=state,overall=overall,students=ledger['primary_students_complete'],FP=ledger['FP_references_complete'],delivery=str(DELIVERY)),ensure_ascii=False))


if __name__=='__main__':
    with (DELIVERY/'.seal.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        main()
