"""Classify only after all10 formal units; seal complete or truthful STOP evidence."""
import csv,json,math,os,subprocess,sys,time
from pathlib import Path
import pilot_engineering as P
import formal_worker as W

def read(path):return json.loads(Path(path).read_text())
def status(name):
    path=P.ROOT/name
    return read(path).get('status','UNKNOWN') if path.exists() else 'NOT_RUN'

def combine(name,files,transform=None):
    rows=[]
    for path in files:
        with Path(path).open(newline='') as f:
            for row in csv.DictReader(f):rows.append(transform(row) if transform else row)
    if rows:return W.write_csv(name,rows)

def main():
    P.check(not (P.ROOT/'RESULT_AND_DELIVERY_HASHES.json').exists(),'already sealed; do not rewrite verdict')
    branches={};references={};anchors={}
    for doc in [1,2]:
        path=P.ROOT/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json'
        if path.exists():references[doc]=read(path)
        path=P.ROOT/f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json'
        if path.exists():anchors[doc]=read(path)
        for arm in W.ARMS:
            path=P.ROOT/f'students/doc_{doc:02d}/{arm}/BRANCH_MANIFEST.json'
            if path.exists():branches[(doc,arm)]=read(path)
    complete=len(references)==2 and len(anchors)==2 and len(branches)==6
    failure_path=P.ROOT/'FIRST_FAILURE_GATE.json';failure=read(failure_path) if failure_path.exists() else None
    spec_path=P.ROOT/'FROZEN_SCIENCE_SPEC.json';spec=read(spec_path) if spec_path.exists() else None
    # Full retained sampled logits permit a second independent mathematical
    # audit. Its numerical threshold is frozen before any formal outcome.
    audit=[]
    if complete and not failure:
        import torch
        P.torch=torch;torch.set_num_threads(1)
        for doc in [1,2]:
            fp=torch.load(references[doc]['logits']['path'],map_location='cpu',weights_only=True)['logits']
            for arm in W.ARMS:
                b=branches[(doc,arm)];P.check(P.sha(b['sampled_logits']['path'])==b['sampled_logits']['sha256'],'retained student sampled logits SHA')
                student=torch.load(b['sampled_logits']['path'],map_location='cpu',weights_only=True)['logits']
                path=P.ROOT/f'students/doc_{doc:02d}/{arm}/SAMPLED_FULLVOCAB_FUTURE_KL.csv'
                with path.open() as f:rows=list(csv.DictReader(f))
                errors=[abs(float(rows[i]['KL_FP32'])-P.kl64(fp[i],student[i])) for i in range(384)]
                audit.append({'document':doc,'condition':arm,'max_abs_KL32_vs_independent_KL64_error':max(errors)})
        e=read(P.ROOT/'KL_REDUCER_QUALIFICATION.json')['FP32_max_abs_error']
        audit_pass=all(r['max_abs_KL32_vs_independent_KL64_error']<=e for r in audit)
        P.save('FORMAL_RETAINED_LOGIT_KL_NUMERICAL_AUDIT.json',{'status':'PASS' if audit_pass else 'FAIL_AND_STOP',
            'frozen_independent_error_bound':e,'rows':audit,'no_threshold_change':True})
        if not audit_pass:
            P.save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP','stage':'FORMAL_KL_NUMERICAL_AUDIT',
                'utc':P.N.now(),'message':'formal KL arithmetic exceeds independently frozen qualification error bound',
                'formal_outcomes_preserved':True,'no_retry':True})
            failure=read(failure_path)
    if references:P.save('FP_REFERENCE_MANIFEST.json',{'status':'COMPLETE' if len(references)==2 else 'INCOMPLETE',
        'documents':references,'frozen_spec_SHA':P.sha(spec_path) if spec_path.exists() else None})
    if branches:P.save('STUDENT_BRANCH_MANIFEST.json',{'status':'COMPLETE' if len(branches)==6 else 'INCOMPLETE',
        'branches':[dict(document_index=doc,condition=arm,**row) for (doc,arm),row in sorted(branches.items())],
        'common_anchors':anchors,'expected_complete_branches':6})
    sample_files=[P.ROOT/f'students/doc_{doc:02d}/{arm}/SAMPLED_FULLVOCAB_FUTURE_KL.csv' for doc,arm in sorted(branches)]
    activation_files=[P.ROOT/f'students/doc_{doc:02d}/{arm}/SCALE_INTERVENTION_ACTIVATION.csv' for doc,arm in sorted(branches)]
    state_files=[P.ROOT/f'students/doc_{doc:02d}/{arm}/STATE_DAMAGE_ENDPOINTS.csv' for doc,arm in sorted(branches)]
    combine('SAMPLED_FULLVOCAB_FUTURE_KL.csv',sample_files)
    combine('SCALE_INTERVENTION_ACTIVATION.csv',activation_files)
    combine('CLIPPING_AND_SATURATION.csv',activation_files,lambda r:{k:v for k,v in r.items() if k not in
        ['inflation_groups','deflation_groups','activated_groups','signed_log2_sum','abs_effective_log2_offset_sum',
         'activation_fraction','mean_signed_log2_scale_ratio','mean_abs_effective_log2_offset']})
    combine('STATE_DAMAGE_ENDPOINTS.csv',state_files)
    doc_rows=[];weak=False;directions=[]
    for doc in [1,2]:
        if not all((doc,arm) in branches for arm in W.ARMS):continue
        parent,cap,floor=[branches[(doc,arm)] for arm in W.ARMS]
        delta=parent['K']-cap['K'];secondary=parent['K']-floor['K']
        fraction=cap['activation_groups']/cap['total_groups'];docweak=fraction<spec['weak_activation_fraction'];weak |= docweak
        direction='NOT_CLASSIFIED_INCOMPLETE'
        if complete and not failure:
            direction='POSITIVE' if delta>spec['DELTA_resolution'] else ('NEGATIVE' if delta<-spec['DELTA_resolution'] else 'UNRESOLVED')
            directions.append(direction)
        doc_rows.append(dict(parent['identity'],document_index=doc,K_PARENT=parent['K'],K_NO_SCALE_INFLATION=cap['K'],
            K_NO_SCALE_DEFLATION=floor['K'],DELTA_PRIMARY=delta,DELTA_NO_DEFLATION=secondary,direction=direction,
            DELTA_resolution=spec['DELTA_resolution'],NO_INFLATION_activated_groups=cap['activation_groups'],
            NO_INFLATION_activation_fraction=fraction,INTERVENTION_WEAKLY_ACTIVATED=docweak,
            clipping_PARENT=parent['clipped_values']/parent['state_values'],clipping_NO_INFLATION=cap['clipped_values']/cap['state_values'],
            clipping_NO_DEFLATION=floor['clipped_values']/floor['state_values']))
    if doc_rows:W.write_csv('DOCUMENT_LEVEL_CAUSAL_EFFECT.csv',doc_rows)
    verdict='INCOMPLETE_OR_ENGINEERING_STOP'
    if complete and not failure:
        if weak or 'UNRESOLVED' in directions:verdict='UNRESOLVED_OR_WEAKLY_ACTIVATED'
        elif directions==['POSITIVE','POSITIVE']:verdict='FUNCTIONAL_EFFECT_DIRECTIONALLY_SUPPORTIVE_SMALL_PILOT'
        elif directions==['NEGATIVE','NEGATIVE']:verdict='NO_INFLATION_INTERVENTION_WORSENED_KL_IN_PILOT'
        else:verdict='MIXED_FUNCTIONAL_EFFECT'
    engineering_pass=spec is not None
    statuses={'ENGINEERING_STATUS':'PASS' if engineering_pass else 'INCOMPLETE_OR_STOP',
        'FULLMODEL_NUMERICAL_STATUS':status('FULLMODEL_NUMERICAL_QUALIFICATION.json'),
        'INTERVENTION_VALIDITY_STATUS':status('QUANTIZER_INTERVENTION_VALIDITY.json'),
        'PHYSICAL_RESOURCE_STATUS':('PASS_FULL32768_CONFIRMED' if complete else ('PASS_PROJECTED_NOT_FULLY_EXECUTED' if status('RESOURCE_AND_STORAGE_READINESS.json')=='PASS' else 'NOT_FULLY_QUALIFIED')),
        'FORMAL_SCIENTIFIC_STATUS':'COMPLETE' if complete and not failure else 'INCOMPLETE_OR_STOP',
        'FUNCTIONAL_CAUSAL_VERDICT':verdict}
    if failure and failure.get('stage')=='FORMAL_KL_NUMERICAL_AUDIT':statuses['FULLMODEL_NUMERICAL_STATUS']='FAIL_AND_STOP'
    P.save('SCIENTIFIC_VERDICT.json',{'utc':P.N.now(),**statuses,'document_results':doc_rows,
        'completed_FP_references':len(references),'completed_common_warmups':len(anchors),'completed_student_continuations':len(branches),
        'first_failure':failure,'measured_effect':'Oracle Scale Intervention Total Causal Effect on full-model Future-KL under persistentINT8',
        'statistical_confirmation':False,'deployment_method_validation':False,'rotation_ranking_claim':False,
        'AIME_or_generation_or_training_executed':False,'scope':'directional2-doc small pilot only'})
    receipts=[]
    paths=list((P.ROOT/'qualification').glob('small_*.json'))+list((P.ROOT/'qualification').glob('observer_reducer_*.json'))
    paths += [P.ROOT/'qualification/ANCHOR_PRODUCTION_RECEIPT.json',P.ROOT/'ANCHOR_CHECKPOINT_AND_FORK_PARITY.json']
    paths += list((P.ROOT/'references').glob('doc_*/FP_REFERENCE_MANIFEST.json'))+list((P.ROOT/'anchors').glob('doc_*/ANCHOR_MANIFEST.json'))+list((P.ROOT/'students').glob('doc_*/*/BRANCH_MANIFEST.json'))
    for path in paths:
        if not path.exists():continue
        r=read(path)
        if 'GPUh' in r:receipts.append({'path':str(path),'sha256':P.sha(path),'GPUh':r['GPUh'],
            'seconds':r['seconds'],'peak_allocated':r.get('peak_allocated',0),'peak_reserved':r.get('peak_reserved',0)})
    if failure and 'unit_GPUh_including_initialization' in failure:receipts.append({'path':str(failure_path),
        'GPUh':failure['unit_GPUh_including_initialization'],'seconds':failure['unit_wall_seconds'],
        'peak_allocated':failure.get('peak_allocated',0),'peak_reserved':failure.get('peak_reserved',0)})
    data_bytes=sum(p.stat().st_size for p in P.ROOT.rglob('*') if p.is_file())
    ledger={'utc':P.N.now(),'GPU_UUID':P.S.UUIDS[0],'units':receipts,'actual_GPUh':math.fsum(r['GPUh'] for r in receipts),
        'GPUh_definition':'one occupied GPU x unit wall time including model loading; excludes CPU-only audit and coordinator idle gaps',
        'peak_allocated_bytes':max([r['peak_allocated'] for r in receipts]+[0]),
        'peak_reserved_bytes':max([r['peak_reserved'] for r in receipts]+[0]),'retained_apparent_file_bytes':data_bytes,
        'all_models_serial_on_same_GPU':True,'fixed_GPUh_limit':None}
    P.save('GPU_COST_LEDGER.json',ledger)
    documents=read(P.ROOT/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json')['documents'] if (P.ROOT/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json').exists() else []
    line=['本轮结果：'+verdict,'','```json',json.dumps(statuses,ensure_ascii=False,indent=2),'```','',
        '测量范围：同一8192-token persistent INT8共同历史分叉后的完整 closed-loop oracle scale intervention 对未来完整词表KL的总效应。工程样本不计入正式两篇。','',
        '1. V3数值闭环复用：'+status('V3_NUMERICAL_CLOSURE_REUSE.json')+'；历史 native/fixed/初次V3失败全部保留。',
        '2. 本次closed-loop新资格：'+statuses['FULLMODEL_NUMERICAL_STATUS']+'。',
        '3. clean FP scale严格配对：'+status('FP_SCALE_COORDINATE_PARITY.json')+'，同C6R/当前token/layer/group，FP不写回量化。',
        '4. 三学生同一8192 anchor：已完成正式共同anchor数='+str(len(anchors))+'；每doc所有分支绑定其同一SHA。',
        '5. Parent fork与连续Parent：'+status('ANCHOR_CHECKPOINT_AND_FORK_PARITY.json')+'；工程8192后8token逐字节比对。',
        '6. Observer/scale数据流：'+('两个新独立进程完整Parent OFF/ON轨迹逐字节相同。' if engineering_pass else '尚未全部获资格。'),
        '7. 24GB完整32768：'+statuses['PHYSICAL_RESOURCE_STATUS']+'；不把投影或工程8K当作32K完成。',
        '8. 正式新文档及SHA：'+json.dumps(documents,ensure_ascii=False),
        '9. NO_SCALE_INFLATION真实触发：'+json.dumps({r['document_index']:r['NO_INFLATION_activated_groups'] for r in doc_rows})+' 个token/layer/group事件；未完成文档不补数。',
        '10. Clipping：'+json.dumps([{k:r[k] for k in ['document_index','clipping_PARENT','clipping_NO_INFLATION','clipping_NO_DEFLATION']} for r in doc_rows])+'；cap引入的范围溢出单独记录，code饱和见CSV。',
        '11. Parent Future-KL：'+json.dumps({r['document_index']:r['K_PARENT'] for r in doc_rows})+'。',
        '12. NoInflation Future-KL：'+json.dumps({r['document_index']:r['K_NO_SCALE_INFLATION'] for r in doc_rows})+'。',
        '13. NoDeflation Future-KL：'+json.dumps({r['document_index']:r['K_NO_SCALE_DEFLATION'] for r in doc_rows})+'。',
        '14. Primary Delta：'+json.dumps({r['document_index']:r['DELTA_PRIMARY'] for r in doc_rows})+'；预冻结分辨率='+str(spec['DELTA_resolution'] if spec else '尚未冻结')+'。',
        '15. 功能因果方向：'+verdict+'。2/2为正也只能称方向性small pilot，不能统计确认或推广。',
        '16. 是否仅状态改善而KL未改善：'+('须对照稀疏终点STATE_DAMAGE_ENDPOINTS与时间平均KL，不能用状态误差替代功能证据。' if complete else '正式面板未完成，不能判断。'),
        '17. 实际GPUh='+str(ledger['actual_GPUh'])+'；峰值allocatedGiB='+str(ledger['peak_allocated_bytes']/1024**3)+'；reservedGiB='+str(ledger['peak_reserved_bytes']/1024**3)+'；保留文件GiB='+str(data_bytes/1024**3)+'。',
        '18. 时空敏感度Loss：'+('结果可作为进一步独立验证机制和时空敏感度的依据；本轮未检验梯度可训练性，不能直接启动或认可新Loss。' if complete and not failure else '当前不据未完成结果启动新Loss研究；先处理真实STOP或完成本轮。'),'',
        '未运行C8训练、rotation优化、AIME、自由生成、S1/S2梯度训练或新的surrogate搜索。Oracle clean scale诊断干预不等于可部署方法。']
    if failure:line += ['','首个失败门禁：',json.dumps(failure,ensure_ascii=False,indent=2)]
    with (P.ROOT/'FINAL_REPORT.md').open('x') as f:f.write('\n'.join(line)+'\n')
    # Scientific/artifact files are write-once. Active operational log/event
    # files are intentionally excluded because coordinator closes after sealing.
    files=[p for p in sorted(P.ROOT.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['logs','__pycache__']) and p.name not in
        ['ACTIVE_COORDINATOR_LOCK','COORDINATOR_EVENTS.jsonl','SHA256SUMS.txt','RESULT_AND_DELIVERY_HASHES.json']]
    manifest=[{'path':str(p.relative_to(P.ROOT)),'bytes':p.stat().st_size,'sha256':P.sha(p)} for p in files]
    P.save('RESULT_AND_DELIVERY_HASHES.json',{'utc':P.N.now(),'status':'SEALED_COMPLETE' if complete and not failure else 'SEALED_STOP',
        'files':manifest,'live_operational_logs_excluded':True,'historical_files_modified':0})
    with (P.ROOT/'SHA256SUMS.txt').open('x') as f:
        for row in manifest:f.write(row['sha256']+'  '+row['path']+'\n')
        f.write(P.sha(P.ROOT/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n')
    print('FINAL_RESULT_SEALED',verdict,flush=True)

if __name__=='__main__':main()
