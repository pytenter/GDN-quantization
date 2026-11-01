# -*- coding: utf-8 -*-
"""CPU-only terminal evidence/delivery after real STOP; no repairs or retries."""
import datetime
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE=Path('/data/zypan/worktrees/ling-kda-p1-l7-full-persistent-prospective-v2')
sys.path.insert(0,str(CODE))
from v2_common import OLD,OUT,RUNS,FREEZE,LATCH,read,checked,emit,ledger,digest
from run_matched_l7 import publish

def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,epoch=time.time())

def text(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f: f.write(value)

def localtime(epoch):
    return datetime.datetime.fromtimestamp(epoch,datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and LATCH.exists()
    observed=time.time(); failures=[]
    for p in sorted(RUNS.glob('*.STOP.json')):
        v=checked(p); assert not Path(f"/proc/{v['pid']}").exists(),'WORKER_STILL_ALIVE_NO_FINAL_DELIVERY'
        v=dict(v,stop_path=str(p),stop_sha256=digest(p),log_path=str(p).replace('.STOP.json','.log'),
            log_sha256=digest(str(p).replace('.STOP.json','.log')))
        failures.append(v)
    assert len(failures)==2
    failures.sort(key=lambda v:v['stopped_epoch']); first=failures[0]
    assert first['document']=='pg19-test-860' and first['error']=="AssertionError('MIDRUN_EXCLUSIVITY_FAIL')"
    assert failures[1]['error']=="AssertionError('V2_PEER_STOP_NO_RETRY')"
    assert not list(RUNS.glob('micro*.LAUNCH_AUTHORIZATION.json'))
    assert not list(RUNS.glob('formal*.LAUNCH_AUTHORIZATION.json'))
    assert not list(RUNS.glob('*fresh1.LAUNCH_AUTHORIZATION.json'))
    assert not list(RUNS.glob('qualification*fresh0.json'))
    identity=checked(OUT/'SOURCE_AND_WEIGHT_IDENTITY.json'); checks=[]
    for p,v in identity['actual_identities'].items():
        actual=digest(p); assert actual==v['sha256'],('FINAL_IDENTITY_CHANGED',p)
        checks.append(dict(path=p,sha256=actual,passed=True))
    inventory={str(p):digest(p) for p in sorted(OLD.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    assert inventory==identity['original_inventory'],'OLD_EVIDENCE_DIRECTORY_CHANGED'
    freeze=checked(FREEZE)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==freeze['code_commit']
    for p,h in freeze['code_sha256'].items(): assert digest(p)==h
    hardware={str(g):dict(query=cmd(['nvidia-smi','-i',str(g),'--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
        apps=cmd(['nvidia-smi','-i',str(g),'--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
        holders=cmd(['fuser',f'/dev/nvidia{g}'])) for g in (3,6)}
    for g,h in hardware.items():
        assert h['apps']['returncode']==0 and not h['apps']['stdout'].strip()
        assert h['holders']['returncode']==1 and not h['holders']['stdout'].strip()
    resources=dict(epoch=observed,physical_gpu_snapshot=hardware,
        disks={str(p):dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in (OUT,Path('/tmp'),Path('/dev/shm'))},
        maintenance=dict(reboot_required=Path('/var/run/reboot-required').exists(),scheduled_shutdown=Path('/run/systemd/shutdown/scheduled').exists()),
        supervisor_tmux=cmd(['tmux','has-session','-t','ling_p1l7_v2_supervision']),
        runtime_packages={n:importlib.metadata.version(n) for n in ('torch','fla-core','triton','transformers')})
    emit(OUT/'FINAL_PRESERVATION_AND_RELEASE_AUDIT.json',dict(status='PASS',epoch=time.time(),checks=checks,
        old_non_git_file_count=len(inventory),old_all_bytes_unchanged=True,full_32_weights_rehashed_again=True,
        measurement_code_unchanged=True,resources=resources,gpu_library_imported=False,new_gpu_execution=False))
    account=ledger()
    assert all(x['status']=='STOP' for x in account['jobs']) and len(account['jobs'])==2
    account.update(status='COMPLETE',runner_elapsed_gpu_hours=sum(v['accounted_gpu_hours'] for v in failures),
        cumulative_v1_plus_v2_receipt_device_hours=account['historical_v1_gpu_hours']+account['v2_gpu_hours'],
        release_observation_epoch=observed,release_uncertainty_upper_bound_gpu_hours=sum(max(0,observed-x['launch_epoch'])/3600 for x in account['jobs']),
        release_bound_is_not_extra_measured_work=True,wall_clock_first_launch_to_last_STOP_seconds=max(v['stopped_epoch'] for v in failures)-min(x['launch_epoch'] for x in account['jobs']),
        FIXED_VRAM_PERCENT_CAP=None,no_administrative_budget_stop=True)
    emit(OUT/'GPU_COST_LEDGER.json',account)
    emit(OUT/'GPU_QUALIFICATION_RESULTS.json',dict(status='STOP',classification='ENGINEERING_OR_NUMERICAL_STOP',
        documents=failures,first_failure=first,completed_512x2_qualifications=0,
        fresh_process_second_runs='NOT_RUN',formal_32k_students_completed=0,
        no_numerical_canonical_completion=True,qualification_cannot_be_claimed_PASS=True,
        immediate_no_retry_stop=True))
    runtime=[dict(document=v['document'],gpu_uuid=v['gpu_uuid'],runtime_telemetry=v['runtime_telemetry']) for v in failures]
    assert all(r['deterministic_algorithms_enabled'] and not r['warn_only'] and not r['matmul_tf32'] and not r['cudnn_tf32']
               for v in failures for r in v['runtime_telemetry'])
    emit(OUT/'DETERMINISTIC_RUNTIME_INIT_REPORT.json',dict(status='PASS',scope='actual child initialization only, NOT complete deterministic numerical qualification',
        runtime=runtime,original_assert_passed_on_both_devices=True,strict_enabled_before_CUDA=True,
        helpers_and_model_loading_did_not_reset_flag=True,complete_fresh_process_repeatability='NOT_RUN',
        cpu_regression_sha256=digest(OUT/'CPU_REGRESSION_TEST_RESULTS.json'),worker_sha256=digest(CODE/'v2_worker.py')))
    cause=dict(status='STOP',first_failure_gate='MIDRUN_EXCLUSIVITY_FAIL',first_stop=first,
        secondary_stop=failures[1],first_stop_local_time=localtime(first['stopped_epoch']),
        established='GPU6 device() returned exclusive=False at first 512-token progress callback; GPU3 stopped via peer latch',
        not_established='Actual offending PID and whether compute occupancy, peer CUDA device-handle enumeration, a transient diagnostic holder or another holder caused the failure',
        telemetry_limitation='The local hw snapshot on failing callback was not persisted before assertion; STOP retained runtime and full stack but not compute-apps/fuser lists at failure',
        source_guard='device(..., pid) permits only current worker PID in BOTH compute-apps and fuser; the parallel peer PID is not allowlisted',
        cross_device_peer_holder_is_hypothesis_not_proven=True,
        no_unsupported_op_or_parity_error_recorded=True,not_a_16GPUh_or_85percent_stop=True,
        next_action_proposal_only='Read-only identity-bound diagnosis of fuser/compute-app holders, including legitimate parallel peer/diagnostic PIDs; any repair and new qualification requires a new explicit isolated scope; do not clear this V2 latch',
        scientific_interpretation='No P1/L7 long-horizon comparison is available; engineering STOP is not scientific evidence against either metric')
    emit(OUT/'FIRST_FAILURE_GATE_REPORT_V2.json',cause)
    for name,meaning in [('SAME_GPU_FP_PARITY_REPORT.json','512-token work entered, but no completed same-GPU FP/full-logit reference hash/canonical was published'),
                         ('PERSISTENT_QDQ_SEMANTICS_REPORT.json','unchanged persistent engine ran; no completed counter/proof canonical published; cannot claim qualified PASS')]:
        emit(OUT/name,dict(status='STOP',stage='qualification',reason=meaning,first_failure_gate='MIDRUN_EXCLUSIVITY_FAIL',
            document_evidence=[dict(document=v['document'],stop_sha256=v['stop_sha256'],log_sha256=v['log_sha256']) for v in failures],
            physical_resident_INT8_storage_claim=False,carrier='FP32 dequantized INT8 semantics',formal_32k_status='NOT_RUN'))
    emit(OUT/'MICROBENCHMARK_RESULTS.json',dict(status='NOT_RUN',reason='Both complete512x2 GPU qualifications did not PASS',
        executed_8448_documents=0,speed=None,peak_vram=None,cache_growth=None,prospective_KL_ranking_accessed=False))
    emit(OUT/'RESOURCE_FEASIBILITY_REPORT.json',dict(status='NOT_RUN',reason='No qualified8448 micro available to predict32K resources',
        GPU_HOURS_HARD_CAP=None,FIXED_VRAM_PERCENT_CAP=None,actual_resource_snapshot=resources,
        physical_32k_capacity_not_established=True,not_a_physical_OOM_result=True))
    emit(OUT/'FROZEN_PREMEASUREMENT_SPEC.json',dict(status='NOT_RUN',reason='Qualification failed before formal premeasurement freeze',
        reused_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
        formal_launched=False,scientific_near_zero_rule_unresolved=True,
        near_zero_rule_not_invented_or_applied=True))
    formal_names=['FP_REFERENCE_MANIFEST.json','FORMAL_32K_RUN_MANIFEST.json','PERSISTENT_FUTURE_KL_RESULTS.csv',
        'PAIRED_ROTATION_COMPARISON.csv','DOCUMENT_LEVEL_VERDICT.json','FINAL_SCIENTIFIC_REPORT.md']
    assert all(not (OUT/n).exists() for n in formal_names)
    manifest=dict(status='STOP',decision='ENGINEERING_OR_NUMERICAL_STOP',task=OUT.name,code_commit=freeze['code_commit'],
        stage_status={'Stage1':'PASS','Stage2':'STOP','Stage3':'NOT_RUN','Stage4':'NOT_RUN','Stage5':'NOT_RUN'},
        original_STOPs_preserved=True,formal_32k_completion='0/4',full_length_FP_completion='0/2',
        prohibited_fake_formal_outputs_absent=formal_names,first_failure_report_sha256=digest(OUT/'FIRST_FAILURE_GATE_REPORT_V2.json'),
        scientific_verdict='NOT_EVALUATED_NO_QUALIFIED_32K_OUTCOME',new_monitor_required=False,
        new_training_AIME_free_generation_executed=False)
    emit(OUT/'DELIVERABLE_MANIFEST.json',manifest)
    report=f'''# Ling/KDA P1 vs 原L7 V2：真实停止报告

最终状态：`ENGINEERING_OR_NUMERICAL_STOP`。原 deterministic 初始化错误已在两张实际GPU child中修复，但本轮512×2完整资格未通过，未启动8448 micro或32K。本STOP不是16GPUh或85%显存行政停止，也不是P1或L7科研FAIL。

## 首故障和证据边界

GPU6/860于{localtime(first['stopped_epoch'])}触发 `MIDRUN_EXCLUSIVITY_FAIL`。堆栈明确位于原引擎首个512-token progress回调，V2 worker第60行。GPU3/5956于{localtime(failures[1]['stopped_epoch'])}触发 `V2_PEER_STOP_NO_RETRY`，是联动保护，不是另一项独立数值失败。两进程和监督进程均自然退出；未发送signal/kill，未换GPU，未自动重试。

运行中独占检查同时要求nvidia compute-apps与fuser holder只包含当前worker PID，未允许另一授权peer PID。**失败时hw局部快照没有写入STOP**，因此无法确认是实际外部CUDA占用、peer的跨设备fd、瞬时诊断holder或其他holder。并行门禁误判只是静态代码支持的候选解释，不能写成已证明根因。停止后的两指定卡无compute-app/holder，仅15MiB背景占用，不足以倒推出失败瞬间情况。详见 `FIRST_FAILURE_GATE_REPORT_V2.json` 和原两STOP/log。

## 按原指令回答15项

1. **初始化真正修复：是（仅初始化层面）**。两实际child在CUDA初始化前enabled=True/warn_only=False，继承helpers及Capture加载后仍True，原assert通过。完整fresh-process numerical repeatability未获得PASS。
2. **原STOP保留：是**。起始950项检查匹配；32权重全哈希；最终再次重哈希全部身份，旧目录335非git/nonpycache文件byte-identical，Attempt04/05和V1 latch/STOP未更改。
3. **frozen pair严格复用：是**。860/GPU6=H_to_L6_a50 vs H_to_L7_a25，5956/GPU3=H_to_L6_a50 vs H_to_L7_a50；P1均预测前者、原L7_TOTAL均预测后者。未重做PhaseA/B、未新selection。
4. **GPU6/GPU3 qualification：STOP/STOP，不能称PASS**。首轮各已实际启动，第二个fresh process未启动，没有完整512×2 canonical。
5. **persistent INT8 QDQ/writeback：未获得完整资格PASS**。数值engine SHA原样，512短程执行已进入；无法用控制流到达某处替代已发布counter/proof。状态载体为包含QDQ值的FP32，不声称物理resident INT8。
6. **8448 micro速度/显存/cache增长：NOT_RUN，无可报告测量**。停止前GPU占用快照约15891MiB不是已核验peak，也不是8448结果。
7. **4/4正式32K：未完成，0/4，未启动**。
8. **同GPU完整FP reference：0/2，NOT_RUN**。512短程没有完成canonical/full-logit hash，不能复用为正式FP。
9. **860 P1/L7 Future-KL：均未测得**。
10. **5956 P1/L7 Future-KL：均未测得**。
11. **两篇支持谁：无法判定，不计为胜利、失败或tie**。
12. **P1增量：NOT_EVALUATED**。不能把工程STOP当NOT_SUPPORTED科研结论。
13. **进入梯度实验：本轮没有新增支持证据，不执行**。仅可单独提案；未训练/optimizer.step/new rotation/L9/AIME/free generation。
14. **GPU成本**：worker start→STOP累计{account['runner_elapsed_gpu_hours']:.12f}GPUh；含launch receipt/startup成本{account['v2_gpu_hours']:.12f}GPUh，墙钟{account['wall_clock_first_launch_to_last_STOP_seconds']:.3f}秒。历史V1={account['historical_v1_gpu_hours']:.12f}GPUh；累计receipt device-time={account['cumulative_v1_plus_v2_receipt_device_hours']:.12f}GPUh。释放时刻未连续采样，独立保守release上界见ledger，不当额外已测工作。没有人为GPUh硬cap。
15. **未解决限制**：运行中holder首因未完整保存；完整fresh-process资格、8448、物理32K容量、正式FP/4条32K均未完成；旧P1/L7 frozen资产没有可定位的near-zero Future-KL resolution阈值，未擅自补定。正式controller尚未资格，现有controller明确拒绝formal。后续需先独立诊断、明确授权与预冻结，不清V2锁续跑。

## 隔离、代码与交付

测量commit `{freeze['code_commit']}`；原数值engine SHA `{digest(CODE/'persistent_engine_v1.py')}`。变更限于child deterministic初始化/telemetry、新结果namespace、资源controller与CPU regression；行政16GPUh/85% gate在V2移除，旧V1 unchanged。CPU tests是CPU证据，不能替代GPU资格。初次git author未配置问题保留在PRE_GPU_ENGINEERING_APPENDIX_001.json，未因此重复任何GPU测量。

远端结果：`{OUT}`

独立clean worktree：`{CODE}`（原f1fa基线，追加V2 commit）

本地交付：`C:/Users/admin/Desktop/操作系统实验/lab2/ling_p1_l7_persistent_v2/results`

13项阶段报告均以真实PASS/STOP/NOT_RUN标识；未生成正式32K的6个结果文件，避免以空表冒充测量。原始STOP/log/receipts/sidecar、冻结JSON、最终复核及全部交付哈希附在目录中。本轮停止，不自动开展新训练或重跑。
'''
    text(OUT/'FINAL_STOP_REPORT.md',report)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt'))
    hashes={str(p.relative_to(OUT)):digest(p) for p in paths}
    publish(OUT/'RESULT_AND_DELIVERY_HASHES.json',dict(schema='all_V2_delivery_relative_path_SHA256',files=hashes,
        code_commit=freeze['code_commit'],manifest_self_excluded=True,final_state='ENGINEERING_OR_NUMERICAL_STOP'))
    hashes['RESULT_AND_DELIVERY_HASHES.json']=digest(OUT/'RESULT_AND_DELIVERY_HASHES.json')
    text(OUT/'SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())))
    print(json.dumps(dict(status='V2_STOP_DELIVERY_COMPLETE',first_failure=first['error'],v2_gpu_hours=account['v2_gpu_hours'],
        runner_gpu_hours=account['runner_elapsed_gpu_hours'],files=len(hashes),report_sha256=digest(OUT/'FINAL_STOP_REPORT.md'))))

if __name__=='__main__': main()
