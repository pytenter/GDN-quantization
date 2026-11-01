"""Correct derived delivery labels after an early runtime-environment STOP."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

R=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1');D=R/'execution_03'
W=Path('/data/zypan/worktrees/ling-kda-prospective-qualified-completion');F=R/'delivery_cpu_revisions/execution_03_runtime_environment_fix'
os.environ['LING_PROSPECTIVE_RESULTS']=str(D);os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(W))
from recovery_state import sha,atomic_json
from gpu_admission import proc_identity
def read(p):return json.loads(Path(p).read_text())

old=read(D/'RESULT_AND_DELIVERY_HASHES.json')
for n,v in old['files'].items():assert sha(D/n)==v['sha256'],n
stop_sha=sha(D/'STOP.json');failure=read(D/'STOP.json');assert failure['no_retry'] and 'PYTHONHASHSEED' in failure['traceback']
assert not list(D.glob('runs/*.NATURAL_EXIT.json')) and len(list(D.glob('runs/*.FAILED_EXIT.json')))==2
for p in D.glob('runs/*.LAUNCH.json'):
    ident=read(p)['worker']
    try:actual=proc_identity(ident['pid'])
    except (OSError,ValueError):actual=None
    assert actual!=ident
assert subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True)==''
assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()=='55509063131e88f5040fbdf85452eef3145e41eb'
fix=read(F/'ENVIRONMENT_FIX_CPU_TEST_REPORT.json');assert fix['status']=='PASS_ENVIRONMENT_FIX_CPU_ONLY' and not fix['CUDA_exposed']
archive=R/'delivery_cpu_revisions/execution_03_revision_00_original_seal.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',dereference=True) as t:
    for n in list(old['files'])+['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:t.add(D/n,arcname=n,recursive=False)
A=D/'cpu_delivery_audit';(A/'original').mkdir(parents=True)
changed=['RESOURCE_REPORT.json','QUALIFICATION_MATRIX.json','FINAL_STATUS.json','FINAL_MACHINE_STATUS.json','FINAL_SCIENTIFIC_REPORT.md','RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']
for n in changed:shutil.copyfile(D/n,A/'original'/n)
for p in F.iterdir():
    if p.is_file():shutil.copyfile(p,A/p.name)

diagnosis=dict(status='RUNTIME_ENVIRONMENT_INVALID_STOP',epoch=time.time(),first_failure_sha256=stop_sha,cause='CPU launcher omitted PYTHONHASHSEED=0 from the environment passed to the fresh Python interpreter',source_evidence=dict(launcher_path=str(D/'launch_qualified_execution.py'),launcher_sha256=sha(D/'launch_qualified_execution.py'),frozen_runtime_path=str(W/'v2_runtime.py'),frozen_runtime_sha256=sha(W/'v2_runtime.py'),assertion="assert os.environ.get('PYTHONHASHSEED')==str(seed)"),physical_GPU2_GPU3_preflight='PASS_13_OF_13_EACH',per_worker_admission='PASS_BOTH',failed_worker_count=2,model_loading_reached=False,token_science_loop_reached=False,CUDA_initialization='NO_INITIALIZATION_PATH_REACHED; NO_RUNTIME_SNAPSHOT_RECORDED',new_numerical_qualification=0,new_formal_trajectories=0,retry_performed=False,first_failure_preserved=True,source_worktree_modified_after_launch=False,environment_fix_candidate='CPU_ONLY_VALIDATED_NOT_APPLIED_TO_STOPPED_EXECUTION',fix_test_sha256=sha(F/'ENVIRONMENT_FIX_CPU_TEST_REPORT.json'))
atomic_json(D/'RUNTIME_ENVIRONMENT_FAILURE_DIAGNOSIS.json',diagnosis)
resource=read(D/'RESOURCE_REPORT.json');resource['status']='RUNTIME_ENVIRONMENT_GATE_STOP'
resource['actual_GPU_kernel_inference']='NOT_STARTED';resource['new_CUDA_peak_status']='NOT_MEASURED_FAILED_BEFORE_RUNTIME_INITIALIZATION';resource['peak_CUDA_allocated_new_execution_03']=None;resource['peak_CUDA_reserved_new_execution_03']=None
resource['GPU_hours_new_execution_03_scope']='2 worker launch-to-exit wall intervals, including CPU import/initialization; no actual model or token inference occurred; do not interpret these as utilization-weighted CUDA compute hours'
resource['GPU_hours_new_execution_03_seconds']=sum(x['wall_seconds'] for x in resource['failed_exit_workers'])
atomic_json(D/'RESOURCE_REPORT.json',resource,replace=True)
matrix=read(D/'QUALIFICATION_MATRIX.json')
for row in matrix['units']:
    key=row['document']+'__'+row['rotation']
    for stage,field in [('parent','parent_status'),('resumed_parity','target_fresh64_status')]:
        p=D/'runs'/(key+'__'+stage+'.FAILED_EXIT.json')
        if p.exists():
            row[field]='FAILED_PRE_MODEL_RUNTIME_ENVIRONMENT_GATE';row['qualification_status']='NOT_QUALIFIED_RUNTIME_ENVIRONMENT_GATE';row['failed_exit_path']=str(p);row['failed_exit_sha256']=sha(p)
matrix['status']='STOP_0_OF_12_TARGET_DEVICE_QUALIFIED';matrix['attempted_initialization_failures']=2
atomic_json(D/'QUALIFICATION_MATRIX.json',matrix,replace=True)
for n in ['FINAL_STATUS.json','FINAL_MACHINE_STATUS.json']:
    s=read(D/n);s.update(ENGINEERING_COMPLETE=False,ENGINEERING_FAILURE='MISSING_PYTHONHASHSEED_IN_CPU_LAUNCHER',NUMERICAL_QUALIFICATION_COMPLETE=False,FORMAL_SCIENCE_COMPLETE=False,FINAL_EXECUTION_STATUS='RUNTIME_ENVIRONMENT_GATE_STOP',QUALIFIED_CONDITIONS=0,FORMAL_FP_COMPLETE=0,FORMAL_STUDENT_COMPLETE=0,ACTUAL_VALID_DOCUMENTS=0,ENVIRONMENT_FIX_CANDIDATE='CPU_VALIDATED_NOT_RESTARTED')
    atomic_json(D/n,s,replace=True)
state=read(D/'CURRENT_STATE.json');state.update(state='STOP_DELIVERY_COMPLETE',engineering_complete=False,first_failure_sha256=stop_sha,reason='MISSING_PYTHONHASHSEED_IN_CPU_LAUNCHER')
atomic_json(D/'CURRENT_STATE.json',state,replace=True)

report=(D/'FINAL_SCIENTIFIC_REPORT.md').read_text()
report=report.replace('ENGINEERING_COMPLETE=True','ENGINEERING_COMPLETE=False').replace('"ENGINEERING_COMPLETE": true','"ENGINEERING_COMPLETE": false').replace('FAIL_CLOSED_STOP','RUNTIME_ENVIRONMENT_GATE_STOP')
report=report.replace('本轮CUDA allocated峰值 0 bytes，reserved 0 bytes','本轮CUDA峰值未记录（在运行环境初始化前退出），不能以缺失测量冒充已测0 bytes')
correction=f'''## 本次启动失败：已核实原因与真实状态

本次未能进入模型加载或GPU逐token科学计算。启动器遗漏了冻结runtime要求的 `PYTHONHASHSEED=0`，两个worker均在 `v2_runtime.initialize()` 第9行环境断言处退出。GPU2/3物理准入均PASS，故障来自启动配置；不是OOM、GPU占用或KL/量化数值失败。首失败STOP SHA `{stop_sha}` 原样保留，`no_retry=true`，没有覆盖STOP或重跑。

37项CPU synthetic测试未覆盖真实启动器到冻结initializer的完整环境传递，不能代替GPU资格。修复候选现已补齐三个冻结环境变量并通过真实冻结initializer的CPU子进程正例、重复解释器hash一致性和缺失seed反例检查，CUDA_VISIBLE_DEVICES为空、无模型推理。这只是下次启动的工程修复证据，未应用为本轮GPU重试，未生成新的科学PASS。

ENGINEERING_COMPLETE=false；NUMERICAL_QUALIFICATION_COMPLETE=false；FORMAL_SCIENCE_COMPLETE=false。目标设备资格0/12、FP0/4、student0/8、valid0/4。历史GPU6完整8/12和9个parent/checkpoint保持原身份。两次实际初始化尝试在资格矩阵中标为FAILED_PRE_MODEL_RUNTIME_ENVIRONMENT_GATE；其他尚未启动项为NOT_RUN。没有Primary KL/Delta、没有新文档胜负，也没有空科学CSV。

新增资源按冻结worker launch-to-exit合同为 `{resource['GPU_hours_new_execution_03']:.17g}` GPUh，即 `{resource['GPU_hours_new_execution_03_seconds']:.17g}` 秒，两条均为启动到退出的墙钟，含CUDA前CPU导入/初始化；不能理解为GPU kernel计算时间。新CUDA allocated/reserved峰值没有runtime测量，标UNKNOWN/未测；保留历史已测峰值及其scope。两worker和tmux均已退出。下一次若获用户明确重启指令，应使用新的独立execution，保留execution03失败证据并重做资源准入和受影响资格，不能在本执行静默重试。

本修订仅更正CPU交付中的工程完成标签、已尝试失败状态及未测峰值表达，未改任何科学输入、checkpoint、资格阈值、数学、worker记录或原STOP。原自动封存与差异SHA均可核验。

'''
position=report.index('## A.');report=report[:position]+correction+report[position:]
report=report.replace('父阶段不冒充完整资格','父阶段不冒充完整资格')
(D/'FINAL_SCIENTIFIC_REPORT.md').write_text(report)
derived=set(changed)|{'CURRENT_STATE.json'}
preserved={n:v['sha256'] for n,v in old['files'].items() if n not in derived}
for n,h in preserved.items():assert sha(D/n)==h,n
assert sha(D/'STOP.json')==stop_sha
atomic_json(D/'CPU_DELIVERY_CORRECTION.json',dict(status='DERIVED_LABELS_CORRECTED_NO_SCIENCE_RETRY',epoch=time.time(),original_manifest_sha256=sha(A/'original/RESULT_AND_DELIVERY_HASHES.json'),original_seal_archive=str(archive),original_seal_archive_sha256=sha(archive),first_failure_sha256=stop_sha,unchanged_original_files=preserved,reason='Generated engineering_complete=true and NOT_RUN/zero peak defaults did not distinguish failed pre-model runtime initialization',changed_derived_files=changed+['CURRENT_STATE.json'],CUDA_work_performed_by_correction=False,GPU_science_retry=False,correction_source_path=str(A/'seal_runtime_stop_cpu.py'),correction_source_sha256=sha(A/'seal_runtime_stop_cpu.py')))
files={}
for p in sorted(D.rglob('*')):
    if not p.is_file() or p.suffix=='.pt' or '__pycache__' in p.parts or p.name.endswith('.lock') or '.partial' in p.name or p.name=='controller.log' or p.name.startswith('lane_GPU-') and p.name.endswith('.log') or p.name in ['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:continue
    files[str(p.relative_to(D))]=dict(bytes=p.stat().st_size,sha256=sha(p))
new=dict(old,files=files,status='RUNTIME_ENVIRONMENT_GATE_STOP_WITH_ACTUAL_ASSETS',CPU_delivery_revision=1,original_automatic_seal_archive_sha256=sha(archive),source_commit='55509063131e88f5040fbdf85452eef3145e41eb')
atomic_json(D/'RESULT_AND_DELIVERY_HASHES.json',new,replace=True)
(D/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n')
archive2=R/'delivery_cpu_revisions/execution_03_revision_01_delivery.tar.gz'
with tarfile.open(archive2,'w:gz',dereference=True) as t:
    for n in list(files)+['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:t.add(D/n,arcname=n,recursive=False)
print(json.dumps(dict(status='STOP_DELIVERY_RESEALED',files=len(files),manifest_sha256=sha(D/'RESULT_AND_DELIVERY_HASHES.json'),archive=str(archive2),archive_sha256=sha(archive2),first_failure_sha256=stop_sha,new_GPU_hours=resource['GPU_hours_new_execution_03'])))
