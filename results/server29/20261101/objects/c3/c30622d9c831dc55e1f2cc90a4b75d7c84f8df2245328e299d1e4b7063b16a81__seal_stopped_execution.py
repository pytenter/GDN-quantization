"""CPU-only append-only closure after already started B workers finish.

Never launches a model worker and never retries the stopped Doc3. Preserves
the earlier report/manifest, which was made while other workers still ran.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numerical_recovery as N

ROOT=N.ROOT
RUNTIME=Path('/data/zypan/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime')


def main():
    N.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only stopped-execution seal')
    controller=Path('/proc/4325/cmdline')
    while controller.exists():
        command=controller.read_bytes().replace(b'\0',b' ').decode()
        N.check(str(ROOT/'scripts/scale_science.py')+' controller' in command,'exact original parent identity')
        time.sleep(15)
    for name in ['science_controller_001.log','pipeline_controller_001.log']:
        source=RUNTIME/name;target=ROOT/'logs'/(name.removesuffix('.log')+'_after_stop_001.log')
        N.check(not target.exists(),'append-only stopped-run logs')
        shutil.copyfile(source,target)
    complete=[]
    for path in sorted(ROOT.glob('track_B/doc_*/result.json')):
        data=N.read(path)
        complete.append({'document':data['scores'][0]['document_index'],'status':data['status'],
                         'score_rows':len(data['scores']),'receipt_sha256':N.sha(path),'path':str(path.relative_to(ROOT)),
                         'measured_GPU_hours':data['GPU_hours'],'peak_allocated_bytes':data['peak_allocated_bytes']})
    diagnosis=N.read(ROOT/'engineering/DOC3_TEXT_TOKEN_AUDIT_FAILURE_DIAGNOSIS_001.json')
    N.save('POSTSTOP_EXECUTION_CLOSURE_001.json',{'utc':N.now(),'status':'FAIL_AND_STOP_PRESERVED',
        'NUMERICAL_CLOSURE_STATUS':'PASS','OBSERVER_PARITY':'PASS','TRACK_B_ENGINEERING_1024':'PASS',
        'completed_formal_document_receipts':complete,'Doc3_formal_model_forward_count':0,
        'formal_full_panel_complete':False,'Track_A':'NOT_RUN','matched_long_G':'NOT_RUN',
        'audit_failure_classification':diagnosis['classification'],'frozen_document_or_token_changed':False,
        'automatic_fix_or_retry_performed':False,'old_report_and_manifest_preserved':True,
        'earlier_root_manifest_warning':'Initial seal was produced before remaining already-started finite units finished; use this append-only final closure and POSTSTOP_SHA256SUMS_001.txt.',
        'final_resource_snapshot':N.resources()})
    N.prose('POSTSTOP_REPORT_001.md', '# V3 停止后的执行封存\n\n'
        '数值闭环 PASS：补偿scan的1110个全模型对象与离线数学参考逐字节相同，三进程重复一致。'
        'Observer两进程OFF/ON全部PASS；Doc1 H/C6R 1024-token工程资格PASS。\n\n'
        '正式扩展的Doc3因只读审核将CRLF转LF而STOP。原始文本SHA与冻结32768 token逐项匹配，资产没有变化；Doc3模型forward=0。未修复或重试。\n\n'
        f'其余已经启动的正式文档完成数：{len(complete)}/4（详见POSTSTOP_EXECUTION_CLOSURE_001.json）。完整面板仍为FAIL_AND_STOP；Track A/G未启动。'
        '不把工程资格分数当正式完整面板，也不输出未运行CSV。\n\n'
        '原FINAL_REPORT.md和SHA256SUMS.txt保持原样。此报告及POSTSTOP_SHA256SUMS_001.txt补充已启动单元自然结束后的完整证据。'
        '只有用户授权只读文本解码审核修复后，才可对零forward的Doc3做重新资格认证并恢复未完成部分。\n')
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='POSTSTOP_SHA256SUMS_001.txt')
    N.prose('POSTSTOP_SHA256SUMS_001.txt',''.join(N.sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    for line in (ROOT/'POSTSTOP_SHA256SUMS_001.txt').read_text().splitlines():
        digest,rel=line.split('  ',1);N.check(N.sha(ROOT/rel)==digest,'poststop full artifact hashes')
    print(json.dumps({'POSTSTOP_SEAL':'PASS','documents_completed':len(complete),'manifest_sha256':N.sha(ROOT/'POSTSTOP_SHA256SUMS_001.txt')}),flush=True)


if __name__=='__main__':main()
