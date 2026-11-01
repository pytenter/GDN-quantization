#!/usr/bin/env python3
"""Append-only CPU audit command compatibility correction; no GPU execution.

Preserves the complete initial audit/STOP. Does not change scientific inputs,
the short-horizon design, metrics, protocols, budgets or production code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(v, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--root', required=True)
    a.add_argument('--spec', required=True)
    a.add_argument('--initial-source', required=True)
    args = a.parse_args()
    root = Path(args.root)
    initial = read(root/'ASSET_AND_PROVENANCE_AUDIT.json')
    assert initial['status'] == 'AUDIT_BLOCKED_IDENTITY'
    assert all(r['status']=='PASS' for r in initial['audited_checks'])
    assert all(r['status']=='PASS' for r in initial['Native_H_cpu_matrix_identity'])
    assert initial['direction_postfinal_manifest']['status']=='PASS'
    old_rows = {}
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        h, n = line.split(None,1)
        p=root/n.strip()
        assert sha(p)==h
        old_rows[n.strip()]=h
    old_manifest_sha = sha(root/'SHA256SUMS.txt')
    repo=Path(initial['source_roots']['EXP']).parents[1]

    def command(args):
        # git1.8.3.1 has no -C; subprocess cwd changes only command invocation.
        p=subprocess.run(args,capture_output=True,text=True,cwd=repo)
        return dict(command=args,cwd=str(repo),returncode=p.returncode,
                    stdout=p.stdout.strip(),stderr=p.stderr.strip())

    before=command(['git','status','--porcelain','--untracked-files=all'])
    version=command(['git','--version'])
    previous_bad=command(['git','-C',str(repo),'cat-file','-t',initial['historical_premeasurement_commit']])
    corrected=command(['git','cat-file','-t',initial['historical_premeasurement_commit']])
    head=command(['git','rev-parse','HEAD'])
    assert previous_bad['returncode']==129 and 'Unknown option: -C' in previous_bad['stderr']
    assert before['returncode']==0 and corrected['returncode']==0 and corrected['stdout']=='commit'
    assert head['returncode']==0 and head['stdout']==initial['historical_premeasurement_commit']
    active=root/'finalized_audit_002'
    active.mkdir(exist_ok=False)
    source=active/'source'
    source.mkdir()
    shutil.copyfile(args.initial_source,source/'audit_short_scale_revision001.py')
    shutil.copyfile(__file__,source/'correct_audit_command_001.py')
    shutil.copyfile(args.spec,source/'TASK_SPEC.txt')
    assert sha(source/'TASK_SPEC.txt')=='e120083c47378a0a3a1844310f31817f037f4bc2e35eb4b8a45c64521bcf1276'
    stamp=datetime.now(timezone.utc).isoformat()
    correction=dict(status='PASS_CPU_AUDIT_COMMAND_COMPATIBILITY',
                    scope='COMMAND_WORKING_DIRECTORY_ONLY_NO_SCIENTIFIC_CHANGE',
                    classification='AUDIT_WRAPPER_UNSUPPORTED_GIT_OPTION',
                    old_call="subprocess.run(['git','-C',repo,'cat-file','-t',commit])",
                    corrected_call="subprocess.run(['git','cat-file','-t',commit],cwd=repo)",
                    git_version=version, original_failed_command=previous_bad,
                    corrected_command=corrected, current_head=head,
                    all_original_frozen_file_identity_checks='PASS',
                    original_failure_preserved=True, old_manifest_sha256=old_manifest_sha,
                    old_files=old_rows, current_P0='AUDIT_BLOCKED',
                    current_stop_reason='RELEASED_REQUIRED_INPUTS_AND4GPUh_BUDGET',
                    actual_gpu_hours=0.0, scientific_units=0,
                    preregistration_changed=False, measurement_started=False,
                    correction_utc=stamp, correction_code_sha256=sha(__file__))
    save(active/'AUDIT_COMMAND_COMPATIBILITY_AMENDMENT_001.json',correction)
    for p in sorted(root.glob('*.json')):
        j=read(p)
        j['CPU_audit_command_correction']='AUDIT_COMMAND_COMPATIBILITY_AMENDMENT_001.json'
        j['initial_audit_preserved_directory']=str(root)
        j['current_final_audit_directory']=str(active)
        j['correction_code_sha256']=sha(__file__)
        if p.name=='ASSET_AND_PROVENANCE_AUDIT.json':
            j['status']='AUDIT_BLOCKED'
            j['identity_revalidation']='PASS'
            j['old_git_commit_command_corrected']=corrected
            j['protected_checkout_unchanged']='NOT_TESTED_BY_INITIAL_GIT_COMMAND; STATIC_READONLY_CODE_AND_SUCCESSFUL_CORRECTED_BEFORE_AFTER_SNAPSHOT'
        if p.name=='DISCOVERY_VERDICT.json':j['P0']='AUDIT_BLOCKED'
        save(active/p.name,j)
    original_report=(root/'FINAL_SCIENTIFIC_REPORT.md').read_text()
    report=original_report.replace('P0 = AUDIT_BLOCKED_IDENTITY','P0 = AUDIT_BLOCKED').replace('身份重新核验：FAIL','身份重新核验：PASS')
    report += '\n## 附录：CPU审计命令兼容性修正\n\n初版审计将Git1.8.3.1不支持的`git -C`返回码误归为identity failure。全部实际文件和矩阵SHA检查此前已PASS。改为`subprocess cwd`后，历史commit对象和HEAD均通过只读核验；没有科学计算、阈值变化或GPU重试。初版全部文件、STOP与原始SHA256SUMS保存在上一层，未覆盖。本目录为当前有效的最终P0审计包。原始连续G的合法性未否定；执行仍因输入释放与4GPUh预算而停止。\n'
    with (active/'FINAL_SCIENTIFIC_REPORT.md').open('x',encoding='utf-8',newline='\n') as f:f.write(report)
    after=command(['git','status','--porcelain','--untracked-files=all'])
    assert before==after
    save(active/'FINAL_INTEGRITY_CHECK.json',dict(status='PASS',git_before=before,git_after=after,
         preserved_old_manifest_sha256=old_manifest_sha, old_files_byte_identical=True,
         science_and_GPU_execution_count=0, sha256_manifest_role='Final audit and source/spec; excludes the manifest itself.'))
    for n,h in old_rows.items():assert sha(root/n)==h
    assert sha(root/'SHA256SUMS.txt')==old_manifest_sha
    with (active/'SHA256SUMS.txt').open('x',encoding='utf-8',newline='\n') as f:
        for p in sorted(active.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt':f.write(f'{sha(p)}  {p.relative_to(active)}\n')
    save(root/'CURRENT_ARTIFACTS.json',dict(status='AUDIT_BLOCKED_BUDGET_STOP',
         current_directory=str(active),current_final_report=str(active/'FINAL_SCIENTIFIC_REPORT.md'),
         current_manifest=str(active/'SHA256SUMS.txt'),current_manifest_sha256=sha(active/'SHA256SUMS.txt'),
         initial_audit_files_preserved=True,original_manifest_sha256=old_manifest_sha,
         initial_false_identity_stop='AUDIT_WRAPPER_COMPATIBILITY_ONLY_CORRECTED_APPEND_ONLY',
         no_existing_scientific_asset_writes=True, actual_gpu_hours=0.0))
    print(json.dumps(dict(current_directory=str(active), identity='PASS', P0='AUDIT_BLOCKED',
                         actual_gpu_hours=0,current_manifest_sha256=sha(active/'SHA256SUMS.txt'))))


if __name__=='__main__':main()
