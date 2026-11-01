"""One-shot CPU analysis after a verified final GPU release; no GPU dispatch."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main(root,old_parent):
    root=root.resolve();cfg=json.loads((root/'continuation_control/CONFIG_SNAPSHOT.json').read_text())
    old=pathlib.Path(cfg['original_root']);gate=json.loads((root/'GATE_STATUS_FINAL.json').read_text());ledger=json.loads((root/'gpu_resource_ledger.json').read_text())
    assert gate['no_further_GPU_work'] and gate['GPU_release_verified'] and ledger['final_GPU_release_verified']
    assert gate['GPU_HOURS_USED']==ledger['GPU_HOURS_USED']>=cfg['prior_GPU_hours']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=dict(status='RUNNING_CPU_ONLY',started_remote_unix=time.time(),GPU_work_performed=False,steps=[])
    for script in ['final_dependency_verify.py','postprocess_scores.py','finalize_report.py']:
        p=root/'analysis_scripts'/script;start=time.monotonic()
        command=[sys.executable,str(p),'--root',str(root)]
        if script=='final_dependency_verify.py':command+=['--old-parent',str(old_parent)]
        q=subprocess.run(command,env=env)
        result['steps'].append(dict(script=script,source_sha256=sha(p),returncode=q.returncode,CPU_seconds=time.monotonic()-start))
        if q.returncode:
            result.update(status='CPU_DELIVERY_FAILED_NO_RETRY',ended_remote_unix=time.time())
            put(root/'analysis/OFFLINE_CONTINUATION_FINISH.json',result);return q.returncode
    original_failure=cfg['original_bindings']['FIRST_FAILURE.json']
    assert sha(old/'FIRST_FAILURE.json')==original_failure
    report=(root/'FINAL_SCIENTIFIC_REPORT.md').read_text()
    header=(
        '# 工程续跑与科学结果交付\n\n'
        '初次执行因 SSH stdout BrokenPipeError 在 Doc1 FP_NATIVE 32K 完成后停止，原 FAIL_AND_STOP、FIRST_FAILURE、封存报告和 checkpoint 全部原样保留。\n\n'
        '随后的人类原话为“你重新连接啊”。结合既有任务，执行者已向人类说明并据此仅恢复离线托管和剩余五条轨迹；这项解释与原话分别记录在 CONTINUATION_RECOVERY_MANIFEST.json，不伪称人类逐字给出五轨迹清单。\n\n'
        '原先两次 canary 和 Doc1 FP_NATIVE 完成资产通过原 seal/SHA 核验后继承，没有重复 FP 或前缀 forward。'
        '本次累计 GPUh 承接初次执行的 1.5062975196954276；24 GPUh 总额不重置。科学源码、原协议、两篇文档、输入、数学、GPU UUID、dtype、runtime 和 KL reducer 均未改变。\n\n'
        f'原失败文件 SHA256：`{original_failure}`。原执行路径：`{old}`。历史小文件原件见 `inherited_initial/`；旧大 tensor 保持原位，实际定位与 SHA 见其 REMOTE_TENSOR_LOCATORS.json。\n\n'
        '以下 COMPLETE/PARTIAL/FAIL 描述含继承资产的后续科学执行状态，绝不把初次执行的 FAIL 改判为 PASS。\n\n---\n\n'
    )
    p=root/'FINAL_CONTINUATION_REPORT.md'
    with p.open('x') as f:f.write(header+report)
    excluded={'hashes/REMOTE_RESULT_SHA256SUMS.txt','hashes/REMOTE_RESULT_VERIFICATION.json','analysis/OFFLINE_CONTINUATION_FINISH.json','logs/OFFLINE_FINISH.log','logs/TMUX_LAUNCH.log'}
    assets={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        rel=str(p.relative_to(root))
        if rel in excluded:continue
        assert not p.is_symlink(),rel
        before=p.stat();digest=sha(p);after=p.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),rel
        q=subprocess.run(['sha256sum',str(p)],capture_output=True,text=True,check=True)
        assert q.stdout.split()[0]==digest,rel
        assets[rel]=dict(sha256=digest,bytes=after.st_size)
    manifest=root/'hashes/REMOTE_RESULT_SHA256SUMS.txt';manifest.parent.mkdir(exist_ok=True)
    with manifest.open('x') as f:
        for rel,item in assets.items():f.write(item['sha256']+'  '+rel+'\n')
    put(root/'hashes/REMOTE_RESULT_VERIFICATION.json',dict(status='PASS',CPU_only=True,remote_only=True,local_delivery_verified=False,
        independent_full_stream_hash_methods=['hashlib.sha256','sha256sum'],payload_files=len(assets),assets=assets,
        manifest_sha256=sha(manifest),mutable_own_finish_logs_excluded=True,original_initial_FAIL_preserved=True,original_first_failure_sha256=original_failure,
        total_GPU_hours=gate['GPU_HOURS_USED'],no_GPU_retry=True,remaining_work='Local small-asset download and independent delivery seal on next connection'))
    result.update(status='REMOTE_RESULTS_READY_FOR_LOCAL_DELIVERY',ended_remote_unix=time.time(),actual_final_execution=gate['status'],total_GPU_hours=gate['GPU_HOURS_USED'],
        remote_result_manifest_sha256=sha(manifest),local_delivery_verified=False)
    put(root/'analysis/OFFLINE_CONTINUATION_FINISH.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='steps'},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old-parent',required=True)
    a=p.parse_args();sys.exit(main(pathlib.Path(a.root),pathlib.Path(a.old_parent)))
