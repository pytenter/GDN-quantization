"""Fail-closed sequential-arm offline controller. Never reads scores/gold."""
import fcntl, os, shutil, subprocess, time, traceback
import canonical60_common as A

def freeze_arm(arm):
    jobs=A.samples();root=A.ROOT/'raw'/arm.lower();files=list(root.glob('*.json'))
    A.require(len(files)==60,'raw arm cardinality not 60')
    rows=[]
    for j in jobs:
        p=A.final_path(arm,j);r=A.read(p);A.validate_raw(r,arm,j)
        c=A.read(A.completion_path(arm,j));A.require(c['file_sha256']==A.sha(p),'completion record SHA drift')
        A.require(p.stat().st_mode&0o222==0,'raw not immutable')
        rows.append(dict(condition=arm,problem_id=j['problem_id'],seed=j['seed'],rank=j['rank'],gpu=j['gpu'],GPU_uuid=j['GPU_uuid'],path=str(p),sha256=A.sha(p),payload_sha256=r['output_payload_sha256'],rotation_sha256=r['rotation_sha256'],runtime_config_sha256=r['runtime_config_sha256'],input_ids_hash=r['input_ids_hash'],num_generated_tokens=r['generated_token_count'],termination=r['finish_reason']))
    p=A.ROOT/f'manifests/{arm.lower()}_raw_outputs_sha256.txt'
    manifest=A.ROOT/f'manifests/{arm.lower()}_raw_output_manifest.json'
    if p.exists() or manifest.exists():
        A.require(p.exists() and manifest.exists(),'incomplete freeze metadata')
        A.verify_manifest(p);A.require(A.read(manifest)['records']==rows,'existing freeze changed')
    else:
        A.write_text(p,''.join(x['sha256']+'  '+x['path']+'\n' for x in rows),True)
        A.save(manifest,dict(status='PASS',condition=arm,count=60,records=rows,raw_outputs_immutable=True,scoring_started=False,freeze_utc=A.utc()),True)
    A.verify_manifest(p)
    print(arm+'_RAW_OUTPUT_FREEZE_GATE=PASS',flush=True)
    return rows

def run_arm(arm):
    A.verify_freeze()
    A.require(not list((A.ROOT/'logs').glob('*.error.json')),'prior failure: no automatic resume')
    A.require(not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists(),'condition STOP sentinel')
    cfg=A.read(A.ROOT/'configs/canonical60_protocol.json')
    A.require(shutil.disk_usage('/data').free>=cfg['disk_stop_threshold_bytes'],'disk stop threshold')
    # All scientific identity files are rehashed before each condition starts.
    revalidation=A.protected_revalidate();A.save(f'logs/{arm.lower()}_prelaunch_revalidation.json',dict(status='PASS',count=revalidation['count'],source_sha256=revalidation['source_sha256'],utc=A.utc()),True)
    n=A.nvml();baseline=[n.snapshot(i) for i in range(4)]
    A.require(all(s['uuid']==A.UUIDS[i] and not s['processes'] for i,s in enumerate(baseline)),'four GPUs not exclusively idle')
    A.save(f'logs/{arm.lower()}_gpu_baseline.json',baseline,True)
    children=[];handles=[];exitcodes={};failed=False
    try:
        for i in range(4):
            f=(A.ROOT/'logs'/f'{arm.lower()}_gpu{i}.log').open('x');handles.append(f)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
            p=subprocess.Popen([A.PY,str(A.ROOT/'scripts/canonical60_worker.py'),'--arm',arm,'--gpu',str(i)],cwd=A.REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            children.append((i,p));A.save(f'logs/{arm.lower()}_gpu{i}_launch.json',dict(pid=p.pid,gpu=i,uuid=A.UUIDS[i],arm=arm,utc=A.utc()),True)
        while len(exitcodes)<4:
            for i,p in children:
                rc=p.poll()
                if rc is not None and i not in exitcodes:
                    exitcodes[i]=rc;A.save(f'logs/{arm.lower()}_gpu{i}_exit.json',dict(pid=p.pid,exitcode=rc,utc=A.utc()),True)
                    if rc!=0: failed=True
            if failed and not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists():
                A.save(f'logs/{arm.lower()}.stop.json',dict(reason='worker nonzero exit; finish current peer requests, prevent next sample',utc=A.utc(),automatic_retry=False),True)
            if len(exitcodes)<4: time.sleep(2)
    except BaseException:
        if not (A.ROOT/'logs'/f'{arm.lower()}.stop.json').exists(): A.save(f'logs/{arm.lower()}.stop.json',dict(reason='controller exception; no next samples',utc=A.utc()),True)
        # Do not signal/kill any GPU worker. Their sentinel check stops next launch.
        raise
    finally:
        for f in handles:f.close()
    A.require(not failed and all(r==0 for r in exitcodes.values()),'post-forward worker condition FAIL; no retries or next arm')
    snapshots=[]
    for k in range(11):
        current=[n.snapshot(i) for i in range(4)];snapshots.append(current)
        if all(not s['processes'] and s['used_bytes']==baseline[i]['used_bytes'] for i,s in enumerate(current)):break
        time.sleep(1)
    reclaim=all(not s['processes'] and s['used_bytes']==baseline[i]['used_bytes'] for i,s in enumerate(current))
    A.save(f'analysis/{arm.lower()}_normal_exit_reclaim.json',dict(status='PASS' if reclaim else 'FAIL',exitcodes=exitcodes,baseline=baseline,after=current,history=snapshots),True)
    A.require(reclaim,'normal process exit reclaim gate FAIL')
    for i in range(4): A.require(A.read(A.ROOT/f'analysis/{arm.lower()}_gpu{i}_worker.json')['completed']==15,'worker completion mismatch')
    return freeze_arm(arm)

def main():
    lock=(A.ROOT/'logs/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    A.verify_freeze()
    binding=A.read(A.ROOT/'logs/premeasurement_commit.json')
    A.require(A.git('rev-parse','HEAD')==binding['commit'],'premeasurement HEAD changed')
    A.require(A.git('rev-parse','HEAD^')==A.PARENT,'unexpected source lineage')
    for arm in A.SHAS:
        for j in A.samples():
            if A.started_path(arm,j).exists() and not A.final_path(arm,j).exists(): raise RuntimeError('interrupted post-forward sample; no automatic retry')
    c6=run_arm('C6R');c7=run_arm('C7')
    A.require([(r['problem_id'],r['seed'],r['gpu'],r['GPU_uuid'],r['input_ids_hash']) for r in c6]==[(r['problem_id'],r['seed'],r['gpu'],r['GPU_uuid'],r['input_ids_hash']) for r in c7],'paired identity closure')
    both=c6+c7
    A.write_text('manifests/canonical60_120_raw_outputs_sha256.txt',''.join(r['sha256']+'  '+r['path']+'\n' for r in both),True)
    A.save('manifests/canonical60_120_raw_outputs.json',dict(count=120,per_condition=60,records=both,immutable=True),True)
    A.verify_manifest(A.ROOT/'manifests/canonical60_120_raw_outputs_sha256.txt')
    A.save('analysis/canonical60_raw_output_closure.json',dict(CANONICAL60_RAW_OUTPUT_CLOSURE_GATE='PASS',C6R=60,C7=60,total=120,no_missing=True,no_duplicates=True,no_overwrite=True,same_keys_prompts_and_GPU=True,all_immutable=True,SHA_reverify='PASS',SCORING_AUTHORIZED='YES',utc=A.utc()),True)
    print('RAW120_FROZEN_SCORING_AUTHORIZED',flush=True)
    subprocess.run([A.PY,str(A.ROOT/'scripts/canonical60_analyze.py'),'--execute'],cwd=A.REPO,check=True)

if __name__=='__main__':
    try: main()
    except BaseException as e:
        A.save('logs/controller.error.json',dict(status='FAIL_CLOSED',error=str(e),traceback=traceback.format_exc(),automatic_retry=False,utc=A.utc()),True)
        print('FAIL_CLOSED_NO_AUTOMATIC_RETRY',str(e),flush=True)
        raise
