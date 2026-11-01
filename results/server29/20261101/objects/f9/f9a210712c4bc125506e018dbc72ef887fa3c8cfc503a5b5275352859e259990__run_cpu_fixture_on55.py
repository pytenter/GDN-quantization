#!/usr/bin/env python3
"""Only CPU tiny-model diagnostics on the known host. Never GPU admission."""
import argparse,datetime,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observer',action='store_true');a=p.parse_args()
    fixture=(R/'qualification/real_qwen_cpu_fixture.py').read_text()
    if a.observer:
        source=(R/'worker/restoration_worker.py').read_text()
        payload='__name__="cpu_observer_stdin"\n'+(R/'qualification/cpu_observer_check.py').read_text()+'\nrun('+repr(source)+','+repr(fixture)+')\n';tag='CPU_OBSERVER_RESULT=';prefix='CPU_OBSERVER'
    else:payload=fixture;tag='CPU_FIXTURE_RESULT=';prefix='CPU_FIXTURE'
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S','/tmp/qwen_gdn_replication_20261013.sock','zypan@202.38.247.55','env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/ydai/miniconda3/envs/bitdecode/bin/python -B -']
    t=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');log=R/'qualification'/(prefix+'_'+t+'.log')
    proc=subprocess.run(cmd,input=payload.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180);log.write_bytes(proc.stdout)
    rows=[line.split(tag,1)[1]for line in proc.stdout.decode(errors='replace').splitlines()if line.startswith(tag)]
    receipt={'returncode':proc.returncode,'log':str(log),'result':json.loads(rows[-1])if rows else None,'GPU_dispatches':0,'GPU_queries':0,'CPU_device_guard':'CUDA_VISIBLE_DEVICES empty plus torch.cuda._lazy_init denied'}
    (R/'qualification'/(prefix+'_'+t+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'returncode':proc.returncode,'status':receipt['result']['status'] if rows else 'NO_RECEIPT','log':str(log),'checks':len(receipt['result'].get('checks',[])) if rows else 0}))
    if proc.returncode:raise SystemExit(proc.returncode)
if __name__=='__main__':main()
