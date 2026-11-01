#!/usr/bin/env python3
"""Offline deployment smoke only; fail-closed batches, no downstream generation."""
import json,os,subprocess,sys,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def runbatch(units):
 assert shutil.disk_usage('/data').free>=15*1024**3
 jobs=[]
 for gpu,repeat in units:
  log=(ROOT/f'logs/gpu{gpu}_repeat{repeat}.log').open('x')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
  proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/deployment_closure_gpu.py'),'--phase','smoke','--gpu',str(gpu),'--repeat',str(repeat)],env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((gpu,repeat,proc,log))
 codes={}
 for gpu,repeat,proc,log in jobs:
  c=proc.wait();log.close();codes[f'{gpu}_{repeat}']=c
  with (ROOT/f'logs/gpu{gpu}_repeat{repeat}.exitcode').open('x') as f:f.write(str(c)+'\n')
 assert set(codes.values())=={0},str(codes)
def main():
 (ROOT/'logs').mkdir(exist_ok=True)
 assert not list((ROOT/'analysis').glob('gpu*'))
 assert not (ROOT/'logs/smoke_controller.error.json').exists()
 runbatch([(0,1)])
 runbatch([(0,2),(1,1),(2,1),(3,1)])
 runbatch([(1,2),(2,2),(3,2)])
 with (ROOT/'logs/smoke_controller.completion.json').open('x') as f:json.dump(dict(status='PASS',units=8,conditions=['Hadamard','C6R','C7'],new_AIME_generation=0,scoring_started=False,stop='GPU_WORK_FINISHED_WAIT_FOR_CPU_FINAL_AUDIT'),f,indent=2)
 print('SMOKE_UNITS_COMPLETE_NO_AIME',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  with (ROOT/'logs/smoke_controller.error.json').open('x') as f:json.dump(dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),retry=False,new_AIME_generation=0),f,indent=2)
  raise
