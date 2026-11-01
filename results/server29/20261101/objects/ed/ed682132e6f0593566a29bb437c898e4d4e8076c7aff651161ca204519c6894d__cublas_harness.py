#!/usr/bin/env python3
"""Pure CPU schedule/entry builder. No GPU execution in this module."""
import argparse,json,os,sys
from pathlib import Path
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
WORKSPACES={'B0':':4096:8','B1':':16:8'}
def build_stage_schedule():
 rows=[]
 for condition,repeat in [('B0',0),('B0',1),('B1',0),('B1',1)]:rows.append(dict(unit_id=f'{condition}_FP_Q{repeat}',mode='qualify',condition=condition,codec='FP',repeat=repeat,reference_unit=None if(condition,repeat)==('B0',0) else 'B0_FP_Q0',optional=False,estimated_seconds=180))
 for repeat in range(3):
  for condition in (['B0','B1'] if repeat%2==0 else ['B1','B0']):rows.append(dict(unit_id=f'{condition}_FP_P{repeat}',mode='benchmark',condition=condition,codec='FP',repeat=repeat,reference_unit=None,optional=False,estimated_seconds=90))
 for condition in ['B0','B1']:rows.append(dict(unit_id=f'{condition}_FP_PROFILE',mode='profile',condition=condition,codec='FP',repeat=0,reference_unit=None,optional=True,estimated_seconds=240))
 for condition,repeat in [('B0',0),('B0',1),('B1',0),('B1',1)]:rows.append(dict(unit_id=f'{condition}_H_C128_Q{repeat}',mode='qualify',condition=condition,codec='H_C128',repeat=repeat,reference_unit=None if(condition,repeat)==('B0',0) else 'B0_H_C128_Q0',optional=True,estimated_seconds=180))
 return rows
def fresh_environment(condition,base=None,device=None):
 if condition not in WORKSPACES:raise ValueError('Unregistered workspace condition')
 device={'physical_index':0} if device is None else device
 if type(device.get('physical_index')) is not int or device['physical_index'] not in range(4):raise ValueError('Unregistered physical GPU index')
 e=dict(os.environ if base is None else base);e.update(CUBLAS_WORKSPACE_CONFIG=WORKSPACES[condition],CUDA_VISIBLE_DEVICES=str(device['physical_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');return e
def build_command(python,root,authorization,unit):
 if unit not in build_stage_schedule():raise ValueError('Unregistered unit')
 return [str(python),'-B',str(Path(root)/'scripts/cublas_worker.py'),'--root',str(root),'--authorization',str(authorization),'--unit-id',unit['unit_id'],'--mode',unit['mode'],'--condition',unit['condition'],'--codec',unit['codec']]
def numeric_group_verdict(root,codec):
 from cublas_compare import reduce_qualification
 try:return reduce_qualification(root,codec)
 except FileNotFoundError as e:return dict(status='NOT_RUN',codec=codec,error=str(e))
 except Exception as e:return dict(status='FAIL',codec=codec,error=str(e))
def performance_verdict(root):
 from cublas_compare import reduce_performance
 try:
  r=reduce_performance(root);return dict(r,status='PASS_BENEFIT' if r['benefit_pass'] else 'PASS_NO_BENEFIT',median_B0_TPS=r['median_TPS']['B0'],median_B1_TPS=r['median_TPS']['B1'],nonoverlap=r['all_B1_above_all_B0'])
 except Exception as e:return dict(status='FAIL',error=str(e))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--python',required=True);a=p.parse_args()
 print(json.dumps(dict(status='PREPARED_NOT_EXECUTED_NO_GPU_AUTHORIZATION',schedule=[dict(unit=u,argv=build_command(a.python,a.root,a.authorization,u),environment_delta={k:fresh_environment(u['condition'],{})[k] for k in fresh_environment(u['condition'],{})}) for u in build_stage_schedule()]),indent=2))
if __name__=='__main__':main()
