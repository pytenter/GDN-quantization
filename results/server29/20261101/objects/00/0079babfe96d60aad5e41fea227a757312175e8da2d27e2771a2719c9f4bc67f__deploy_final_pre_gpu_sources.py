#!/usr/bin/env python3
"""Deploy this authorized engineering attempt before scientific STARTED/freeze.

An already completed load-only engineering probe is preserved and counted as
GPU work, never relabelled scientific qualification or zero GPU work.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'ROOT.json').read_text())
REMOTE=CONFIG['remote_root']
PYTHON=CONFIG['remote_python']
SSH=['ssh','-S',CONFIG['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=20','zypan@202.38.247.55']
PRIOR153='f0c27e48442e924b1a49fa254c4f262f49be85da00a3ec52c3073d7e73e77d5c'
PROBE_COMPLETION='operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json'
PROBE_REVIEW='operations/ENGINEERING_NATIVE_MAPS_FINAL_REVIEW.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def safe_relative(value):
    p=Path(value)
    if not isinstance(value,str)or not value or p.is_absolute()or str(p)!=value or any(x in('','..','.')for x in value.split('/')):
        raise RuntimeError('Unsafe relative source path: '+str(value))
    return p


def ordinary_tree(root):
    """Reject every local symlink/special file; no hidden negative-fixture tree."""
    root=Path(root)
    if root.is_symlink()or not root.is_dir():raise RuntimeError('Source root is not an ordinary directory')
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(directory)/name;s=p.lstat()
            if stat.S_ISLNK(s.st_mode)or not(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)):
                raise RuntimeError('Symlink/special file anywhere in candidate science root: '+str(p.relative_to(root)))


def probe_counts(root):
    root=Path(root);p=root/PROBE_COMPLETION
    # A started but unfinished engineering probe must be resolved before any
    # source replacement. No qualification/controller STARTED is permitted.
    started=list((root/'operations').glob('*LOAD_ONLY*STARTED*.json'))+list((root/'operations').glob('*NATIVE*PROBE*STARTED*.json'))
    started.extend(x for x in(root/'operations/ENGINEERING_LOAD_ONLY_PROBE_LAUNCH.json',root/'operations/loader_native_probe/DISPATCH.json',root/'operations/loader_native_probe/STARTED.json')if x.exists())
    if not p.exists():
        if started:raise RuntimeError('Engineering native loader probe started without completion')
        return dict(scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=0,total_GPU_work_dispatches=0)
    a=json.loads(p.read_text())
    if not(a['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and a['actual_wait_returncode']==0
           and a['all_exact_own_processes_exited']is True and a['own_GPU_UUID_released']is True
           and a['model_forward_calls']==a['QDQ_calls']==a['AIME_generations']==0
           and a['counts_as_scientific_qualification']is False):
        raise RuntimeError('Engineering probe has not honestly completed/exited/released')
    return dict(scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=1,total_GPU_work_dispatches=1)


def paths(root=ROOT):
    root=Path(root);ordinary_tree(root)
    grant=json.loads((root/'HUMAN_REPAIR_AUTHORIZATION.json').read_text())
    if grant['prior_source_freeze_sha256']!=PRIOR153:raise RuntimeError('Repair authorization does not bind prior153 STOP')
    probe_counts(root)
    excluded={'SOURCE_FREEZE.json','GPU_PERMIT.json','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','SHA256_MANIFEST.json'}
    selected={p for p in root.iterdir()if p.is_file()and p.suffix in('.json','.md','.txt')and p.name not in excluded}
    raw={'.pyc','.pyo','.npz','.npy','.bin','.pt','.pth','.safetensor','.safetensors'}
    for name in('src','scripts','vendor','protocol','calibration','history','analysis','provenance'):
        for p in(root/name).rglob('*'):
            if p.is_file()and '__pycache__'not in p.parts and p.suffix.lower()not in raw:selected.add(p)
    selected.update((root/'operations').glob('*.py'))
    seeds=[root/'analysis/RUNTIME_NATIVE_CLOSURE.json',root/'operations/REMOTE_ONLY25_ASSET_COPY.json',
           root/'operations/P1_AUTHORIZED_STOP_COMPLETION/USER_STOP_RECEIPT.json']
    seeds.extend(p for p in(root/'operations').glob('ENGINEERING_*.json')if p.is_file())
    remote_rows=json.loads((root/'operations/REMOTE_ONLY25_ASSET_COPY.json').read_text())['files']
    remote_only={x['path']:x for x in remote_rows}
    queue=list(seeds);seen=set()
    while queue:
        p=queue.pop(0)
        if p in seen:continue
        seen.add(p)
        if not p.is_file()or p.is_symlink():raise RuntimeError('Missing ordinary source-bound file: '+str(p))
        selected.add(p)
        if p.suffix!='.json':continue
        value=json.loads(p.read_text())
        def visit(v):
            if isinstance(v,dict):
                for key,item in v.items():
                    if key in('path','relative_path','receipt_path')and isinstance(item,str)and item and not Path(item).is_absolute():
                        rel=safe_relative(item);target=root/rel
                        if item in remote_only:
                            row=remote_only[item]
                            if 'sha256'in v and(v['sha256']!=row['sha256']or v.get('bytes',row['bytes'])!=row['bytes']):
                                raise RuntimeError('Remote raw asset binding differs: '+item)
                        else:
                            if not target.is_file()or target.is_symlink():raise RuntimeError('Bound local file missing: '+item)
                            b=target.read_bytes()
                            if 'sha256'in v and(sha(b)!=v['sha256']or v.get('bytes',len(b))!=len(b)):
                                raise RuntimeError('Bound source/report bytes differ: '+item)
                            queue.append(target)
                    elif key not in('path','relative_path','receipt_path'):visit(item)
            elif isinstance(v,list):
                for item in v:visit(item)
        visit(value)
    return sorted(selected)


RECEIVER=r'''
import hashlib,json,os,pathlib,stat,sys,tarfile
r=pathlib.Path(sys.argv[1]);assert not r.is_symlink();r=r.resolve()
assert not any((r/n).exists()for n in('SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','OFFLINE_STARTED.json')), 'Already frozen/scientifically started'
completion=r/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json'
started=any((r/n).exists()for n in('operations/ENGINEERING_LOAD_ONLY_PROBE_LAUNCH.json','operations/loader_native_probe/DISPATCH.json','operations/loader_native_probe/STARTED.json'))
assert not started or completion.exists(),'Engineering probe active/unresolved; no source replacement'
if completion.exists():
 proof=json.loads(completion.read_text());assert proof['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'
 for expected in [proof['worker_identity']]+([proof['supervisor_identity']]if 'supervisor_identity'in proof else[]):
  proc=pathlib.Path('/proc')/str(expected['pid'])
  try:
   raw=(proc/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
   current=dict(pid=expected['pid'],starttick=int(tail[19]),cmdline=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
   assert current!=expected or tail[0]in('Z','X','x'),'Exact engineering probe process still active'
  except FileNotFoundError:pass

def regular_parent(p):
 cur=r
 for part in p.relative_to(r).parts[:-1]:
  cur=cur/part
  if os.path.lexists(cur):assert cur.is_dir()and not cur.is_symlink(),'Nonordinary parent'
  else:cur.mkdir()
 assert not os.path.lexists(p)or(p.is_file()and not p.is_symlink()),'Nonordinary destination'
rows=[];preserved=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|')as tar:
 for m in tar:
  rel=pathlib.PurePosixPath(m.name);assert m.isfile()and not rel.is_absolute()and '..'not in rel.parts
  p=r/str(rel);assert p.resolve().is_relative_to(r);regular_parent(p)
  b=tar.extractfile(m).read();assert len(b)==m.size;h=hashlib.sha256(b).hexdigest()
  if p.exists()and p.read_bytes()!=b:
   old=p.read_bytes();oldsha=hashlib.sha256(old).hexdigest();prior=r/'operations/PRE_FREEZE_DEPLOYMENT_VERSIONS'/oldsha/str(rel)
   regular_parent(prior)
   if not prior.exists():prior.write_bytes(old)
   assert prior.read_bytes()==old;preserved.append(dict(path=str(rel),prior_sha256=oldsha))
  p.write_bytes(b);assert p.stat().st_size==len(b)and hashlib.sha256(p.read_bytes()).hexdigest()==h
  rows.append(dict(path=str(rel),bytes=len(b),sha256=h))
repair=json.loads((r/'HUMAN_REPAIR_AUTHORIZATION.json').read_text())
assert repair['prior_source_freeze_sha256']=='f0c27e48442e924b1a49fa254c4f262f49be85da00a3ec52c3073d7e73e77d5c'
completion=r/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json';probes=0
if completion.exists():
 a=json.loads(completion.read_text());assert a['status']=='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED'and a['actual_wait_returncode']==0
 assert a['all_exact_own_processes_exited']and a['own_GPU_UUID_released']and a['counts_as_scientific_qualification']is False
 assert a['model_forward_calls']==a['QDQ_calls']==a['AIME_generations']==0;probes=1
print(json.dumps(dict(status='PASS_PRE_SCIENTIFIC_GPU_DEPLOYMENT_NOT_FROZEN_OR_STARTED',files=rows,preserved_prior_versions=preserved,
 scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=probes,total_GPU_work_dispatches=probes),sort_keys=True))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--receipt',required=True);args=parser.parse_args()
    selected=paths();assert not(ROOT/'SOURCE_FREEZE.json').exists(),'No post-freeze deployment'
    if not args.execute:
        print(json.dumps(dict(count=len(selected),bytes=sum(p.stat().st_size for p in selected),execute=False,**probe_counts(ROOT))));return
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w')as tar:
        for p in selected:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    command='exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join([PYTHON,'-u','-B','-c',RECEIVER,REMOTE])
    result=subprocess.run(SSH+[command],input=buffer.getvalue(),capture_output=True,timeout=300)
    target=ROOT/'operations'/safe_relative(args.receipt);assert target.parent==ROOT/'operations'
    with target.open('x')as handle:json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),handle,sort_keys=True,indent=2);handle.write('\n')
    assert result.returncode==0,result.stderr.decode();actual=json.loads(result.stdout);assert len(actual['files'])==len(selected)
    print(json.dumps(dict(status=actual['status'],files=len(selected),bytes=len(buffer.getvalue()),
                         scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=actual['engineering_native_loader_GPU_probes'])))


if __name__=='__main__':main()
