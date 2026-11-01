#!/usr/bin/env python3
"""Transport-only pinned wheels: curl HTTPS, official wheel SHA, historical transitive versions."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]
WHEELS=Path('/root/qwen_a800_wheelhouse')
LOCK=Path('/root/autodl-tmp/GDN-formal-c5-c6-h64-v1/experiments/qwen_gdn/recurrent_training/QWEN_GDN_48GB_SINGLE_GPU_RECURRENT_TRAINING_V1/configs/vgpu_pip_freeze.txt')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def fetch(row):
    path=WHEELS/row['filename']
    if path.exists():
        if sha(path)!=row['sha256']:raise RuntimeError('existing wheel hash mismatch')
        return row
    url=row['url'].replace('https://files.pythonhosted.org/packages/','https://mirrors.aliyun.com/pypi/packages/')
    start=time.monotonic()
    subprocess.run(['curl','--fail','--location','--silent','--show-error','--max-time','900',url,'--output',str(path)],check=True)
    if sha(path)!=row['sha256']:raise RuntimeError('wheel exact SHA failed: '+row['filename'])
    row['download_seconds']=time.monotonic()-start
    print(json.dumps({'event':'wheel_transport_verified','file':path.name,'bytes':path.stat().st_size,'seconds':row['download_seconds']}),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--extra-only',action='store_true');args=p.parse_args()
    suffix='_extras' if args.extra_only else ''
    WHEELS.mkdir(exist_ok=True)
    lock={}
    for line in LOCK.read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name,version=line.strip().split('==',1);lock[canonicalize_name(name)]=version
    required={}
    for line in (ROOT/'scripts/a800_requirements.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            r=Requirement(line);version=next(s.version for s in r.specifier if s.operator=='==');required[canonicalize_name(r.name)]=version
    if args.extra_only:required={'typer':'0.27.2'}
    queue=list(required);required['torch']='2.5.1+cu121';resolved={};available=set(sys_tags());environment=default_environment()
    environment.update(python_version='3.10',python_full_version='3.10.18',extra='')
    while queue:
        name=queue.pop(0)
        if name in resolved:continue
        version=required[name]
        text=subprocess.check_output(['curl','--fail','--silent','--show-error','--max-time','60',f'https://pypi.org/pypi/{name}/{version}/json'],text=True)
        meta=json.loads(text)
        choices=[]
        for item in meta['urls']:
            if item['filename'].endswith('.whl'):
                _,_,_,tags=parse_wheel_filename(item['filename'])
                if tags & available:choices.append(item)
        if not choices:raise RuntimeError('no compatible pinned wheel, no opportunistic source build: '+name)
        item=sorted(choices,key=lambda x:x['filename'])[0]
        row={'name':name,'version':version,'filename':item['filename'],'sha256':item['digests']['sha256'],'url':item['url'],'size':item['size']}
        resolved[name]=row
        for text in meta['info'].get('requires_dist') or []:
            dep=Requirement(text)
            if dep.marker and not dep.marker.evaluate(environment):continue
            key=canonicalize_name(dep.name)
            choice=required.get(key,lock.get(key))
            if choice is None:raise RuntimeError('transitive dependency has no historical version: '+key)
            if choice not in dep.specifier:raise RuntimeError('historical dependency incoherent: '+text+' vs '+choice)
            if key not in required:required[key]=choice;queue.append(key)
    plan={'versions':required,'historical_lock_sha256':sha(LOCK),'explicit_requirements_sha256':sha(ROOT/'scripts/a800_requirements.txt'),'wheels':list(resolved.values()),'only_transport_changed':True}
    with (ROOT/f'provenance/wheel_transport_plan{suffix}.json').open('x') as f:json.dump(plan,f,indent=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(fetch,resolved.values()))
    with (ROOT/f'provenance/wheel_transport_verified{suffix}.json').open('x') as f:json.dump({'status':'PASS','wheels':completed},f,indent=2)
    with (ROOT/f'provenance/resolved_install_requirements{suffix}.txt').open('x') as f:
        for n,v in sorted(required.items()):f.write(f'{n}=={v}\n')


if __name__=='__main__':main()
