#!/usr/bin/env python3
"""Inventory a provenanced native-loader closure without a model or CUDA context.

Run in a fresh target Python with CUDA_VISIBLE_DEVICES='' and -B. The input is
the unmodified original external-bindings JSON; this tool never writes runtime
libraries, model files, or an experiment. Output is a CPU diagnostic receipt.
The explicit provider inventory is preparation evidence, not a runtime prefix
allow-list. Runtime admission must match individual canonical files exactly.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def elf(path):
    try:
        with Path(path).open('rb')as f:return f.read(4)==b'\x7fELF'
    except(OSError,ValueError):return False


def command(args,env=None,timeout=60):
    r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,timeout=timeout)
    return dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)


def identity():
    pid=os.getpid();raw=Path('/proc/self/stat').read_text();tail=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,starttick=int(tail[19]),cmdline=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def maps():
    raw=Path('/proc/self/maps').read_text();rows=[]
    for line in raw.splitlines():
        fields=line.split(None,5)
        if len(fields)<6 or not fields[5].startswith('/'):continue
        path=fields[5]
        if path.endswith(' (deleted)'):raise RuntimeError('Deleted mapped file: '+line)
        p=Path(path)
        if not elf(p):continue
        st=p.stat();major,minor=(int(x,16)for x in fields[3].split(':'))
        if(st.st_ino!=int(fields[4])or os.major(st.st_dev)!=major or os.minor(st.st_dev)!=minor):
            raise RuntimeError('Native map device/inode mismatch: '+line)
        rows.append(dict(path=str(p.resolve()),mapped_path=path,inode=st.st_ino,device=st.st_dev,
                         map_device=fields[3],map_inode=int(fields[4])))
    unique={x['path']:x for x in rows}
    return raw,[unique[p]for p in sorted(unique)]


def dynamic(path):
    r=command(['readelf','-d',str(path)])
    if r['returncode']:raise RuntimeError('readelf failed '+str(path)+': '+r['stderr'])
    return dict(DT_NEEDED=re.findall(r'\(NEEDED\).*?\[(.*?)\]',r['stdout']),
                SONAME=re.findall(r'\(SONAME\).*?\[(.*?)\]',r['stdout']),
                RUNPATH_RPATH=[line.strip()for line in r['stdout'].splitlines()if '(RUNPATH)'in line or '(RPATH)'in line],
                raw_sha256=hashlib.sha256(r['stdout'].encode()).hexdigest())


def normname(name):return re.sub(r'[-_.]+','-',name).lower()


def run(base,base_sha):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Explicit empty CUDA_VISIBLE_DEVICES required')
    started=time.monotonic();stages=[];torch=None
    recipe=['torch','torchvision','transformers','transformers.models.qwen3_5.modeling_qwen3_5',
            'numpy','tokenizers','safetensors.torch','PIL.Image']
    for name in recipe:
        m=importlib.import_module(name)
        if name=='torch':torch=m
        initialized=bool(torch and torch.cuda.is_initialized())
        if initialized:raise RuntimeError('CPU probe unexpectedly initialized CUDA')
        raw,actual=maps()
        stages.append(dict(module=name,module_file=getattr(m,'__file__',None),
                           CUDA_initialized=initialized,mapped_native_count=len(actual)))
    rawmaps,actual_maps=maps()
    prefix=Path(sys.prefix).resolve()
    # Index exact wheel RECORD membership and conda metadata. Inventory selection
    # is explicit; imports' native providers are added by exact mapped-file owners.
    all_dists=list(md.distributions());wheel_index={};dist_meta={};selected=set()
    explicit={'torch','torchvision','triton','numpy','tokenizers','safetensors','pillow'}
    for dist in all_dists:
        name=normname(dist.metadata.get('Name',''));version=dist.version
        key=name+'=='+version
        meta=dict(distribution=name,version=version,metadata_root=str(dist._path))
        dist_meta[key]=(dist,meta)
        if name in explicit or name.startswith('nvidia-'):selected.add(key)
        for entry in dist.files or []:
            rel=str(entry)
            if '.so'not in rel and not rel.endswith(('/python','/python3','/python3.10')):continue
            path=Path(dist.locate_file(entry))
            if not path.exists():continue
            canonical=str(path.resolve())
            row=dict(meta,entry=rel,RECORD_hash=None if entry.hash is None else dict(mode=entry.hash.mode,value=entry.hash.value),
                     RECORD_size=entry.size)
            wheel_index.setdefault(canonical,[]).append((key,row))
    conda_index={};conda_packages={}
    for meta_path in sorted((prefix/'conda-meta').glob('*.json')):
        data=json.loads(meta_path.read_text());package=dict(name=data.get('name'),version=data.get('version'),build=data.get('build'),
          metadata_path=str(meta_path),metadata_sha256=digest(meta_path),url=data.get('url'),channel=data.get('channel'))
        conda_packages[str(meta_path)]=data
        details={x['_path']:x for x in data.get('paths_data',{}).get('paths',[])if '_path'in x}
        for rel in data.get('files',[]):
            if '.so'not in rel and rel not in('bin/python','bin/python3','bin/python3.10'):continue
            p=prefix/rel
            if p.exists():conda_index.setdefault(str(p.resolve()),[]).append(dict(package,entry=rel,record=details.get(rel)))
    for row in actual_maps:
        for key,_ in wheel_index.get(row['path'],[]):selected.add(key)
    candidates={};aliases={}
    def add(path,origin):
        p=Path(path)
        if not p.exists()or not elf(p):return
        canonical=str(p.resolve());candidates.setdefault(canonical,[])
        if origin not in candidates[canonical]:candidates[canonical].append(origin)
        if str(p)!=canonical:aliases[str(p)]=canonical
    for row in base['files']:
        add(row.get('resolved_path',row['path']),dict(kind='original_61_external_binding',path=row['path'],sha256=row['sha256']))
    for row in actual_maps:add(row['path'],dict(kind='two_fresh_CPU_import_recipe_map_candidate',mapped_path=row['mapped_path']))
    for path,entries in wheel_index.items():
        for key,entry in entries:
            if key in selected:add(path,dict(kind='explicit_installed_runtime_provider_ELF',distribution=key))
    # Original conda CUDA provider packages may coexist with wheel providers.
    # Pin their exact ELF file inventory as well; never allow an unlisted prefix.
    conda_provider_names={'cuda-cudart','cuda-cudart-dev','cuda-nvrtc','cuda-nvrtc-dev','cuda-cupti','cuda-nvtx',
                          'libcublas','libcusolver','libcusparse','libcurand','libnvjitlink','cudnn','nccl'}
    for path,entries in conda_index.items():
        if any(e['name']in conda_provider_names for e in entries):add(path,dict(kind='explicit_conda_CUDA_runtime_provider_ELF'))
    # Discover the installed driver package from the frozen driver file; include
    # that package's exact native components, covering driver lazy JIT providers.
    driver_rows=[x for x in base['files']if '/libcuda.so' in x['path']]
    driver_packages=[]
    for row in driver_rows:
        qr=command(['rpm','-qf',row['path']])
        if qr['returncode']==0:
            for package in qr['stdout'].splitlines():
                driver_packages.append(package);listing=command(['rpm','-ql',package])
                if listing['returncode']:raise RuntimeError('RPM driver file inventory failed')
                for path in listing['stdout'].splitlines():add(path,dict(kind='exact_installed_driver_package_ELF',package=package))
    driver_kernel_version=Path('/proc/driver/nvidia/version').read_text()
    driver_version=re.search(r'Kernel Module\s+(\d+\.\d+\.\d+)',driver_kernel_version)
    if not driver_version:raise RuntimeError('Cannot identify actual installed driver version')
    driver_version=driver_version.group(1)
    ldconfig=command(['/sbin/ldconfig','-p'])
    if ldconfig['returncode']:raise RuntimeError('Cannot inventory exact driver provider paths')
    manual_driver_paths=set()
    for line in ldconfig['stdout'].splitlines():
        m=re.match(r'\s*(lib(?:cuda|nvidia-[\w-]+)\.so[^ ]*)\s+.*=>\s+(/\S+)',line)
        if not m:continue
        soname,target=m.groups();p=Path(target)
        if p.exists()and p.resolve().name.endswith('.'+driver_version)and elf(p):
            canonical=str(p.resolve());manual_driver_paths.add(canonical)
            add(target,dict(kind='exact_installed_NVIDIA_driver_provider',SONAME=soname,
                driver_version=driver_version,driver_kernel_version_sha256=hashlib.sha256(driver_kernel_version.encode()).hexdigest()))
    libdirs=sorted({str(Path(p).parent)for p in candidates})
    env=dict(os.environ,LD_LIBRARY_PATH=':'.join(libdirs+([os.environ['LD_LIBRARY_PATH']]if os.environ.get('LD_LIBRARY_PATH')else[])))
    queue=sorted(candidates);processed=set();ldd_records=[];dynamic_records={};unresolved=[]
    while queue:
        p=queue.pop(0)
        if p in processed:continue
        processed.add(p);dyn=dynamic(p);dynamic_records[p]=dyn
        # A static ELF executable legitimately reports no dynamic dependencies.
        lr=command(['ldd',p],env=env)
        resolved=[]
        if 'not found'in lr['stdout']:unresolved.append(dict(path=p,stdout=lr['stdout'],stderr=lr['stderr']))
        for line in lr['stdout'].splitlines():
            match=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
            direct=re.match(r'\s*(/\S+)\s+\(',line)
            if match:soname,target=match.groups()
            elif direct:target=direct.group(1);soname=Path(target).name
            else:continue
            t=Path(target)
            if not t.exists():raise RuntimeError('ldd emitted missing path '+target)
            canonical=str(t.resolve());resolved.append(dict(SONAME=soname,path=canonical,resolved_loader_path=target))
            was=canonical in candidates
            add(target,dict(kind='ELF_loader_dependency',needed_by=p,SONAME=soname))
            if not was and canonical in candidates:queue.append(canonical)
        ldd_records.append(dict(path=p,returncode=lr['returncode'],resolved=resolved,
                                stdout_sha256=hashlib.sha256(lr['stdout'].encode()).hexdigest(),stderr=lr['stderr']))
        if lr['returncode']and dyn['DT_NEEDED']and 'not found'not in lr['stdout']:
            raise RuntimeError('ldd failed for dynamic ELF '+p+': '+lr['stderr'])
    optional_audio=[];required_unresolved=[]
    unavailable_audio_names={'_torchaudio_sox.so','libtorchaudio_sox.so','_torio_ffmpeg4.so',
                            '_torio_ffmpeg5.so','libtorio_ffmpeg4.so','libtorio_ffmpeg5.so'}
    mapped_paths={x['path']for x in actual_maps}
    for row in unresolved:
        if Path(row['path']).name in unavailable_audio_names and row['path']not in mapped_paths:
            optional_audio.append(dict(row,scope='UNAVAILABLE_OPTIONAL_AUDIO_PLUGIN_NOT_QWEN_RUNTIME',
                no_runtime_mapping_in_CPU_recipe=True,exact_file_still_pinned=True))
        else:required_unresolved.append(row)
    if required_unresolved:raise RuntimeError('Unresolved required native loader dependencies: '+json.dumps(required_unresolved))
    files=[];provenance_fail=[]
    for path in sorted(candidates):
        p=Path(path);before=p.stat();sha=digest(p);after=p.stat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if fields(before)!=fields(after):raise RuntimeError('ELF changed while hashing '+path)
        wheel=[]
        for _,entry in wheel_index.get(path,[]):
            e=dict(entry);h=e['RECORD_hash']
            if h and h['mode']=='sha256':
                expected=base64.urlsafe_b64decode(h['value']+'='*((-len(h['value']))%4)).hex()
                e['RECORD_sha256_matches_actual']=expected==sha
                if expected!=sha:raise RuntimeError('Installed wheel RECORD checksum mismatch: '+path)
            elif h:e['RECORD_hash_supported']=False
            if e['RECORD_size']is not None and int(e['RECORD_size'])!=after.st_size:raise RuntimeError('RECORD size mismatch '+path)
            wheel.append(e)
        conda=[]
        for entry in conda_index.get(path,[]):
            e=dict(entry);record=e.get('record')or{};expected=record.get('sha256')
            # Prefix-replaced binaries may differ from package bytes; retain this
            # distinction instead of claiming an installed hash verification.
            if expected:e['package_archive_sha256_matches_installed']=expected==sha
            conda.append(e)
        rpm=None;manual_system_driver=None
        base_rows=[x for x in base['files']if str(Path(x.get('resolved_path',x['path'])).resolve())==path]
        if not wheel and not conda:
            rr=command(['rpm','-qf',path])
            if rr['returncode']==0:
                rpm=dict(packages=rr['stdout'].splitlines(),ownership_command_returncode=0)
            elif path in manual_driver_paths or any('/libcuda.so'in x['path']for x in base_rows):
                note=command(['readelf','-n',path])
                manual_system_driver=dict(RPM_managed=False,RPM_query=rr,
                    identity_basis='Exact installed driver SONAME/version with original immutable libcuda binding and running kernel driver version',
                    driver_version=driver_version,kernel_version_text=driver_kernel_version,
                    ELF_build_ids=re.findall(r'Build ID: ([0-9a-f]+)',note['stdout']),
                    no_claim_of_distribution_package_hash_verification=True)
            else:provenance_fail.append(dict(path=path,error=rr['stderr']or rr['stdout']))
        for old in base_rows:
            if old['sha256']!=sha or old['bytes']!=after.st_size:raise RuntimeError('Original immutable native binding changed '+path)
        provenance=dict(wheel_RECORD=wheel,conda_metadata=conda,system_RPM=rpm,manual_system_driver=manual_system_driver,
                        origins=candidates[path],ELF=dynamic_records[path],
                        original_61_binding_verified=bool(base_rows))
        files.append(dict(path=path,bytes=after.st_size,inode=after.st_ino,device=after.st_dev,mtime_ns=after.st_mtime_ns,
                          sha256=sha,provenance=provenance))
    if provenance_fail:raise RuntimeError('Unprovenanced native file(s): '+json.dumps(provenance_fail))
    nvml=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])
    if nvml['returncode']:raise RuntimeError('Cannot check CPU PID absence in NVML')
    if any(line.split(',')[0].strip()==str(os.getpid())for line in nvml['stdout'].splitlines()):raise RuntimeError('CPU probe PID unexpectedly in NVML')
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before receipt')
    return dict(schema_version=1,status='PASS_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU',
        identity=identity(),host=socket.gethostname(),python=sys.executable,base_external_bindings_sha256=base_sha,
        import_recipe=recipe,import_stages=stages,files=files,count=len(files),bytes=sum(x['bytes']for x in files),
        aliases=[dict(path=p,canonical_path=aliases[p])for p in sorted(aliases)],
        selected_installed_distributions=[dist_meta[k][1]for k in sorted(selected)],driver_packages=driver_packages,
        driver_kernel_version=driver_kernel_version,driver_provider_ldconfig=ldconfig,
        recursive_ELF_loader_records=ldd_records,ldd_environment={'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']},
        mapped_native_files=actual_maps,raw_proc_self_maps=rawmaps,unresolved_required_ELF_dependencies=[],
        unavailable_optional_audio_plugins=optional_audio,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),Torch_CUDA_initialized=False,
        NVML_apps_raw=nvml['stdout'],CPU_PID_absent_from_NVML=True,model_instances_created=0,
        Tensor_forwards=0,GPU_dispatches=0,GPU_contexts_created=0,
        scientific_or_runtime_files_modified=False,not_GPU_qualification=True,
        started_monotonic=started,finished_monotonic=time.monotonic())


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-json',required=True);a=ap.parse_args()
    raw=sys.stdin.buffer.read()if a.base_json=='-'else Path(a.base_json).read_bytes()
    result=run(json.loads(raw),hashlib.sha256(raw).hexdigest())
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
