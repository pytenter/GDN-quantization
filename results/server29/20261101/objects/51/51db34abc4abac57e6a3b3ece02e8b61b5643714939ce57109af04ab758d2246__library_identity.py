"""Durable, exhaustive loaded-library identity gates; stdlib only.

The selected roles retain the previous worker's strict prefixes, including
libtorchaudio.so under libtorch*. All map lines and all selected failures are
retained before an exception. No library is imported or initialized here.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

SCHEMA='QWEN_LOADED_LIBRARY_IDENTITY_V2'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def classify_library(name):
    base=Path(name).name
    return ('torch' if base.startswith(('libtorch','libc10')) else
            'cuBLAS' if base.startswith('libcublas') else
            'cuDNN' if base.startswith('libcudnn') else
            'CUDA_runtime' if base.startswith('libcudart') else
            'driver' if base.startswith('libcuda.so') else None)

def atomic_bytes_new(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
    with temporary.open('xb')as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)

def file_receipt(path,phase,raw):
    return dict(path=str(Path(path).relative_to(phase)),bytes=len(raw),sha256=digest(raw))

def metadata(stat):
    return dict(device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)

def stable_file_identity(path,progress=None):
    path=Path(path);resolved=path.resolve(strict=True);before=metadata(resolved.stat())
    h=hashlib.sha256();done=0
    with resolved.open('rb')as f:
        if metadata(os.fstat(f.fileno()))!=before:raise RuntimeError('File identity changed before open')
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block);done+=len(block)
            if progress is not None and done%(64*1024**2)==0:progress(str(resolved),done)
        if metadata(os.fstat(f.fileno()))!=before:raise RuntimeError('File identity changed while hashing')
    after=metadata(resolved.stat())
    if before!=after or path.resolve(strict=True)!=resolved:raise RuntimeError('File identity changed after hashing')
    return dict(path=str(resolved),bytes=before['bytes'],sha256=h.hexdigest(),
                metadata_before=before,metadata_after=after,symlink=path.is_symlink())

def parse_maps(text):
    records=[]
    for lineno,line in enumerate(text.splitlines(),1):
        fields=line.split(None,5)
        row=dict(line=lineno,raw_line=line)
        if len(fields)>=5:
            row.update(address=fields[0],permissions=fields[1],offset=fields[2],device=fields[3],inode_text=fields[4])
        if len(fields)==6:
            raw=fields[5];deleted=raw.endswith(' (deleted)');clean=raw[:-10]if deleted else raw
            row.update(raw_path=raw,deleted=deleted)
            if clean.startswith('/'):
                try:resolved=str(Path(clean).resolve());error=None
                except Exception as exc:resolved=clean;error=type(exc).__name__+':'+str(exc)
                role=classify_library(resolved) or classify_library(clean)
                row.update(path=clean,resolved_path=resolved,role=role,resolve_error=error)
                try:row['inode']=int(fields[4])
                except ValueError:row['inode']=None
        records.append(row)
    return records

def selected_map_signature(records):
    return sorted(set((r['resolved_path'],r.get('device'),r.get('inode')if type(r.get('inode'))is int else -1,r.get('deleted',False))
                      for r in records if r.get('role')is not None))

class LibraryIdentityError(RuntimeError):
    def __init__(self,summary,report):
        self.summary=summary;self.report=report
        paths=report['missing_bindings']
        message=('Unbound actual loaded libraries:'+','.join(paths))if paths else 'Loaded library identity gate failed:'+','.join(sorted(set(x['kind']for x in report['issues'])))
        super().__init__(message)

def library_identity_gate(contract,phase,stage,*,maps_reader=None,progress=None,context=None):
    """Publish full maps+all issues, then PASS or raise with durable summary.

    PREFLIGHT requires torch only, allowing cuBLAS/cuDNN to load lazily without
    any synthetic model forward. POSTFLIGHT preserves torch+cuBLAS requirements.
    The previous driver exception is retained; every actual driver is still
    hashed and checked against its binding whenever such a binding exists.
    """
    if stage not in ('PREFLIGHT','POSTFLIGHT'):raise ValueError('Unknown identity gate stage')
    phase=Path(phase);directory=phase/'library_identity'
    reader=maps_reader if maps_reader is not None else lambda:Path('/proc/self/maps').read_text()
    issues=[];read_errors=[]
    def capture(label):
        try:return reader()
        except Exception as exc:
            message=type(exc).__name__+':'+str(exc);read_errors.append(dict(stage=label,error=message));issues.append(dict(kind='MAPS_READ_ERROR',observation=label,error=message));return ''
    first=capture('INITIAL');maps_path=directory/(stage+'_MAPS.txt');raw=first.encode()
    atomic_bytes_new(maps_path,raw);maps_receipt=file_receipt(maps_path,phase,raw)
    initial_records=parse_maps(first)
    allowed={};incomplete=[];binding_rows=contract['prospective_library_bindings']
    for row in binding_rows:
        if 'path'not in row or 'sha256'not in row:
            incomplete.append(row);continue
        try:name=str(Path(row.get('resolved_path',row['path'])).resolve())
        except Exception as exc:
            issues.append(dict(kind='BINDING_PATH_RESOLVE_ERROR',binding=row,error=type(exc).__name__+':'+str(exc)));continue
        if name in allowed and any(allowed[name][key]!=row[key]for key in ('bytes','sha256')):
            issues.append(dict(kind='CONFLICTING_BINDINGS',path=name,bindings=[allowed[name],row]))
        allowed[name]=row
    by_path={}
    for row in initial_records:
        if row.get('role')is not None:by_path.setdefault(row['resolved_path'],[]).append(row)
    identities=[];details=[];missing=[];mismatches=[]
    for name,segments in sorted(by_path.items()):
        role=next(x['role']for x in segments);binding=allowed.get(name)
        detail=dict(path=name,role=role,mapped_segments=segments,binding=binding)
        if binding is None and role!='driver':missing.append(name);issues.append(dict(kind='MISSING_BINDING',path=name,role=role))
        if any(x.get('deleted')for x in segments):issues.append(dict(kind='DELETED_MAPPED_LIBRARY',path=name))
        try:
            actual=stable_file_identity(name,progress);detail['actual']=actual
            identities.append(dict(path=name,bytes=actual['bytes'],sha256=actual['sha256'],role=role))
            stat=actual['metadata_before']
            for segment in segments:
                try:
                    major,minor=(int(x,16)for x in segment['device'].split(':'))
                    same=(segment['inode']==stat['inode']and (major,minor)==(os.major(stat['device']),os.minor(stat['device'])))
                except Exception:same=False
                if not same:issues.append(dict(kind='MAPPED_FILE_DEVICE_INODE_MISMATCH',path=name,line=segment['line'],mapped_device=segment.get('device'),mapped_inode=segment.get('inode'),current_file_metadata=stat))
            if binding is not None:
                changed=[]
                for key in ('bytes','sha256'):
                    if actual[key]!=binding[key]:changed.append(key)
                if changed:
                    mismatch=dict(path=name,fields=changed,expected={k:binding[k]for k in ('bytes','sha256')},actual={k:actual[k]for k in ('bytes','sha256')})
                    mismatches.append(mismatch);issues.append(dict(kind='BOUND_LIBRARY_BYTES_SHA_MISMATCH',**mismatch))
        except Exception as exc:
            detail['hash_error']=type(exc).__name__+':'+str(exc);issues.append(dict(kind='LIBRARY_HASH_ERROR',path=name,error=detail['hash_error']))
        details.append(detail)
    final=capture('FINAL');final_path=directory/(stage+'_MAPS_FINAL.txt');final_raw=final.encode()
    atomic_bytes_new(final_path,final_raw);final_receipt=file_receipt(final_path,phase,final_raw)
    final_records=parse_maps(final)
    if selected_map_signature(initial_records)!=selected_map_signature(final_records):
        issues.append(dict(kind='SELECTED_MAP_IDENTITY_CHANGED_DURING_GATE',initial=selected_map_signature(initial_records),final=selected_map_signature(final_records)))
    # Enumerate every missing selected path from both observations, including
    # a library that appeared while the initial selected files were hashed.
    for row in final_records:
        if row.get('role')is not None and row['role']!='driver'and row['resolved_path']not in allowed and row['resolved_path']not in missing:
            missing.append(row['resolved_path']);issues.append(dict(kind='MISSING_BINDING',path=row['resolved_path'],role=row['role'],first_seen='FINAL'))
    required=('torch',)if stage=='PREFLIGHT'else ('torch','cuBLAS')
    roles={x['role']for x in identities}
    for role in required:
        if role not in roles:issues.append(dict(kind='REQUIRED_RUNTIME_ROLE_ABSENT',role=role))
    identities.sort(key=lambda x:x['path']);identity_digest=digest(canonical(identities));status='FAIL'if issues else 'PASS'
    report=dict(schema=SCHEMA,stage=stage,status=status,identity_digest=identity_digest,identities=identities,
                maps_receipt=maps_receipt,maps_final_receipt=final_receipt,
                all_mapped_paths=initial_records,all_mapped_paths_final=final_records,
                selected_library_details=details,missing_bindings=sorted(missing),mismatches=mismatches,issues=issues,
                incomplete_prospective_binding_rows=incomplete,required_roles=list(required),
                context=context or{},unix=time.time(),pid=os.getpid(),maps_read_errors=read_errors,
                selected_roles_policy='Original libtorch*/libc10/libcublas/libcudnn/libcudart/libcuda.so prefixes retained; deleted selected mappings fail rather than silently skip.',
                driver_policy='Original unbound-driver exception retained; current bytes/SHA+mapped inode recorded; bound driver must match.',
                no_scientific_forward_or_library_initialization=True)
    identity_path=directory/(stage+'_IDENTITY.json');encoded=canonical(report)+b'\n'
    atomic_bytes_new(identity_path,encoded)
    summary=dict(schema=SCHEMA,stage=stage,status=status,identity_digest=identity_digest,identities=identities,
                 maps_receipt=maps_receipt,maps_final_receipt=final_receipt,
                 identity_receipt=file_receipt(identity_path,phase,encoded))
    if issues:raise LibraryIdentityError(summary,report)
    return summary
