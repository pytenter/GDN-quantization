"""Pin and preload the existing CUPTI ELF before CUDA Driver/Torch initialization.

Only native library load order changes. No CUDA/CUPTI API is called by preload,
no file is replaced, and normal process/interpreter teardown remains required.
"""
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

_HELD_LIBRARIES={}

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def _exact_file(row,basename):
    path=Path(row['path']);before=path.stat()
    if path.name!=basename or not path.is_file() or path.is_symlink() or str(path.resolve())!=str(path):
        raise RuntimeError('Exact existing canonical CUPTI file required')
    expected=(row['bytes'],row['inode'],row['device'],row['mtime_ns'])
    actual=(before.st_size,before.st_ino,before.st_dev,before.st_mtime_ns)
    if actual!=expected or _sha(path)!=row['sha256']:
        raise RuntimeError('Exact frozen CUPTI ELF/header identity changed')
    after=path.stat()
    if (after.st_size,after.st_ino,after.st_dev,after.st_mtime_ns)!=expected:
        raise RuntimeError('CUPTI identity changed during fresh SHA')
    return path,expected

def preload_cupti_before_driver(root):
    root=Path(root).resolve();manifest_path=root/'analysis/CUPTI_LIFECYCLE_BINDING.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS' or manifest['execution_root']!=str(root) or len(manifest['files'])!=1:
        raise RuntimeError('Exact current CUPTI lifecycle binding scope required')
    row=manifest['files'][0];path,expected=_exact_file(row,'libcupti.so.12');_exact_file(manifest['header'],'cupti_activity.h')
    loaded_before=str(path) in _HELD_LIBRARIES
    if not loaded_before:
        # A first preload after Driver/runtime load cannot repair load order.
        maps=Path('/proc/self/maps').read_text()
        names=[Path(line.split(maxsplit=5)[5]).name for line in maps.splitlines() if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[5].startswith('/')]
        if any(name.startswith(('libcuda.so','libcudart.so','libcupti.so')) for name in names):
            raise RuntimeError('CUPTI preload must occur before any Driver/runtime/CUPTI ELF load')
        _HELD_LIBRARIES[str(path)]=ctypes.CDLL(str(path),mode=ctypes.RTLD_GLOBAL)
    after=path.stat()
    if (after.st_size,after.st_ino,after.st_dev,after.st_mtime_ns)!=expected:
        raise RuntimeError('CUPTI identity changed across ordinary ELF preload')
    return dict(status='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL',path=str(path),bytes=row['bytes'],sha256=row['sha256'],inode=row['inode'],device=row['device'],mtime_ns=row['mtime_ns'],manifest_sha256=_sha(manifest_path),already_held=loaded_before,CUDA_Driver_API_calls=0,CUPTI_API_calls=0,model_forward_calls=0,QDQ_calls=0,normal_process_teardown_required=True,monotonic=time.monotonic())
