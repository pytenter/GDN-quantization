"""Pin and preload the existing CUPTI ELF before CUDA Driver initialization.

The installed Driver-first/CUPTI-later order reproducibly faults from libcudart
exit callbacks. No CUDA/CUPTI API is called here and no library is replaced.
The normal interpreter/driver teardown remains enabled; actual exit is required.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import time

_HELD_LIBRARIES={}

def preload_cupti_before_driver(root):
    root=Path(root)
    manifest_path=root/'analysis/RUNTIME_NATIVE_CLOSURE.json'
    manifest=json.loads(manifest_path.read_text())
    rows=[row for row in manifest['files']if Path(row['path']).name=='libcupti.so.12']
    if len(rows)!=1:raise RuntimeError('Exactly one pre-frozen canonical CUPTI ELF is required')
    row=rows[0];path=Path(row['path']);before=path.stat()
    if not path.is_file()or path.is_symlink()or str(path.resolve())!=str(path):
        raise RuntimeError('CUPTI must be the exact existing canonical frozen file')
    expected=(row['bytes'],row['inode'],row['device'],row['mtime_ns'])
    actual=(before.st_size,before.st_ino,before.st_dev,before.st_mtime_ns)
    if actual!=expected:raise RuntimeError('CUPTI file identity changed before load-order repair')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=row['sha256']:raise RuntimeError('CUPTI exact frozen bytes changed')
    loaded_before=str(path)in _HELD_LIBRARIES
    if not loaded_before:
        _HELD_LIBRARIES[str(path)]=ctypes.CDLL(str(path),mode=ctypes.RTLD_GLOBAL)
    after=path.stat()
    if (after.st_size,after.st_ino,after.st_dev,after.st_mtime_ns)!=expected:
        raise RuntimeError('CUPTI identity changed across ordinary ELF preload')
    return dict(status='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL',
        path=str(path),bytes=row['bytes'],sha256=digest,inode=row['inode'],device=row['device'],mtime_ns=row['mtime_ns'],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),already_held=loaded_before,
        CUDA_Driver_API_calls=0,CUPTI_API_calls=0,model_forward_calls=0,QDQ_calls=0,
        normal_process_teardown_required=True,monotonic=time.monotonic())
