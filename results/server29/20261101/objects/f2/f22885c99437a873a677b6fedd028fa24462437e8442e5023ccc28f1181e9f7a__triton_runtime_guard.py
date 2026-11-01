"""Once-per-worker source/native/compiler pin check before Triton JIT.

This guard reads metadata and bytes only. It does not initialize CUDA, import
torch/Triton, compile, launch, signal, mutate science, or write a receipt.
The admitted worker may persist the returned dependency receipt.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil

EXPECTED_MANIFEST_SHA256='bc695cc73e549272d9231cfe1f6b833dbf3519c9964e3820746acfee92dcafe5'
_SKIP_DIRS={'__pycache__','.git','.cache','cache','caches'}
_CACHED=None


def _require(value,message):
    if not value:raise RuntimeError('Triton dependency guard: '+message)


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _check_file(row):
    path=Path(row['path'])
    _require(path.is_file(),'missing regular file '+str(path))
    before=path.stat()
    _require(before.st_ino==row['inode'] and before.st_dev==row['device'] and
        before.st_size==row['bytes'] and before.st_mtime_ns==row['mtime_ns'],
        'stat identity differs '+str(path))
    if 'mode' in row:
        _require(oct(before.st_mode)==row['mode'],'file mode differs '+str(path))
    if 'resolved_path' in row:
        _require(str(path.resolve())==row['resolved_path'],'resolved path differs '+str(path))
    actual=_sha(path);after=path.stat()
    _require((before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)==
             (after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns),
             'file changed while hashing '+str(path))
    _require(actual==row['sha256'],'SHA differs '+str(path))
    return dict(path=str(path),sha256=actual,inode=after.st_ino,
                bytes=after.st_size,mtime_ns=after.st_mtime_ns)


def _check_environment(root,manifest):
    expected_cache=(root/'engineering/triton_cache').resolve()
    _require(expected_cache.is_relative_to(root),'task cache escapes task root')
    cache=os.environ.get('TRITON_CACHE_DIR')
    _require(cache is not None and Path(cache).resolve()==expected_cache,
             'TRITON_CACHE_DIR must name this task engineering/triton_cache')
    for name,expected in manifest['observed_package_environment'].items():
        if name=='TRITON_CACHE_DIR':continue
        _require(os.environ.get(name)==expected,'unfrozen override environment '+name)
    # Installed compiler.py derives these names dynamically; do not rely only
    # on literal getenv calls discovered in Python sources.
    for name in ('TRITON_PTXAS_PATH','TRITON_NVDISASM_PATH','TRITON_CUOBJDUMP_PATH'):
        _require(name in manifest['observed_package_environment'],
                 'manifest omits dynamic compiler override '+name)
    chosen=os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    _require(chosen is not None,'host compiler unavailable')
    if not Path(chosen).is_absolute():chosen=shutil.which(chosen)
    driver=next(x for x in manifest['host_c_compiler'] if x['role']=='driver')
    _require(Path(chosen).resolve()==Path(driver['resolved_path']),
             'host compiler selected by PATH differs')
    return dict(TRITON_CACHE_DIR=str(expected_cache),frozen_override_variables=
                dict(manifest['observed_package_environment']),host_compiler=chosen)


def verify_before_import(root):
    global _CACHED
    root=Path(root).resolve()
    path=root/'analysis/TRITON_RUNTIME_FRESH_SHA.json'
    digest=_sha(path)
    _require(digest==EXPECTED_MANIFEST_SHA256,'manifest SHA differs')
    if _CACHED is not None:
        _require(_CACHED['task_root']==str(root),'guard reused across task roots')
        # Check environment on repeated access; large package byte audit is
        # intentionally once per fresh worker, before its first JIT compile.
        _check_environment(root,json.loads(path.read_text()))
        return dict(_CACHED)
    manifest=json.loads(path.read_text())
    _require(manifest['status']=='PASS_FRESH_CPU_READONLY_TRITON_PACKAGE_SHA',
             'manifest is not an actual fresh dependency inventory')
    _require(manifest['distribution_version']=='3.1.0','frozen Triton version differs')
    package=Path(manifest['package_root']).resolve()
    expected={x['path'] for x in manifest['files']};observed=set()
    for dirname,dirs,files in os.walk(package,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in _SKIP_DIRS)
        _require(not any((Path(dirname)/d).is_symlink() for d in dirs),
                 'unfrozen symlink package directory')
        for name in files:
            p=Path(dirname)/name
            if p.suffix not in {'.pyc','.pyo'}:observed.add(str(p))
    _require(observed==expected,'package file roster differs')
    _require(len(expected)==manifest['count'] and len(expected)<=4000,
             'package count differs')
    _require(sum(x['bytes'] for x in manifest['files'])==manifest['total_bytes']
             <=1024**3,'package byte bound/count differs')
    environment=_check_environment(root,manifest)
    verified=[_check_file(x) for x in manifest['files']]
    metadata=[_check_file(x) for x in manifest['distribution_metadata']]
    host_compiler=[_check_file(x) for x in manifest['host_c_compiler']]
    _require(importlib.metadata.version('triton')==manifest['distribution_version'],
             'installed distribution metadata version differs')
    required={x['relative_path'] for x in manifest['files']}
    for mandatory in ('_C/libtriton.so','backends/nvidia/bin/ptxas',
        'backends/nvidia/lib/libdevice.10.bc','language/extra/cuda/libdevice.py'):
        _require(mandatory in required,'required compiler/math dependency absent '+mandatory)
    _CACHED=dict(status='PASS_FRESH_TRITON_DEPENDENCIES_BEFORE_IMPORT',task_root=str(root),
        manifest_sha256=digest,package_root=str(package),version=manifest['distribution_version'],
        files_verified=len(verified),bytes_verified=manifest['total_bytes'],
        actual_verified_inventory_sha256=hashlib.sha256(json.dumps(verified,sort_keys=True,
            separators=(',',':')).encode()).hexdigest(),
        distribution_metadata_verified=len(metadata),host_c_compiler_verified=host_compiler,
        environment=environment,source_native_compiler_bytes_fresh=True,
        gpu_qualification_credit=0,scope='dependency provenance only; no CUDA kernel/model/math claim')
    return dict(_CACHED)


def verify_after_import(root,triton_module):
    global _CACHED
    receipt=verify_before_import(root)
    package=Path(receipt['package_root'])
    _require(Path(triton_module.__file__).resolve()==package/'__init__.py',
             'actual imported Triton path differs')
    _require(triton_module.__version__==receipt['version'],'actual imported version differs')
    manifest=json.loads((Path(root)/'analysis/TRITON_RUNTIME_FRESH_SHA.json').read_text())
    files={x['resolved_path']:x for x in manifest['files']}
    loaded={}
    maps=Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            parts=line.split(None,5)
            if len(parts)<6:continue
            name=parts[5]
            if name.startswith(str(package)+'/') and '.so' in name:
                _require(not name.endswith(' (deleted)'),'actual mapped Triton file deleted')
                resolved=str(Path(name).resolve());row=files.get(resolved)
                _require(row is not None and int(parts[4])==row['inode'],
                         'actual mapped native inode differs '+name)
                loaded[resolved]=dict(path=resolved,inode=int(parts[4]),sha256=row['sha256'])
    _CACHED=dict(receipt,status='PASS_FRESH_TRITON_DEPENDENCIES_ACTUAL_IMPORT',
        actual_triton_file=str(Path(triton_module.__file__).resolve()),
        actual_triton_version=triton_module.__version__,actual_mapped_native=list(loaded.values()))
    return dict(_CACHED)


def dependency_receipt():
    return None if _CACHED is None else json.loads(json.dumps(_CACHED))
