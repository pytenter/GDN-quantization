"""ASYM32_CANONICAL_V2 CPU implementation and explicitly authorized CUDA binding.

Import is CPU-only and does not import torch, discover devices, compile CUDA, or
load a CUDA library. CPU native compilation is explicit/lazy on a CPU encode.
CUDA code is a separately built artifact; loading/execution requires a fresh
explicit authorization argument. No model or KDA kernel is implemented here.
"""
from __future__ import annotations
import ctypes
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np

VERSION = 'ASYM32_CANONICAL_V2'
GROUP_SIZE = 32
CPU_FLAGS = ('-std=c++17', '-O2', '-shared', '-fPIC', '-msse2', '-mfpmath=sse',
             '-fno-fast-math', '-ffp-contract=off', '-frounding-math', '-fno-associative-math')
STAGE_GROUP_KEYS = ('S2_min_bits32','S2_max_bits32','S3_span_bits32','S3_fit_bits32',
                    'S4_raw_bits32','S5_scale_bits32','S6_ratio_bits32')
_CPU_LIBRARY = None


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cpu(output_dir=None, compiler=None):
    """Build only the isolated C++ CPU .so and persist actual compiler provenance."""
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('CPU_QUALIFICATION_REQUIRES_LINUX_X86_64_SSE')
    src=Path(__file__).with_name('canonical_cpu.cpp')
    digest=file_sha256(src)
    target=Path(output_dir) if output_dir is not None else src.parent/'cpu_build'/digest
    target.mkdir(parents=True,exist_ok=True)
    so=target/'canonical_cpu.so';receipt=target/'CPU_BUILD_RECEIPT.json'
    compiler=compiler or os.environ.get('CXX','c++')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha256']!=digest or old['compiler_flags']!=list(CPU_FLAGS) or file_sha256(so)!=old['artifact_sha256']:
            raise RuntimeError('CPU_ARTIFACT_IDENTITY_MISMATCH')
        return so
    if so.exists(): raise RuntimeError('UNRECEIPTED_CPU_ARTIFACT_EXISTS')
    version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True)
    fd,tmp=tempfile.mkstemp(prefix='canonical_cpu.',suffix='.so',dir=target);os.close(fd)
    command=[compiler,*CPU_FLAGS,str(src),'-o',tmp]
    try:
        run=subprocess.run(command,capture_output=True,text=True)
        if run.returncode: raise RuntimeError('CPU_COMPILE_FAILED: '+run.stderr)
        os.replace(tmp,so)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    rec=dict(status='CPU_NATIVE_ARTIFACT_BUILT_NOT_GPU_QUALIFICATION',codec=VERSION,
             source=str(src),source_sha256=digest,artifact=str(so),artifact_sha256=file_sha256(so),
             compiler=compiler,compiler_version=version.stdout,compiler_flags=list(CPU_FLAGS),
             compiler_stdout=run.stdout,compiler_stderr=run.stderr,platform=platform.platform(),
             architecture=platform.machine(),GPU_dispatches=0,CUDA_compile='NOT_RUN',
             floating_environment='operation entry saves caller fenv/MXCSR, FE_TONEAREST, FTZ/DAZ off, restores caller on exit')
    with receipt.open('x') as f:json.dump(rec,f,indent=2);f.write('\n')
    return so


def _cpu_library():
    global _CPU_LIBRARY
    if _CPU_LIBRARY is None:
        lib=ctypes.CDLL(str(build_cpu()))
        p=ctypes.c_void_p
        lib.canonical_v2_encode.argtypes=[p,ctypes.c_size_t,p,p,p,p,p,p,p,p,ctypes.c_size_t]
        lib.canonical_v2_encode.restype=ctypes.c_int
        lib.canonical_v2_cpu_contract.restype=ctypes.c_char_p
        if not lib.canonical_v2_cpu_contract().decode().startswith(VERSION+'|RN32|FP16RNE|'):
            raise RuntimeError('WRONG_CPU_CODEC_CONTRACT')
        _CPU_LIBRARY=lib
    return _CPU_LIBRARY


def _input_cpu(value):
    if hasattr(value,'device') and hasattr(value,'detach'):
        if value.device.type!='cpu': raise PermissionError('CPU_ENCODER_REJECTS_NON_CPU_TENSOR')
        if str(value.dtype)!='torch.float32':raise TypeError('CANONICAL_INPUT_REQUIRES_FP32')
        value=value.detach().numpy()
    array=np.asarray(value)
    if array.dtype!=np.dtype('float32'):raise TypeError('CANONICAL_INPUT_REQUIRES_FP32')
    if array.ndim<1 or array.shape[-1]!=128 or array.size==0:
        raise ValueError('VALUE_AXIS_128_FOUR_CONTIGUOUS_GROUPS_REQUIRED')
    if not np.isfinite(array).all():raise ArithmeticError('NONFINITE_CANONICAL_INPUT')
    return np.ascontiguousarray(array)


def _bits(value):return np.ascontiguousarray(value).view(np.uint32).copy()


@dataclasses.dataclass
class CanonicalEncoded:
    codes: object
    scales: object
    zero_points: object
    decoded: object
    stages: dict|None=None
    version: str=VERSION
    group_size: int=32
    metadata_dtype: str='fp16'
    asymmetric: bool=True

    def decode(self, torch=None):
        # Only quantized decoded values are kept; no pre-QDQ continuation state.
        if isinstance(self.decoded,np.ndarray):
            out=self.decoded.copy()
            return out if torch is None else torch.from_numpy(out)
        return self.decoded.clone()

    def metadata_audit(self):
        def nbytes(v):return int(v.nbytes) if isinstance(v,np.ndarray) else int(v.numel()*v.element_size())
        return dict(version=self.version,group_size=32,metadata_dtype='fp16',asymmetric=True,
                    codes_dtype=str(self.codes.dtype),scales_dtype=str(self.scales.dtype),
                    zero_points_dtype=str(self.zero_points.dtype),
                    tensor_storage_bytes={'codes':nbytes(self.codes),'group_scales':nbytes(self.scales),'zero_points':nbytes(self.zero_points)},
                    physical_packed_INT8_persistent=False,unquantized_shadow=False)


def encode(value, capture_stages=False):
    """CPU actual codec. Input FP32 [...,128]; group axes are last-axis chunks32.

    stages are flat group order, with IEEE bit arrays under *_bits32 names;
    S0/S1/S7/S8/S9 shape [number_of_groups,32], group scalars [number_of_groups].
    """
    x=_input_cpu(value);groups=x.size//32
    q=np.empty_like(x,dtype=np.uint8);hs=np.empty(groups,dtype=np.uint16);z=np.empty(groups,dtype=np.uint8);decoded=np.empty_like(x)
    gs=np.empty((groups,7),dtype=np.float32) if capture_stages else None
    levels=np.empty((groups,32),dtype=np.float32) if capture_stages else None
    rounded=np.empty((groups,32),dtype=np.float32) if capture_stages else None
    ptr=lambda a:None if a is None else ctypes.c_void_p(a.ctypes.data)
    error=ctypes.create_string_buffer(512)
    rc=_cpu_library().canonical_v2_encode(ptr(x),groups,ptr(q),ptr(hs),ptr(z),ptr(decoded),ptr(gs),ptr(levels),ptr(rounded),error,len(error))
    if rc:raise ArithmeticError(error.value.decode())
    compact=x.shape[:-1]+(4,)
    stages=None
    if capture_stages:
        stages={'S0_bits32':_bits(x).reshape(groups,32),'S1_bits32':_bits(x).reshape(groups,32)}
        stages.update({key:_bits(gs[:,i]) for i,key in enumerate(STAGE_GROUP_KEYS)})
        stages.update(S5_stored_bits16=hs.copy(),S6_z=z.copy(),S7_levels_bits32=_bits(levels),
                      S7_round_bits32=_bits(rounded),S8_q=q.reshape(groups,32).copy(),S9_decode_bits32=_bits(decoded).reshape(groups,32))
    return CanonicalEncoded(q,hs.view(np.float16).reshape(compact),z.reshape(compact),decoded,stages)


def encode_torch(torch,value,capture_stages=False):
    """CPU tensor adapter; never silently moves CUDA input to CPU."""
    out=encode(value,capture_stages)
    return CanonicalEncoded(torch.from_numpy(out.codes),torch.from_numpy(out.scales),
                            torch.from_numpy(out.zero_points),torch.from_numpy(out.decoded),out.stages)


class AuthorizedCudaBackend:
    """Prebuilt, isolated Codec CUDA kernel binding. No compile or device discovery.

    The authorizing controller must bind .so/source/contract SHA and stage budget.
    Merely constructing this class does not represent numerical qualification.
    """
    def __init__(self,torch,library_path,*,gpu_stage_authorized=False,expected_library_sha256=None):
        if gpu_stage_authorized is not True:raise PermissionError('GPU_STAGE_NOT_AUTHORIZED')
        if expected_library_sha256 is None or file_sha256(library_path)!=expected_library_sha256:
            raise RuntimeError('EXPLICIT_PREBUILT_GPU_ARTIFACT_SHA_REQUIRED')
        self.torch=torch;self.authorized=True
        self.library=ctypes.CDLL(str(library_path))
        self.library.canonical_v2_launch.argtypes=[ctypes.c_void_p,ctypes.c_size_t,*([ctypes.c_void_p]*9)]
        self.library.canonical_v2_launch.restype=ctypes.c_int

    def encode(self,x,capture_stages=True):
        torch=self.torch
        if x.device.type!='cuda' or x.dtype!=torch.float32 or x.shape[-1]!=128:
            raise ValueError('AUTHORIZED_GPU_CODEC_REQUIRES_NATIVE_CUDA_FP32_VALUE128')
        x=x.contiguous();groups=x.numel()//32
        q=torch.empty_like(x,dtype=torch.uint8);hs=torch.empty(x.shape[:-1]+(4,),dtype=torch.uint16,device=x.device)
        z=torch.empty_like(hs,dtype=torch.uint8);decoded=torch.empty_like(x)
        gs=torch.empty((groups,7),dtype=torch.float32,device=x.device) if capture_stages else None
        levels=torch.empty((groups,32),dtype=torch.float32,device=x.device) if capture_stages else None
        rounded=torch.empty((groups,32),dtype=torch.float32,device=x.device) if capture_stages else None
        error=torch.zeros((1,),dtype=torch.int32,device=x.device)
        ptr=lambda v:None if v is None else ctypes.c_void_p(v.data_ptr())
        stream=ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        rc=self.library.canonical_v2_launch(ptr(x),groups,ptr(q),ptr(hs),ptr(z),ptr(decoded),ptr(gs),ptr(levels),ptr(rounded),ptr(error),stream)
        if rc:raise RuntimeError('CANONICAL_CUDA_LAUNCH_ERROR:'+str(rc))
        code=int(error.item())
        if code:raise ArithmeticError('CANONICAL_CUDA_FINITE_GUARD:'+str(code))
        stages=None
        if capture_stages:
            host=lambda t:t.detach().to(device='cpu',copy=True).numpy()
            stages={'S0_bits32':_bits(host(x)).reshape(groups,32),'S1_bits32':_bits(host(x)).reshape(groups,32)}
            gst=host(gs);stages.update({key:_bits(gst[:,i]) for i,key in enumerate(STAGE_GROUP_KEYS)})
            stages.update(S5_stored_bits16=host(hs).copy().reshape(groups),S6_z=host(z).copy().reshape(groups),
                          S7_levels_bits32=_bits(host(levels)),S7_round_bits32=_bits(host(rounded)),
                          S8_q=host(q).reshape(groups,32),S9_decode_bits32=_bits(host(decoded)).reshape(groups,32))
        return CanonicalEncoded(q,hs.view(torch.float16),z,decoded,stages)
