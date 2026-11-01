#!/usr/bin/env python3
"""Bounded CPU import/closure checks; never loads9B or initializes CUDA."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('shared_cpu_closure',root/'source/shared_runtime_worker.py')
    w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
    manifest=w.read(root/'RUNTIME_CPU_CLOSURE_SOURCE_MANIFEST.json')
    for row in manifest['files']:
        path=root/row['path'];w.check(w.sha(path)==row['sha256'],'CPU input source changed')
    import torch
    w.check(not torch.cuda.is_initialized(),'CUDA already initialized')
    def forbidden(*a,**k):raise RuntimeError('CPU closure attempted CUDA initialization')
    torch.cuda._lazy_init=forbidden;torch.set_num_threads(1)
    c=w.Components(root)
    x=torch.arange(32*128*128,dtype=torch.float32).reshape(1,32,128,128).remainder(101).sub(50)/37
    encoded,raw=c.codec.trace_encode(x,'DUAL_M_INT8');decoded=encoded.decode()
    refplan=c.codec.references.STEP['QuantizationPlan']('gdn',torch.full((32,128),8,dtype=torch.long),torch.ones((32,128)))
    z,r,col,two,ref=c.codec.references.STEP['fit_state'](x,refplan)
    w.check(len(raw)==3 and two is None and torch.equal(encoded.codes.to(torch.int16),z),'Original DualM code closure failed')
    w.check(torch.equal(encoded.rows.squeeze(-1),r) and torch.equal(encoded.columns.squeeze(-2),col),'Original DualM stored scale closure failed')
    w.check(torch.equal(decoded,ref),'Original DualM reference reconstruction failed')
    sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    bindings={name:w.native_kernel_identity(getattr(qmod,name)) for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')}
    w.check(not torch.cuda.is_initialized(),'CPU test initialized CUDA')
    proof={'status':'PASS_CPU_IMPORT_CLOSURE_ONLY','worker_sha256':w.sha(root/'source/shared_runtime_worker.py'),
           'test_script_sha256':w.sha(__file__),'source_manifest_sha256':w.sha(root/'RUNTIME_CPU_CLOSURE_SOURCE_MANIFEST.json'),
           'CUDA_VISIBLE_DEVICES':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'CUDA_initialized':False,
           'shape':list(x.shape),'codes_sha256':c.tsha(encoded.codes),'rows_sha256':c.tsha(encoded.rows),
           'columns_sha256':c.tsha(encoded.columns),'decoded_sha256':c.tsha(decoded),'raw_scale_sites':len(raw),
           'DualM_all8_reference_codes_scales_values':'EXACT','actual_native_kernel_bindings':bindings,
           'architecture_sha256':w.sha(w.ARCH),'scope':'Synthetic CPU state and source closures only; no9B forward/sampling, no newfull6 Plan test',
           'GPU_qualification':'NOT_RUN','formal_generation':'NOT_RUN'}
    w.atomic_new(root/'RUNTIME_CPU_CLOSURE_FINAL.json',proof)
    print(json.dumps(proof,sort_keys=True),flush=True)


if __name__=='__main__':main()
