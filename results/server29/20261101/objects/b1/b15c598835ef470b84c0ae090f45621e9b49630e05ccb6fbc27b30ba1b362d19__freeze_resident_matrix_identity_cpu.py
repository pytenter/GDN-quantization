"""CPU-only exact reproduction of frozen Cq eigenbasis, not a new fit/choice."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)

def main(root):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only explicit empty CVD required')
    root=Path(root).resolve()
    source=root/'analysis/PRIOR_EIGENBASIS_CONSTRUCTION.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='basis')
    ns=dict(np=np,C=SimpleNamespace(require=require))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source)+'::frozen_basis_only','exec'),ns)
    cq=root/'reuse/QUERY_SECOND_MOMENTS_FP64.npz'
    catalog=json.loads((root/'operations/INITIAL_ASSET_AND_P1_IDENTITY_REVIEW.json').read_text())['files']
    expected=next(x for x in catalog if x['path']=='reuse/QUERY_SECOND_MOMENTS_FP64.npz')
    require(sha(cq)==expected['sha256'] and cq.stat().st_size==expected['bytes'],'Frozen original Cq bytes changed')
    h=np.ones((1,1),np.float64)
    while h.shape[0]<128:
        h=np.block([[h,h],[h,-h]])
    h/=math.sqrt(128)
    p=np.array([int(format(i,'07b')[::-1],2) for i in range(128)],np.int64)
    require(np.array_equal(np.sort(p),np.arange(128)) and np.array_equal(p[p],np.arange(128)),'Invalid fixed bit reversal')
    manifest=json.loads((root/'CANONICAL_UH_MATRICES_AND_SHA.json').read_text())
    rows=[]
    with np.load(cq,allow_pickle=False) as moments:
        for layer in manifest['layers']:
            index=layer['layer'];path=root/layer['path']
            require(sha(path)==layer['sha256'],'U/A original asset changed')
            with np.load(path,allow_pickle=False) as pack:
                u=pack['U_fp64_unique'].copy();a=pack['A_fp32_unique'].copy()
                mapping=pack['recurrent_to_unique'].copy()
                eigenvalues=pack['eigenvalues_unique'].copy()
            c=moments[str(index)]
            require(c.shape==(32,128,128) and c.dtype==np.float64,'Cq layer identity/shape differs')
            require(np.array_equal(mapping,np.arange(32)//2),'Head mapping differs')
            require(np.array_equal(c[::2],c[1::2]),'Shared physical Q/K head Cq pairs differ')
            heads=[]
            for head in range(16):
                w,reproduced,clusters=ns['basis'](c[2*head])
                require(np.array_equal(reproduced,u[head]) and np.array_equal(w,eigenvalues[head]),'Frozen deterministic eigenbasis did not reproduce exactly')
                a64=u[head]@h
                require(np.array_equal(a64.astype(np.float32),a[head]),'Original single-cast A32 differs')
                s3=a64[:,p].astype(np.float32)
                require(np.array_equal(s3,a[head][:,p]),'S3 must only permute C1 columns')
                residual=np.linalg.norm(c[2*head]@u[head]-u[head]*w)/max(np.linalg.norm(c[2*head]),1e-300)
                heads.append(dict(unique_head=head,recurrent_heads=[2*head,2*head+1],
                    Cq_sha256=digest(c[2*head]),U64_sha256=digest(u[head]),C1_A64_sha256=digest(a64),
                    C1_A32_sha256=digest(a[head]),S3_A64_sha256=digest(a64[:,p]),S3_A32_sha256=digest(s3),
                    eigen_relative_residual=float(residual),canonical_degenerate_clusters=clusters))
            rows.append(dict(layer=index,source_asset_path=layer['path'],source_asset_sha256=layer['sha256'],
                C1_all16_A32_sha256=digest(a),S3_all16_A32_sha256=digest(a[...,p]),heads=heads))
    result=dict(status='PASS_FROZEN_CQ_AND_DETERMINISTIC_U_EXACT_REPRODUCTION_ALL384_HEADS',
        Cq=dict(path=str(cq.relative_to(root)),bytes=expected['bytes'],sha256=expected['sha256']),
        eigenbasis_rule_source_sha256=sha(source),H64_sha256=digest(h),H_single_cast_FP32_sha256=digest(h.astype(np.float32)),
        PBR128=p.tolist(),PBR128_int64_sha256=digest(p),recurrent_to_unique=(np.arange(32)//2).tolist(),
        canonical_rule='U64@H64 then oneFP32cast; S3sameUandH withfixedcolumnPBR',layers=rows,
        independent_query_samples_per_unique_head=4096,new_Query_tokens=0,AIME_used=False,
        original_U_bytes_reproduced_not_replaced=True,source_script_sha256=sha(__file__),GPU_dispatches=0,
        deployment_GPU_H_generator_still_requires_qualified_originalTorch_GPU_arithmetic=True)
    path=root/'RESIDENT_ROTATION_MATRICES_SHA.json'
    with path.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],heads=sum(len(x['heads']) for x in rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
