"""Once-only CPU selection from frozen actual nonAIME FP captures.

Run with CUDA_VISIBLE_DEVICES empty after the FP collection has actually exited.
No model/torch import, GPU calls, AIME inputs, or scoring. All originals readonly.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from calibration_math import CalibrationSample,select_rowbal_parameters
from collect_fp_once import TASK,LAYERS,RAW_ROOT,verify_phase_a_sources


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _pairwise_sum32(value,axis=-1):
    x=np.moveaxis(np.asarray(value,dtype=np.float32),axis,-1)
    if x.shape[-1] not in (32,128):raise ValueError('reduction must be exact32 or128')
    while x.shape[-1]>1:
        half=x.shape[-1]//2
        x=(x[...,:half]+x[...,half:]).astype(np.float32)
    return x[...,0]


def rowbal_fp32_reference(state,omega,a,b):
    """Independent vectorized FP32 CAL quantizer; does not search lambdas.

    Explicit pairwise32/128 reductions. FP32 log/exp/pow arithmetic is a
    calibration rule, not a claim of bitidentity with future Triton pow lowering.
    Production GPU qualification must verify the same frozen mathematical rule.
    """
    x=np.asarray(state,dtype=np.float32);w=np.asarray(omega,dtype=np.float32)
    if x.shape!=(32,128,128) or w.shape!=(32,128) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('FP32 reference input identity/finite failure')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        m=np.sqrt((_pairwise_sum32(x*x,-1)*np.float32(1/128)+np.float32(1e-12)).astype(np.float32)).astype(np.float32)
        mg=m.reshape(32,4,32);wg=w.reshape(32,4,32)
        gm_m=np.exp((_pairwise_sum32(np.log(mg).astype(np.float32))*np.float32(1/32)).astype(np.float32)).astype(np.float32)
        gm_w=np.exp((_pairwise_sum32(np.log(wg).astype(np.float32))*np.float32(1/32)).astype(np.float32)).astype(np.float32)
        rm=(mg/gm_m[...,None]).astype(np.float32);rw=(wg/gm_w[...,None]).astype(np.float32)
        r=(np.power(rm,np.float32(a)).astype(np.float32)*np.power(rw,np.float32(-b)).astype(np.float32)).astype(np.float32)
        r=np.clip(r,np.float32(.25),np.float32(4.)).reshape(32,128)
        y=(x/r[...,None]).astype(np.float32).reshape(32,4,32,128)
        peak=np.max(np.abs(y),axis=-2)
        scale=(np.maximum(peak,np.float32(1e-12))*np.float32(1/127)).astype(np.float32)
        rounded=np.rint((y/scale[...,None,:]).astype(np.float32)).astype(np.float32)
        clipped=np.clip(rounded,np.float32(-127),np.float32(127)).astype(np.float32)
        codes=clipped.astype(np.int8)
        decoded=(codes.astype(np.float32)*scale[...,None,:]).astype(np.float32)
        decoded=np.where((clipped==0)&np.signbit(clipped),np.float32(-0.),decoded).astype(np.float32)
        result=(decoded.reshape(32,128,128)*r[...,None]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite decoded reference')
    return result


def select(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU selection requires empty CUDA_VISIBLE_DEVICES')
    if root.name!=TASK or root!=Path(__file__).resolve().parents[1]:raise RuntimeError('exact new independent root required')
    sys.path.insert(0,str(root/'src'));import execution_control as C
    source_sha=verify_phase_a_sources(root)
    plan=C.read(root/'CALIBRATION_CANDIDATE_PLAN.json');manifest=C.read(root/'calibration/FP_COLLECTION_MANIFEST.json')
    if manifest['status']!='PASS_ACTUAL_BOUNDED_NONAIME_FP_COLLECTION' or manifest['source_freeze_sha256']!=source_sha or manifest['candidate_plan_sha256']!=C.sha(root/'CALIBRATION_CANDIDATE_PLAN.json'):
        raise RuntimeError('actual FP collection is not source-bound complete')
    if manifest['raw_root']!=RAW_ROOT:raise RuntimeError('raw source path changed without independent relocation receipt')
    # An external actual process-exit+UUID-release receipt is required. A logical
    # end-of-loop marker is deliberately not accepted as a real process exit.
    release=C.read(root/'calibration/FP_COLLECTION_EXIT_RELEASE.json')
    if release.get('status')!='PASS_ACTUAL_EXIT0_AND_RELEASE' or release.get('exit_returncode')!=0 or release.get('release')!='PASS' or release.get('worker_identity')!=manifest['identity']:
        raise RuntimeError('actual collector exit/release is not independently verified')
    if C.same_process(manifest['identity']):raise RuntimeError('collector still live')
    C.atomic_new(root/'calibration/PARAMETER_SELECTION_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic(),source_freeze_sha256=source_sha))
    try:
        start=time.monotonic();raw=Path(RAW_ROOT);rows=manifest['files'];doc_map={d['document_id']:d for d in plan['new_collection_proposal']['documents']}
        actual=set();expected={(d['document_id'],t,l) for d in doc_map.values() for t in d['State_anchors'] for l in LAYERS}
        samples={'CAL':[],'VAL':[]}
        # Verify every preserved scientific file, not just the subset used for
        # selection. This remains CPU SHA/bytes work and never replays a model.
        for row in rows:
            p=Path(row['path'])
            if p.is_symlink() or not p.is_file() or p.parent.resolve()==raw.parent.resolve() or raw not in p.parents or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
                raise RuntimeError('raw scientific file changed:'+str(p))
            if row['kind']!='FP_STATE_AND_NEXT_TRANSPORT':continue
            ident=(row['document_id'],row['token'],row['layer'])
            if ident not in expected or ident in actual:raise RuntimeError('anchor roster differs')
            actual.add(ident);doc=doc_map[row['document_id']]
            if row['role']!=doc['role'] or row['next_token']!=row['token']+1:raise RuntimeError('anchor identity alignment differs')
            with np.load(p,allow_pickle=False) as z:
                x=z['state_h'].copy();g=z['transported_next'].copy()
            samples[doc['role']].append(CalibrationSample(doc['document_id'],doc['raw_document_sha256'],doc['role'],row['layer'],row['token'],row['next_token'],x,g))
        if actual!=expected:raise RuntimeError('missing actual scientific State anchors')
        omega_manifest=C.read(root/'TRANSPORTED_QUERY_OMEGA.json')
        if omega_manifest['collection_manifest_sha256']!=C.sha(root/'calibration/FP_COLLECTION_MANIFEST.json') or omega_manifest['layer_ids']!=list(LAYERS):raise RuntimeError('omega provenance differs')
        ws=np.asarray(omega_manifest['omega_normalized_fp32'],np.float32)
        if ws.shape!=(24,32,128):raise RuntimeError('actual omega layer/head/row shape differs')
        weights={l:ws[i] for i,l in enumerate(LAYERS)}
        chosen=select_rowbal_parameters(samples['CAL'],samples['VAL'],weights,rowbal_fp32_reference)
        chosen.update(task_id=TASK,selected=dict(a=chosen['selected_a'],b=chosen['selected_b']),lambdas=plan['WScale']['lambdas'],epsilon=plan['RowBal']['epsilon'],row_bounds=plan['RowBal']['r_clamp'],source_freeze_sha256=source_sha,omega_manifest_sha256=C.sha(root/'TRANSPORTED_QUERY_OMEGA.json'),candidate_plan_sha256=C.sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),FP_collection_manifest_sha256=C.sha(root/'calibration/FP_COLLECTION_MANIFEST.json'),FP_collector_exit_release_sha256=C.sha(root/'calibration/FP_COLLECTION_EXIT_RELEASE.json'),selection_arithmetic='CPU NumPy FP32 pairwise32/128 RMS/GM, FP32 pow, Canonical CUDA reciprocal127 Sym32 decode; one-step readout error reducedFP64',runtime_bitidentity_claim=False,all_three_formal_candidates_retained=True,rows_include_VAL_for_reporting_only=True,actual_CPU_wall_seconds=time.monotonic()-start,GPU_calls=0,new_AIME_samples=0)
        C.atomic_new(root/'CALIBRATION_PARAMETER_SELECTION.json',chosen)
        return chosen
    except BaseException as error:
        C.atomic_new(root/'calibration/PARAMETER_SELECTION_FIRST_FAILURE.json',dict(status='CALIBRATION_PARAMETER_SELECTION_FAIL_CLOSED_STOP',error=repr(error),traceback=traceback.format_exc(),identity=C.process_identity(os.getpid()),AIME_used=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();result=select(Path(args.root).resolve());print(json.dumps(dict(status=result['status'],selected=result['selected'],AIME_used=False)))
