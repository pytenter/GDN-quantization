#!/usr/bin/env python3
"""One original HF/manual sampling path, two post-forward state codecs.

No CUDA/model import occurs before the root coordinator's frozen authorization
is checked.  `audit` is CPU/std-library source closure qualification only.
This file adds orchestration and observations; original GDN, sampler, H transform
and qualified H/Dual-M arithmetic are compiled from SHA-pinned original AST.
"""
import argparse
import ast
import collections
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

TASK = 'QWEN_GDN_AIME26_STEPQUANT6_VS_H_DUALM_2SEED_V1'
LAYERS = (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
METHODS = ('STEPQUANT6', 'H_DUALM')
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
ARCH_SHA = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
PINS = {
    'sampler': ('protocol_recovery/snapshots', '9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd', 'run_qwen_aime26_formal.py'),
    'common': ('protocol_recovery/snapshots', '34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7', 'aime26_common.py'),
    'h': ('provenance/h_dual_m_sources', '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69', 'formal_c128_runner.py'),
    'worker': ('provenance/h_dual_m_sources', 'e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773', 'worker.py'),
    'codec': ('provenance/h_dual_m_sources', 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f', 'codec.py'),
    'encode': ('provenance/h_dual_m_sources', 'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6', 'dual_axis_codec.py'),
    'all8': ('provenance/h_dual_m_sources', '0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69', 'stepquant_quantization.py'),
    'c128': ('provenance/h_dual_m_sources', '62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067', 'cayley_rotation.py'),
    'numerical': ('provenance/h_dual_m_sources', 'b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c', 'numerical_recovery.py'),
}
SAMPLER_NODES = {'setup_seed','render_input','build_warpers','sample_next','stop_ids','build_top_k_audit'}
SAMPLER_CONSTANTS = {'TEMPERATURE','TOP_P','TOP_K','MIN_P','PRESENCE_PENALTY','REPETITION_PENALTY',
                     'MAX_NEW_TOKENS','TOP_K_AUDIT_VERSION','TOP_K_IMPLEMENTATION'}


def check(ok, why):
    if not bool(ok):
        raise RuntimeError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_new(path, obj):
    """Durably publish without replacing a prior success, failure, or attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pin_path(root, key):
    folder, digest, name = PINS[key]
    path = root/folder/(digest+'_'+name)
    check(sha(path) == digest, 'Original source SHA mismatch: '+key)
    return path


def selected(path, names):
    result = []
    for node in ast.parse(Path(path).read_text()).body:
        assigned = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if getattr(node, 'name', None) in names or assigned & names:
            result.append(node)
    found = {getattr(n,'name',None) for n in result}
    found |= {t.id for n in result if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    check(names <= found, 'Missing original AST: '+str(names-found))
    return result


def extract(path, names, namespace):
    exec(compile(ast.Module(body=selected(path,names),type_ignores=[]),str(path),'exec'), namespace)
    return namespace


def ast_identity(root):
    paths = {k:pin_path(root,k) for k in PINS}
    names = SAMPLER_NODES | SAMPLER_CONSTANTS | {'hadamard','normalize_qk','QwenKeyHadamardPatch','get_state','load_model_and_tokenizer','GDN_LAYERS','EPS'}
    left = {getattr(n,'name',None) or n.targets[0].id:ast.dump(n,include_attributes=False) for n in selected(paths['sampler'],names)}
    right = {getattr(n,'name',None) or n.targets[0].id:ast.dump(n,include_attributes=False) for n in selected(paths['h'],names)}
    check(left == right, 'Historical sampler/H AST identity mismatch')
    sampler_tree = ast.Module(body=selected(paths['sampler'],SAMPLER_NODES|SAMPLER_CONSTANTS),type_ignores=[])
    check(not any(isinstance(n,ast.Name) and n.id in {'score_aime','extract_aime_answer','score_response'} for n in ast.walk(sampler_tree)), 'Scorer entered sampling closure')
    return {'status':'PASS_CPU_SOURCE_CLOSURE_ONLY', 'no_torch_or_CUDA_import':True,
            'sources':{k:{'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()},
            'original_AST_equal_between_9ff56d_and_77b1': sorted(left),
            'sampler_AST_sha256':canonical_sha(left),
            'dual_AST_sha256':canonical_sha(ast.dump(selected(paths['worker'],{'dual'})[0],include_attributes=False)),
            'new_GPU_or_numeric_qualification':'NOT_RUN'}


def verify_manifest(root, filename, digest):
    path = root/filename
    check(sha(path)==digest, 'Frozen manifest SHA mismatch: '+filename)
    obj = read(path)
    rows = obj.get('execution_sources',obj.get('files',[])) + obj.get('external_dependencies',[])
    check(rows, 'Empty source manifest')
    for row in rows:
        p = Path(row['path'])
        if not p.is_absolute(): p = root/p
        check(p.is_file() and sha(p)==row['sha256'], 'Source dependency mismatch: '+str(p))
        if 'bytes' in row: check(p.stat().st_size==row['bytes'], 'Source dependency bytes mismatch: '+str(p))
    return obj


def admission(root, args):
    check(args.authorization is not None, 'Root coordinator authorization file required; no GPU dispatch')
    auth = read(args.authorization)
    check(auth.get('task_id')==TASK and auth.get('status')=='PASS', 'Authorization task/status mismatch')
    check(args.mode in auth.get('allowed_modes',[]) and str(args.gpu) in auth.get('gpu_uuid_map',{}), 'Mode/GPU not authorized')
    check(os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu), 'Physical GPU visibility must be exactly the requested GPU')
    check(sha(root/'PREREGISTRATION.json')==auth['prereg_sha256'], 'Preregistration SHA changed')
    prereg = read(root/'PREREGISTRATION.json')
    check(prereg.get('frozen') is True and prereg.get('TASK_ID')==TASK, 'Preregistration not frozen')
    check(prereg['sampling']['max_new_tokens']==81920 and prereg['sampling']['seeds']==[1,2], 'Sampling contract changed')
    check(prereg['common_GDN_state_timing']['prefill_quantized'] is True, 'Prefill QDQ contract missing')
    verify_manifest(root,auth.get('source_manifest','SOURCE_MANIFEST.json'),auth['source_manifest_sha256'])
    verify_manifest(root,'RUNTIME_SOURCE_MANIFEST.json',auth['runtime_source_manifest_sha256'])
    if args.mode!='calibration':
        check(sha(root/'calibration/qwen35_9b_stepquant6_plan.pt')==auth['plan_sha256'], 'Full STEP plan SHA changed')
        plan_gate = read(root/'calibration/PLAN_CPU_QUALIFICATION.json')
        check(plan_gate.get('status')=='PASS' and plan_gate.get('plan_sha256')==auth['plan_sha256'], 'Full6 plan not CPU qualified')
    else:
        check(sha(root/'CALIBRATION_INPUTS.json')==auth['calibration_inputs_sha256'], 'Calibration input manifest changed')
    for rel,digest in auth.get('required_gate_receipts',{}).items():
        check(sha(root/rel)==digest and read(root/rel).get('status')=='PASS', 'Required gate not PASS: '+rel)
    return auth,prereg


class Components:
    """Bind unchanged numeric AST. No training adapters or replacement kernels."""
    def __init__(self,root):
        import torch
        self.torch=torch; self.root=root
        self.paths={k:pin_path(root,k) for k in PINS}
        self.tsha=lambda t:hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        ns=dict(torch=torch,math=math,os=os,sys=sys,random=random,Path=Path,
                MODEL_PATH=Path('/data/zypan/modelscope_models/Qwen3.5-9B'),
                TRANSFORMERS_SRC=Path('/data/zypan/transformers-qwen35'),
                STATE_QUANT_SRC=Path('/data/zypan/experiments/qwen35_gdn_quant'))
        extract(self.paths['common'],{'raw_messages','ids_sha256'},dict_ns:=dict(hashlib=hashlib,json=json))
        ns.update(dict_ns)
        extract(self.paths['sampler'],SAMPLER_NODES|SAMPLER_CONSTANTS|{'EPS','GDN_LAYERS','hadamard','normalize_qk','QwenKeyHadamardPatch','get_state','load_model_and_tokenizer'},ns)
        # Same memoization used by qualified core.formal(True); values unchanged.
        original_h=ns['hadamard']; matrices={}
        def memo_h(n=128,dtype=torch.float32,device=None):
            key=(n,dtype,str(device))
            if key not in matrices: matrices[key]=original_h(n,dtype,device)
            return matrices[key]
        ns['hadamard']=memo_h
        self.f=SimpleNamespace(**ns)
        canon=extract(self.paths['c128'],{'EPS','QuantizationResult','canonical_symmetric_int8_qdq','qwen_c128_qdq'},dict(torch=torch,dataclass=dataclasses.dataclass))
        all8=extract(self.paths['all8'],{'QuantizationPlan','_fp16_scale','fit_state'},dict(torch=torch,dataclass=dataclasses.dataclass))
        codec_ns=dict(torch=torch,dataclass=dataclasses.dataclass,SYNC_COUNT=0,SCALE_TRACE=None,
                      NAMES=('C128_CANONICAL','C128_FP16_SCALE','C128_LS_FP16','DUAL_M_INT8'))
        extract(self.paths['codec'],{'require','source','EncodedState'},codec_ns)
        encode_ns=dict(torch=torch,CANON=canon,NAMES=codec_ns['NAMES'],LOW=2**-24,HIGH=65504.,source=codec_ns['source'],EncodedState=codec_ns['EncodedState'])
        extract(self.paths['encode'],{'safe_half','levels','encode'},encode_ns)
        codec_ns.update(_namespace=encode_ns,_original_safe=encode_ns['safe_half'],_original_levels=encode_ns['levels'])
        extract(self.paths['codec'],{'safe_half','levels','trace_encode'},codec_ns)
        encode_ns.update(safe_half=codec_ns['safe_half'],levels=codec_ns['levels'])
        codec_ns.update(encode=encode_ns['encode'],references=SimpleNamespace(STEP=all8))
        self.codec=SimpleNamespace(**codec_ns)
        self.current={}
        k=SimpleNamespace(states=lambda cache,f:[f.get_state(cache,l) for l in f.GDN_LAYERS],
                          check=check,tsha=self.tsha,save=lambda rel,obj:atomic_new(root/'failures'/rel,obj))
        dual_ns=dict(K=k,C=self.codec,torch=torch,CURRENT=self.current)
        extract(self.paths['worker'],{'dual'},dual_ns)
        self.dual=dual_ns['dual']
        self.scan_ns=dict(torch=torch,ast=ast,Path=Path,ARCH=ARCH,ARCH_SHA=ARCH_SHA,
                          ROOT=root/'source',sha=sha,check=check)
        extract(self.paths['numerical'],{'compensated_scan','backend','chunk_binding'},self.scan_ns)
        sys.path.insert(0,str(root/'official_source/STEPQuant_f2deac58'))
        self.step=importlib.import_module('stepquant.quantization')
        check(Path(self.step.__file__).resolve()==(root/'official_source/STEPQuant_f2deac58/stepquant/quantization.py').resolve(), 'Wrong official STEP import')

    def load(self):
        torch=self.torch
        check(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1', 'Torch/CUDA changed')
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model,tok=self.f.load_model_and_tokenizer()
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        check(all(p.device.type=='cuda' and p.device.index==0 for p in model.parameters()), 'Single-GPU device map/offload changed')
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        check(Path(qmod.__file__).resolve()==ARCH.resolve() and sha(ARCH)==ARCH_SHA, 'Architecture changed')
        check(Path(inspect.unwrap(qmod.torch_recurrent_gated_delta_rule).__code__.co_filename).resolve()==ARCH.resolve(), 'Recurrent backend changed')
        qmod.torch_chunk_gated_delta_rule=self.scan_ns['chunk_binding'](qmod,'compensated')
        metadata={'architecture_sha256':ARCH_SHA,'torch':torch.__version__,'cuda':torch.version.cuda,
                  'deterministic':True,'TF32':False,'model_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
                  'chunk_scan':'original qualified FP32 compensated g.cumsum binding, shared both methods',
                  'recurrent_helper':inspect.unwrap(qmod.torch_recurrent_gated_delta_rule).__code__.co_filename,
                  'attention_implementation':getattr(model.config.get_text_config(),'_attn_implementation',None)}
        check(metadata['model_dtypes']==['torch.bfloat16'], 'Model weight dtype changed')
        return model,tok,qmod,metadata


def tensor_version(t):
    try: return t._version
    except RuntimeError: return 'INFERENCE_TENSOR_NO_VERSION_COUNTER'


class Consumption:
    """Observation only around original kernel; formal never CPU-hashes states."""
    def __init__(self,c,model,qmod,trace):
        self.c=c;self.qmod=qmod;self.trace=trace;self.active=None;self.step=0
        self.previous={};self.calls=collections.Counter();self.checks=0;self.trace_rows=[];self.handles=[];self.details={}
        modules={int(m.layer_idx):m for m in model.modules() if type(m).__name__=='Qwen3_5GatedDeltaNet'}
        check(set(modules)==set(LAYERS), 'Actual GDN layer mapping changed')
        for l,m in modules.items():
            self.handles.append(m.register_forward_pre_hook(lambda _m,_a,l=l:setattr(self,'active',l)))
        self.saved=[]
        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
            original=getattr(qmod,name);self.saved.append((name,original));setattr(qmod,name,self.wrap(original))

    def wrap(self,fn):
        def call(q,k,v,*a,**kw):
            l=self.active;check(l in LAYERS, 'Unknown GDN layer');self.calls[(self.step,l)]+=1
            initial=kw.get('initial_state')
            if self.previous:
                prior=self.previous[l]
                check(initial is prior['tensor'] and initial.data_ptr()==prior['ptr'], 'Next kernel did not consume QDQ tensor/storage')
                check(tensor_version(initial)==prior['version'], 'QDQ tensor modified before next kernel')
                if self.trace:check(self.c.tsha(initial)==prior['sha256'], 'QDQ bytes changed before next kernel')
                self.checks+=1
            y,s=fn(q,k,v,*a,**kw)
            check(s.dtype==self.c.torch.float32 and tuple(s.shape)==(1,32,128,128), 'State shape/dtype changed')
            if self.trace:self.details[str(l)]={'q':self.c.tsha(q),'k':self.c.tsha(k),'state_before_QDQ':self.c.tsha(s),'readout':self.c.tsha(y.to(v.dtype)),'initial':None if initial is None else self.c.tsha(initial)}
            return y,s
        return call

    def after(self,cache):
        for l in LAYERS:check(self.calls[(self.step,l)]==1, 'Not one original GDN call/layer/forward')
        self.previous={l:{'tensor':self.c.f.get_state(cache,l),'ptr':self.c.f.get_state(cache,l).data_ptr(),
                          'version':tensor_version(self.c.f.get_state(cache,l)),
                          'sha256':self.c.tsha(self.c.f.get_state(cache,l)) if self.trace else None} for l in LAYERS}

    def close(self):
        for name,old in self.saved:setattr(self.qmod,name,old)
        for h in self.handles:h.remove()
        self.previous.clear()


def packed_description(packed):
    def d(t):return None if t is None else {'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()}
    return {'nbytes':packed.nbytes,'codes':{str(b):d(t) for b,t in packed.codes.items()},
            'rows':d(packed.rows),'columns':d(packed.columns),'pivots':d(packed.pivots),'two_bit_rows':d(packed.two_bit_rows),
            'resident_representation':'transient PackedState, released after FP32 QDQ writeback'}


class PostForwardQDQ:
    def __init__(self,c,model,method,artifact,trace):
        self.c=c;self.method=method;self.trace=trace;self.calls=0;self.writebacks=0
        self.scale_counts=collections.Counter();self.peak_payload=0;self.payload=None;self.trace_rows=[]
        self.codecs={};self.names={}
        actual={name:int(m.layer_idx) for name,m in model.named_modules() if type(m).__name__=='Qwen3_5GatedDeltaNet'}
        check(actual==artifact['layer_name_to_index'], 'Plan/module/layer mapping mismatch')
        check(set(actual.values())==set(LAYERS) and set(artifact['plans'])==set(actual), 'Plan must cover all24 GDN layers')
        if method=='STEPQUANT6':
            for name,l in actual.items():
                check(artifact['state_shape_per_layer'][name]==[1,32,128,128], 'Plan state shape mismatch')
                plan=c.step.QuantizationPlan(**artifact['plans'][name]).to('cuda:0')
                self.codecs[l]=c.step.StateCodec(plan);self.names[l]=name

    def step_qdq(self,cache,phase,step):
        torch=self.c.torch;trace=self.trace;metric=trace or step%1024==0
        detail={};stat={};started=time.monotonic()
        if self.method=='H_DUALM':
            self.c.current.update(method=self.method,phase=phase,step=step)
            detail,stat,timing=self.c.dual(cache,self.c.f,trace=trace,metrics=metric,parity=trace and step==0)
            for key in ('row_floor_hits','row_ceiling_hits','column_floor_hits','column_ceiling_hits'):
                self.scale_counts[key]+=stat[key]
            payload=24*(32*128*128+2*32*128*2)
        else:
            payload=0;layout={};bounds=collections.Counter();riskenergy=0.;riskmse=0.
            for l in LAYERS:
                state=self.c.f.get_state(cache,l);codec=self.codecs[l]
                check(state.dtype==torch.float32 and tuple(state.shape)==(1,32,128,128), 'STEP native state changed')
                check(torch.isfinite(state).all(), 'Nonfinite STEP source')
                raw=[];old=self.c.step._fp16_scale
                def observed_scale(x):
                    raw.append(x.detach());return old(x)
                self.c.step._fp16_scale=observed_scale
                try:packed=codec.encode(state)
                finally:self.c.step._fp16_scale=old
                decoded=codec.decode(packed)
                check(decoded.dtype==torch.float32 and tuple(decoded.shape)==tuple(state.shape) and torch.isfinite(decoded).all(), 'Invalid STEP reconstruction')
                for t in (packed.rows,packed.columns,packed.pivots):check(torch.isfinite(t).all(), 'Nonfinite STEP FP16 metadata/pivots')
                check((packed.rows>0).all() and (packed.columns>0).all(), 'Nonpositive STEP scale')
                check(packed.rows.dtype==torch.float16 and packed.columns.dtype==torch.float16 and packed.pivots.dtype==torch.float16, 'STEP stored dtype changed')
                check(len(raw)==3, 'Unexpected full6 fitting scale sites')
                rowbad=(raw[0]<2**-24)|(raw[0]>65504.)
                colbad=((raw[1]<2**-24)|(raw[1]>65504.))|((raw[2]<2**-24)|(raw[2]>65504.))
                for key,seq in (('row',[raw[0]]),('column',raw[1:])):
                    bounds[key+'_floor_hits']+=sum(int((v<2**-24).sum()) for v in seq)
                    bounds[key+'_ceiling_hits']+=sum(int((v>65504.).sum()) for v in seq)
                if bool(rowbad.any()|colbad.any()):
                    integer=(codec.plan.bits!=16)[None,:,:,None]
                    mask=(rowbad.expand_as(state)|colbad.expand_as(state))&(state!=0)&integer
                    xd=state.double();err=decoded.double()-xd;energy=xd.square().sum((-1,-2));affected=(xd.square()*mask).sum((-1,-2))
                    frac=torch.where(energy>0,affected/energy,0);mse=torch.where(affected>0,(err.square()*mask).sum((-1,-2))/affected,0)
                    riskenergy=max(riskenergy,float(frac.max()));riskmse=max(riskmse,float(mse.max()))
                    check(not bool(((frac>=1e-4)&(mse>1e-2)).any()), 'Full6 per-layer/head FP16 scale risk STOP')
                    del xd,err,energy,affected,frac,mse,mask,integer
                if trace:
                    z,r,col,two,ref=self.c.step.fit_state(state,codec.plan)
                    check(two is None and packed.two_bit_rows is None, 'Full6 unexpected INT2 grouping')
                    check(torch.equal(decoded,ref), 'Official StateCodec vs fit_state numerical reconstruction mismatch')
                    flat=z.reshape(1,-1,128)
                    for b in (2,4,6,8):
                        idx=codec.indices[b];levels=self.c.step.unpack(packed.codes[b],b,idx.numel()*128).reshape(1,-1,128)
                        check(torch.equal(levels-(2**(b-1)-1),flat[:,idx]), 'Official packed code parity mismatch')
                    check(torch.equal(packed.rows,r.reshape(1,-1)[:,codec.integer_rows]) and torch.equal(packed.columns,col[:,codec.integer_heads]), 'Official stored scale parity mismatch')
                    piv=state.reshape(1,-1,128)[:,codec.indices[16]].half()
                    check(torch.equal(packed.pivots,piv), 'Official FP16 pivot parity mismatch')
                    detail[str(l)]={'pre':self.c.tsha(state),'dequant':self.c.tsha(decoded),'fit_reference':self.c.tsha(ref),
                                    'fit_byte_equal':self.c.tsha(decoded)==self.c.tsha(ref),'rows':self.c.tsha(packed.rows),
                                    'columns':self.c.tsha(packed.columns),'pivots':self.c.tsha(packed.pivots),
                                    'codes':{str(b):self.c.tsha(v) for b,v in packed.codes.items()},
                                    'parity':'PASS_EXACT_VALUES_CODES_STORED_SCALES_PIVOTS'}
                    del z,r,col,two,ref,flat,piv
                state.copy_(decoded)
                if trace:detail[str(l)]['post']=self.c.tsha(state)
                payload+=packed.nbytes
                if step==0:layout[self.names[l]]=packed_description(packed)
                del packed,decoded,raw,state
            if step==0:self.payload=layout
            self.scale_counts.update(bounds)
            stat={**bounds,'max_affected_energy_fraction':riskenergy,'max_affected_relative_mse':riskmse}
            timing={'wall_including_safety_seconds':time.monotonic()-started}
        self.calls+=1;self.writebacks+=24;self.peak_payload=max(self.peak_payload,payload)
        if trace:self.trace_rows.append({'step':step,'phase':phase,'layers':detail,'scale_safety':stat})
        return stat,timing


def generate(c,model,tok,qmod,item,method,artifact,gpu,metadata,max_tokens,trace=False):
    torch=c.torch;f=c.f;tokens=[];survivors=[];past=None;obs=None;patch=None;qdq=None
    record={'status':'INCOMPLETE','task':TASK,'method':method,'problem_id':item['problem_id'],
            'problem_idx':item.get('problem_idx'),'canonical_index':item['canonical_index'],'seed':item['seed'],
            'gpu':gpu,'runtime_version':metadata,'retry_count':0,'scoring_status':'DEFERRED_FROZEN_V4_UNTIL_ALL120_LEGAL',
            'correct':None,'extracted_answer':None,'runtime_error':None,'nonfinite':False}
    started=time.monotonic();record['start_unix']=time.time();trace_rows=[];qdq_wall=0.;forward_wall=0.
    try:
        f.setup_seed(item['seed'])
        messages,input_ids=f.render_input(tok,item['problem']);input_ids=input_ids.to('cuda:0')
        inputhash=f.ids_sha256(input_ids[0].tolist())
        if 'input_ids_hash' in item:check(inputhash==item['input_ids_hash'], 'Frozen prompt token identity changed')
        record.update(raw_messages=messages,input_ids_hash=inputhash,input_tokens=int(input_ids.shape[-1]),problem=item['problem'],gold_answer=item.get('answer'))
        generator=torch.Generator(device='cuda:0').manual_seed(item['seed'])
        record['initial_generator_state_sha256']=c.tsha(generator.get_state())
        warpers=f.build_warpers();stops=f.stop_ids(tok);mask=None
        qdq=PostForwardQDQ(c,model,method,artifact,trace)
        obs=Consumption(c,model,qmod,trace)
        patch=f.QwenKeyHadamardPatch(model) if method=='H_DUALM' else contextlib.nullcontext()
        torch.cuda.reset_peak_memory_stats()
        with patch as h,torch.inference_mode():
            obs.step=0
            t0=time.monotonic();output=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=True);past=output.past_key_values
            torch.cuda.synchronize();forward_wall+=time.monotonic()-t0
            check(torch.isfinite(output.logits).all(), 'Nonfinite prefill logits')
            t0=time.monotonic();qdq.step_qdq(past,'prefill',0);obs.after(past);qdq_wall+=time.monotonic()-t0
            mask=torch.zeros_like(output.logits[:,-1,:],dtype=torch.bool)
            if trace:trace_rows.append({'forward':0,'logits_sha256':c.tsha(output.logits),'helper':dict(obs.details),'state_sha256':{str(l):c.tsha(f.get_state(past,l)) for l in LAYERS}})
            current,n=f.sample_next(warpers,output.logits[:,-1,:],mask,generator)
            mask.scatter_(1,current,True);tokens.append(int(current.item()));survivors.append(n)
            del output
            while len(tokens)<max_tokens and tokens[-1] not in stops:
                step=len(tokens);obs.step=step;obs.details={}
                t0=time.monotonic();output=model(input_ids=current,past_key_values=past,use_cache=True);past=output.past_key_values
                torch.cuda.synchronize();forward_wall+=time.monotonic()-t0
                check(torch.isfinite(output.logits).all(), 'Nonfinite decode logits')
                t0=time.monotonic();qdq.step_qdq(past,'decode',step);obs.after(past);qdq_wall+=time.monotonic()-t0
                if trace:trace_rows.append({'forward':step,'logits_sha256':c.tsha(output.logits),'helper':dict(obs.details),'state_sha256':{str(l):c.tsha(f.get_state(past,l)) for l in LAYERS}})
                current,n=f.sample_next(warpers,output.logits[:,-1,:],mask,generator)
                mask.scatter_(1,current,True);tokens.append(int(current.item()));survivors.append(n)
                del output
                if len(tokens)%128==0:print(json.dumps({'event':'PROGRESS','method':method,'canonical_index':item['canonical_index'],'output_tokens':len(tokens),'wall_seconds':time.monotonic()-started}),flush=True)
            n=len(tokens);check(n>0 and qdq.calls==n and qdq.writebacks==24*n and obs.checks==24*(n-1), 'Persistent QDQ counts failed')
            check(sum(obs.calls.values())==24*n and len(obs.calls)==24*n, 'Original core calls mismatch')
            ha={'calls':h.calls,'layers':sorted(h.layers),'phase_counts':dict(h.phase_counts)} if method=='H_DUALM' else {'calls':0,'layers':[],'phase_counts':{}}
            if method=='H_DUALM':check(ha['calls']==24*n and ha['phase_counts']=={'chunk_prefill':24,'recurrent_decode':24*(n-1)}, 'H transform calls changed')
            record.update(status='COMPLETE',finish_reason='eos' if tokens[-1] in stops else 'length',
                          final_generator_state_sha256=c.tsha(generator.get_state()),hadamard_audit=ha,
                          quantizer_audit={'calls':qdq.calls,'prefill_calls':1,'decode_calls':n-1,'prefill_quantized':True,
                              'layer_writebacks':qdq.writebacks,'next_consumption_checks':obs.checks,
                              'consumption_check':'same tensor object, data_ptr and available version; canary additionally byte SHA',
                              'formal_per_token_CPU_tensor_hashing':False,'cache_representation':'FP32 reconstructed QDQ carrier; no unquantized shadow',
                              'persistent_recurrent_tensor_bytes':50331648,'transient_encoded_payload_peak_bytes':qdq.peak_payload,
                              'transient_STEP_payload_layout':qdq.payload,'scale_safety_totals':dict(qdq.scale_counts)},
                          sampling_audit=f.build_top_k_audit(survivors),stop_token_ids=sorted(stops),
                          generation_config={'enable_thinking':True,'do_sample':True,'temperature':1.,'top_p':.95,'top_k':20,'min_p':0.,
                              'presence_penalty':1.5,'repetition_penalty':1.,'presence_penalty_scope':'generated_output_tokens_only',
                              'max_new_tokens':max_tokens,'sampling_seed':item['seed']},
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception:
        record['runtime_error']=traceback.format_exc()
        record['nonfinite']='Nonfinite' in record['runtime_error'] or 'nonfinite' in record['runtime_error']
        record['finish_reason']='technical_failure'
    finally:
        # H context restores its saved observers before the observer is removed.
        if obs is not None:obs.close()
        record.update(generated_token_ids=tokens,generated_token_count=len(tokens),output_tokens=len(tokens),
                      response=tok.decode(tokens,skip_special_tokens=True) if tokens else '',
                      forward_wall_seconds=forward_wall,qdq_and_consumption_audit_wall_seconds=qdq_wall,
                      latency_s=time.monotonic()-started,end_unix=time.time())
        record['tokens_per_s']=len(tokens)/max(record['latency_s'],1e-9)
        if trace:record['qualification_trace']={'forward':trace_rows,'QDQ':[] if qdq is None else qdq.trace_rows}
        past=None;obs=None;qdq=None;patch=None;gc.collect();torch.cuda.empty_cache()
    return record


def gpu_identity(torch,args,auth):
    torch.cuda.set_device(0)
    check(torch.cuda.device_count()==1, 'Worker must see exactly one GPU')
    props=torch.cuda.get_device_properties(0)
    expected=auth['gpu_uuid_map'][str(args.gpu)]
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    rows=[line.split(', ') for line in out.strip().splitlines()]
    actual=next(r for r in rows if int(r[0])==args.gpu)
    check(actual[1]==expected and '3090' in actual[2] and '3090' in props.name, 'Physical GPU UUID/class changed')
    uuid=getattr(props,'uuid',None)
    if uuid is not None:check(str(uuid)==expected or str(uuid).replace('GPU-','')==expected[4:], 'Torch visible UUID disagrees')
    return {'physical_index':args.gpu,'UUID':expected,'logical_index':0,'name':props.name,'total_memory_bytes':props.total_memory,'nvidia_smi':actual}


def units(root,auth):
    path=root/'FROZEN_UNITS.json';check(sha(path)==auth['units_sha256'], 'Formal units changed')
    rows=read(path)['units'];check(len(rows)==60, 'Exactly60 fixed pairs required')
    check([r['canonical_index'] for r in rows]==list(range(60)), 'Canonical order changed')
    check(len({(r['problem_id'],r['seed']) for r in rows})==60 and len({r['problem_id'] for r in rows})==30, 'Question/seed identity mismatch')
    for r in rows:
        i=r['canonical_index'];g=i%4;local=i//4
        check(r['gpu']==g and r['seed']==[1,2][i%2], 'Fixed worker/seed assignment changed')
        check(r['method_order']==(list(METHODS) if local%2==0 else list(reversed(METHODS))), 'Method order changed')
        check(r.get('gpu_uuid',auth['gpu_uuid_map'][str(g)])==auth['gpu_uuid_map'][str(g)], 'Pair physical UUID changed')
    return rows


def compare_canary(old,new):
    keys=('generated_token_ids','input_ids_hash','initial_generator_state_sha256','final_generator_state_sha256','sampling_audit',
          'hadamard_audit','qualification_trace')
    check(old['status']=='COMPLETE' and new['status']=='COMPLETE', 'Incomplete canary cannot qualify')
    for key in keys:check(old[key]==new[key], 'Fresh-process exact canary mismatch: '+key)
    check(old['output_tokens']==128 and new['output_tokens']==128, '128 sampled-token qualification not completed')


def resource_pressure(c,model,auth):
    """Storage/workspace pressure only, never claimed 81920-token numeric test."""
    torch=c.torch;n=81920+int(auth['max_prompt_tokens']);held=[]
    torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();started=time.monotonic()
    with torch.inference_mode():
        for _ in range(8):
            held.extend([torch.zeros((1,4,n,256),dtype=torch.bfloat16,device='cuda:0') for _ in range(2)])
        # One layer's old/new cache cat overlap and replicated 16-query-head KV.
        extra=[torch.zeros((1,4,n+1,256),dtype=torch.bfloat16,device='cuda:0') for _ in range(2)]
        expanded=[torch.zeros((1,16,n+1,256),dtype=torch.bfloat16,device='cuda:0') for _ in range(2)]
        query=torch.zeros((1,16,1,256),dtype=torch.bfloat16,device='cuda:0')
        attention_start=time.monotonic()
        attended=torch.nn.functional.scaled_dot_product_attention(query,expanded[0],expanded[1],dropout_p=0.,is_causal=False)
        check(torch.isfinite(attended).all(), 'Nonfinite long-KV native SDPA output')
        torch.cuda.synchronize();attention_wall=time.monotonic()-attention_start
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        free,total=torch.cuda.mem_get_info()
        check(int(auth['resource_min_free_bytes'])==1024**3, 'Frozen long-resource free margin changed')
        check(free>=1024**3, 'Long-KV workspace pressure insufficient1GiB free bytes')
        result={'status':'PASS_RESOURCE_PRESSURE_ONLY','total_context_slots':n,'output_cap':81920,
                'KV_BF16_bytes':sum(t.numel()*t.element_size() for t in held),'additional_workspace_bytes':sum(t.numel()*t.element_size() for t in extra+expanded),
                'model_before_bytes':before,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,
                'CUDA_driver_free_bytes':free,'total_bytes':total,'wall_seconds':time.monotonic()-started,
                'long_SDPA_query_shape':list(query.shape),'long_SDPA_KV_shape':list(expanded[0].shape),
                'long_SDPA_output_dtype':str(attended.dtype),'long_SDPA_finite':True,'long_SDPA_wall_seconds':attention_wall,
                'science_numeric_81920':'NOT_TESTED_NEW_METHODS','limitations':'Synthetic long KV/workspace allocation and one native SDPA attention call with live model; no81920 GDN trajectory. Historical complete81920 source is separate evidence.'}
        del held,extra,expanded,query,attended
    gc.collect();torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--gpu',type=int,choices=range(4),default=0)
    p.add_argument('--mode',choices=('audit','calibration','canary','fresh','formal','resource'),required=True)
    p.add_argument('--authorization',type=Path)
    args=p.parse_args();root=args.root.resolve()
    if args.mode=='audit':
        result=ast_identity(root);result['worker_sha256']=sha(__file__)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True));return
    auth,prereg=admission(root,args)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    c=Components(root);torch=c.torch;gpu=gpu_identity(torch,args,auth)
    phase=root/('formal/worker'+str(args.gpu) if args.mode=='formal' else 'qualification/gpu'+str(args.gpu)+'/'+args.mode)
    atomic_new(phase/'STARTED.json',{'task':TASK,'pid':os.getpid(),'mode':args.mode,'unix':time.time(),'authorization_sha256':sha(args.authorization)})
    try:
        model,tok,qmod,metadata=c.load()
        if args.mode=='calibration':
            sys.path.insert(0,str(root/'source'))
            from stepquant_calibration import run_calibration
            result=run_calibration(model,tok,qmod,root,read(root/'CALIBRATION_INPUTS.json'),
                                   progress_callback=lambda r:print(json.dumps(r,ensure_ascii=False,sort_keys=True),flush=True))
            result.update(authorization_sha256=sha(args.authorization),source_manifest_sha256=auth['source_manifest_sha256'],
                          runtime_source_manifest_sha256=auth['runtime_source_manifest_sha256'],gpu=gpu)
            atomic_new(phase/'RESULT.json',result)
            del model;gc.collect();torch.cuda.empty_cache();return
        artifact=torch.load(root/'calibration/qwen35_9b_stepquant6_plan.pt',map_location='cpu',weights_only=False)
        check(artifact['format_version']==2 and artifact['settings']['nominal_bits']==6, 'Full official6 Plan format changed')
        check(artifact['settings']['pivots']==32 and artifact['settings']['horizon']==2048 and artifact['settings']['candidates']==[4,6,8], 'Official full6 settings changed')
        bindings={'prereg_sha256':auth['prereg_sha256'],'plan_sha256':auth['plan_sha256'],
                  'source_manifest_sha256':auth['source_manifest_sha256'],'runtime_source_manifest_sha256':auth['runtime_source_manifest_sha256'],
                  'model_tokenizer_identity_sha256':auth['model_tokenizer_identity_sha256'],'authorization_sha256':sha(args.authorization)}
        if args.mode=='resource':
            result=resource_pressure(c,model,auth);result.update(bindings);atomic_new(phase/'RESULT.json',result)
        else:
            if args.mode=='formal':items=[r for r in units(root,auth) if r['gpu']==args.gpu]
            else:
                path=root/'CANARY_INPUT.json';check(sha(path)==auth['canary_input_sha256'],'Canary input changed')
                item=read(path);check(item['non_AIME'] is True and item['seed'] in (1,2),'Illegal canary identity')
                items=[item]
            for item in items:
                for method in item.get('method_order',list(METHODS)):
                    path=phase/(str(item['canonical_index']).zfill(2)+'_'+method+'.json')
                    check(not path.exists(),'Refuse rerun of completed/failed unit: '+str(path))
                    atomic_new(path.with_suffix('.ATTEMPT.json'),{'canonical_index':item['canonical_index'],'method':method,'pid':os.getpid(),'unix':time.time(),**bindings})
                    record=generate(c,model,tok,qmod,item,method,artifact,gpu,metadata,81920 if args.mode=='formal' else 128,args.mode!='formal')
                    record.update(bindings);record['formal_identity']={'task':TASK,'model':'Qwen3.5-9B','configuration':method,'problem_id':item['problem_id'],'seed':item['seed']}
                    record['output_payload_sha256']=canonical_sha(record);atomic_new(path,record)
                    check(record['status']=='COMPLETE','Technical failure: unit INCOMPLETE, first failure preserved')
                    if args.mode=='fresh':compare_canary(read(root/('qualification/gpu'+str(args.gpu)+'/canary')/path.name),record)
                    if args.mode!='formal':check(record['output_tokens']==128,'Canary ended before128 sampled tokens')
            result={'status':'PASS','mode':args.mode,'units':len(items)*2,'gpu':gpu,**bindings,
                    'canary_coverage':'new batched-prefill +128 actual sampled tokens' if args.mode!='formal' else None}
            atomic_new(phase/'RESULT.json',result)
        del model;gc.collect();torch.cuda.empty_cache()
    except Exception:
        atomic_new(phase/'FIRST_FAILURE.json',{'status':'INCOMPLETE','mode':args.mode,'pid':os.getpid(),'traceback':traceback.format_exc(),'unix':time.time()})
        raise


if __name__=='__main__':main()
