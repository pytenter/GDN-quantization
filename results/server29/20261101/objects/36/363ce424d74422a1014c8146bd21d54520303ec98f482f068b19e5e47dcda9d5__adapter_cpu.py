"""Real adapter control integration using explicitly CPU-only virtual mocks.

No torch, Triton, model, CUDA driver, NVML, remote, or scientific worker call.
The production adapter source executes unchanged; mocks prove call ordering
and state ownership only, never GPU math, QDQ kernels, or model consumption.
"""
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import types
import unittest
from unittest import mock
import weakref

ROOT=Path(__file__).resolve().parents[1]
LAYERS=tuple(i for i in range(32) if i%4!=3)
LOG=[]
TEMPORARIES=[]
INJECT_ENCODE_ERROR=False


def flatten(values):
    if isinstance(values,(list,tuple)):
        for value in values:yield from flatten(value)
    else:yield values


class BoolValue:
    def __init__(self,value):self.value=bool(value)
    def all(self):return self
    def item(self):return self.value
    def __bool__(self):return self.value
    def __and__(self,other):return BoolValue(self.value and bool(other))


class Tensor:
    """Virtual uniform-per-layer tensor; shapes/storage bytes are metadata."""
    def __init__(self,shape,payload,dtype='float32',device='cuda:0',label=None,numeric=None):
        self.shape=tuple(shape);self.payload=tuple(payload);self.dtype=dtype
        self.device=device;self.label=label;self.numeric=numeric
    def contiguous(self):return self
    def numel(self):return math.prod(self.shape)
    def element_size(self):return {'float32':4,'int8':1,'uint8':1,'bool':1}[self.dtype]
    def copy_(self,other):
        LOG.append(('copy',self.label));self.payload=other.payload;return self
    def __getitem__(self,index):
        if not isinstance(index,slice):raise TypeError('mock supports batch slices only')
        p=self.payload[index]
        return Tensor((len(p),*self.shape[1:]),p,self.dtype,self.device,self.label)
    def __ge__(self,x):return BoolValue(all(v>=x for v in self._values()))
    def __le__(self,x):return BoolValue(all(v<=x for v in self._values()))
    def _values(self):return self.numeric if self.numeric is not None else self.payload


class Encoded:
    def __init__(self,shape,codes,scales,device='cuda:0',decoded=None):
        self.shape=tuple(shape);b,h=shape[:2];self.method='RW'
        self.codes=Tensor((b,h,4,32,128),codes,'int8',device)
        self.scales=Tensor((b,h,4,128),scales,'float32',device)
        self.negative_zero_flags=Tensor((b,h,4,32,128),[False]*b,'bool',device)
        self.row_factors=Tensor((b,h,128),[1.]*b,'float32',device)
        self.lambda_indices=Tensor((b,h,4,128),[3]*b,'uint8',device)
        self.weighted_losses=Tensor((b,h,4,128),[.01]*b,'float32',device)
        self.zero_points=None;self._decoded=decoded
    def cpu(self):
        LOG.append(('encoded_cpu',))
        # Only integer codec fields and their legal scale/r metadata survive.
        return Encoded(self.shape,self.codes.payload,self.scales.payload,'cpu',None)


def decode(encoded):
    if encoded._decoded is not None:return encoded._decoded
    values=tuple(q*s*r for q,s,r in zip(encoded.codes.payload,encoded.scales.payload,
                                       encoded.row_factors.payload))
    return Tensor(encoded.shape,values,'float32','cpu','decoded_CPU_fields')


def encode(current,condition,omega,**kwargs):
    LOG.append(('encode',condition,tuple(current.shape),kwargs))
    if INJECT_ENCODE_ERROR:raise FloatingPointError('injected CPU mock nonfinite field')
    codes=tuple(round(x*2) for x in current.payload)
    scales=(.5,)*len(codes)
    rebuilt=Tensor(current.shape,[q*.5 for q in codes],'float32','cuda:0','decoded_transient')
    TEMPORARIES.append(weakref.ref(rebuilt))
    return Encoded(current.shape,codes,scales,decoded=rebuilt)


def slice_encoded(encoded,index):
    return Encoded((1,*encoded.shape[1:]),encoded.codes.payload[index:index+1],
        encoded.scales.payload[index:index+1],encoded.codes.device,
        None if encoded._decoded is None else encoded._decoded[index:index+1])


def require_consumed(actual,encoded):
    LOG.append(('consume_encoded_fields',actual.label,encoded.codes.device))
    expected=decode(encoded)
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise RuntimeError('mock next-consumption shape/dtype mismatch')
    if b''.join(struct.pack('<f',v) for v in actual.payload)!=\
       b''.join(struct.pack('<f',v) for v in expected.payload):
        raise RuntimeError('mock next-consumption codec bytes differ')
    return True


class HadamardPatch:
    def __init__(self,model):self.current_layer=None;self.entered=False
    def __enter__(self):self.entered=True;return self
    def __exit__(self,*args):self.entered=False


def _record_function(name):
    @contextmanager
    def cm():
        LOG.append(('range_begin',name))
        try:yield
        finally:LOG.append(('range_end',name))
    return cm()


def _tensor(values,dtype,device):
    dims=[];v=values
    while isinstance(v,list):dims.append(len(v));v=v[0] if v else 0
    numbers=tuple(flatten(values))
    return Tensor(dims,[numbers[0]]*dims[0],dtype,device,'static_omega',numbers)


def _cat(states,dim):
    if dim!=0:raise AssertionError('adapter must batch actual layers on B axis')
    LOG.append(('cat',len(states),dim))
    current=Tensor((sum(s.shape[0] for s in states),*states[0].shape[1:]),
        tuple(v for s in states for v in s.payload),'float32','cuda:0','current_transient')
    TEMPORARIES.append(weakref.ref(current))
    return current


def _native(query,key,value,*args,**kwargs):
    state=kwargs.get('initial_state',args[2] if len(args)>2 else None)
    LOG.append(('native_helper',None if state is None else state.label))
    return query,state


@contextmanager
def production_adapter(*,omega_value=1.0,selected_pair=(.5,.25),aime_used=False):
    torch=types.ModuleType('torch');torch.float32='float32';torch.tensor=_tensor;torch.cat=_cat
    torch.isfinite=lambda x:BoolValue(all(math.isfinite(v) for v in x._values()))
    torch.profiler=types.SimpleNamespace(record_function=_record_function)
    canonical=types.ModuleType('canonical_h128')
    canonical.QwenKeyHadamardPatch=HadamardPatch
    canonical.get_state=lambda cache,layer:cache[layer]
    codec=types.ModuleType('unified_codec')
    for name,value in dict(encode=encode,decode=decode,require_consumed=require_consumed,
                           slice_encoded=slice_encoded,PARAMETER_GRID=((.25,.25),(.5,.25),(.5,.5))).items():
        setattr(codec,name,value)
    modules={'torch':torch,'canonical_h128':canonical,'unified_codec':codec}
    parent=None
    for name in ('transformers','transformers.models','transformers.models.qwen3_5',
                 'transformers.models.qwen3_5.modeling_qwen3_5'):
        m=types.ModuleType(name);m.__path__=[];modules[name]=m
        if parent is not None:setattr(parent,name.rsplit('.',1)[1],m)
        parent=m
    native_module=parent
    native_module.torch_recurrent_gated_delta_rule=_native
    native_module.torch_chunk_gated_delta_rule=_native
    weights=dict(layer_ids=list(LAYERS),shape=[24,32,128],AIME_used=aime_used,
                 omega_normalized_fp32=[[[omega_value]*128 for _ in range(32)] for _ in range(24)])
    selected=dict(selected=dict(a=selected_pair[0],b=selected_pair[1]),AIME_used=False,status='PASS')
    actual_read_text=Path.read_text
    def read_fixture(path,*args,**kwargs):
        if path.name=='TRANSPORTED_QUERY_OMEGA.json':return json.dumps(weights)
        if path.name=='CALIBRATION_PARAMETER_SELECTION.json':return json.dumps(selected)
        return actual_read_text(path,*args,**kwargs)
    with mock.patch.dict(sys.modules,modules),mock.patch.object(Path,'read_text',read_fixture):
        p=ROOT/'src/unified_adapter.py'
        spec=importlib.util.spec_from_file_location('adapter_control_under_test',p)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module,native_module


def cache():
    return {layer:Tensor((1,32,128,128),[index+.26],label=layer)
            for index,layer in enumerate(LAYERS)}


class AdapterIntegration(unittest.TestCase):
    def setUp(self):
        global INJECT_ENCODE_ERROR
        LOG.clear();TEMPORARIES.clear();INJECT_ENCODE_ERROR=False
        self.callbacks=[]
    def callback(self,kind,token,layer,fields):
        # Store names only: callback does not itself retain tensors.
        self.callbacks.append((kind,token,layer,tuple(sorted(fields))))
        LOG.append(('callback',kind,token,layer))

    def test_01_all24_once_encode_before_any_copy(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'RW',LAYERS,callback=self.callback)
            c=cache();a.after_token(c)
            self.assertEqual([e for e in LOG if e[0]=='cat'],[('cat',24,0)])
            enc=[e for e in LOG if e[0]=='encode'];self.assertEqual(len(enc),1)
            self.assertEqual(enc[0][2],(24,32,128,128))
            self.assertIs(enc[0][3]['check_finite'],True)
            self.assertEqual(sum(e[0]=='copy' for e in LOG),24)
            self.assertLess(next(i for i,e in enumerate(LOG) if e[0]=='encode'),
                            next(i for i,e in enumerate(LOG) if e[0]=='copy'))
            self.assertEqual([e[2] for e in self.callbacks if e[0]=='PRE_QDQ'],list(LAYERS))
            self.assertEqual([e[2] for e in self.callbacks if e[0]=='POST_QDQ'],list(LAYERS))
            self.assertGreater(next(i for i,e in enumerate(LOG) if e[:2]==('callback','POST_QDQ')),
                               max(i for i,e in enumerate(LOG) if e[0]=='copy'))
            self.assertEqual(a.current_token,1);self.assertEqual(a.qdq_calls,1)
            self.assertEqual([c[l].payload for l in LAYERS],[(i+.5,) for i in range(24)])

    def test_02_wrapped_native_consumes_cpu_encoded_fields(self):
        with production_adapter() as (module,Q):
            a=module.UnifiedQDQ(object(),'W',LAYERS,callback=self.callback);c=cache()
            q=Tensor((1,1,32,128),[0.]);original=Q.torch_recurrent_gated_delta_rule
            with a:
                a.after_token(c)
                for index,layer in enumerate(LAYERS):
                    a.h.current_layer=layer
                    if index%2:Q.torch_chunk_gated_delta_rule(q,q,q,None,None,c[layer])
                    else:Q.torch_recurrent_gated_delta_rule(q,q,q,initial_state=c[layer])
                self.assertEqual(a.consumption_calls,24)
                self.assertEqual(a.consumption_layers,set(LAYERS))
                checks=[e for e in LOG if e[0]=='consume_encoded_fields']
                self.assertEqual(len(checks),48) #24 immediate +24actual wrapper-entry checks
                self.assertTrue(all(e[2]=='cpu' for e in checks))
                self.assertEqual(sum(e[0]=='native_helper' for e in LOG),24)
            self.assertIs(Q.torch_recurrent_gated_delta_rule,original)
            self.assertFalse(a.h.entered)

    def test_03_no_fp_carrier_shadow_retained(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'R',LAYERS);c=cache();a.after_token(c);gc.collect()
            self.assertEqual(len(a.previous),24)
            for e in a.previous.values():
                self.assertEqual(e.codes.device,'cpu');self.assertEqual(e.codes.dtype,'int8')
                self.assertIsNone(e._decoded)
                self.assertFalse(any(isinstance(v,Tensor) and v.dtype=='float32' and
                    v.shape[-2:]==(128,128) for v in e.__dict__.values()))
            self.assertTrue(TEMPORARIES)
            self.assertTrue(all(ref() is None for ref in TEMPORARIES))
            self.assertFalse(any(isinstance(v,Tensor) and v.shape[-2:]==(128,128)
                                 for v in a.__dict__.values()))

    def test_04_nonfinite_encoder_error_atomic_no_writeback(self):
        global INJECT_ENCODE_ERROR
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'RW',LAYERS,callback=self.callback);c=cache()
            before={k:v.payload for k,v in c.items()};INJECT_ENCODE_ERROR=True
            with self.assertRaises(FloatingPointError):a.after_token(c)
            self.assertEqual({k:v.payload for k,v in c.items()},before)
            self.assertEqual(sum(e[0]=='copy' for e in LOG),0)
            self.assertEqual(a.qdq_calls,0);self.assertEqual(a.current_token,0)
            self.assertEqual(a.previous,{})
            self.assertFalse(any(e[0]=='POST_QDQ' for e in self.callbacks))

    def test_05_corruption_rejected_before_native_helper(self):
        with production_adapter() as (module,Q):
            a=module.UnifiedQDQ(object(),'W',LAYERS,callback=self.callback);c=cache()
            q=Tensor((1,1,32,128),[0.])
            with a:
                a.after_token(c);a.h.current_layer=LAYERS[0];c[LAYERS[0]].payload=(999.,)
                with self.assertRaises(RuntimeError):Q.torch_recurrent_gated_delta_rule(q,q,q,initial_state=c[LAYERS[0]])
                self.assertEqual(a.consumption_calls,0)
                self.assertFalse(any(e[0]=='native_helper' for e in LOG))
                self.assertTrue(any(e[0]=='NEXT_CONSUMPTION_FAILURE' for e in self.callbacks))

    def test_06_invalid_carrier_rejected_before_cat_or_encode(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'W',LAYERS);c=cache();c[LAYERS[-1]].dtype='float16'
            with self.assertRaises(RuntimeError):a.after_token(c)
            self.assertFalse(any(e[0] in ('cat','encode','copy') for e in LOG))

    def test_07_real_shape_metadata_accounting(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'RW',LAYERS);a.after_token(cache());m=a.metadata
            self.assertEqual(m['FP32_carrier_bytes'],48*1024**2)
            self.assertEqual(m['temporary_current_FP32_bytes'],48*1024**2)
            self.assertEqual(m['temporary_decoded_FP32_bytes'],48*1024**2)
            self.assertEqual(m['codes_bytes'],12*1024**2)
            self.assertEqual(m['scales_bytes'],int(1.5*1024**2))
            self.assertEqual(m['omega_bytes'],384*1024)
            self.assertEqual(m['row_factors_bytes'],384*1024)

    def test_08_formal_observe_false_still_immediate_writeback(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'RW',LAYERS,observe=False);c=cache();a.after_token(c)
            self.assertEqual(a.previous,{})
            self.assertEqual(sum(e[0]=='copy' for e in LOG),24)
            self.assertEqual(sum(e[0]=='encode' for e in LOG),1)
            gc.collect();self.assertTrue(all(ref() is None for ref in TEMPORARIES))

    def test_09_sequence_mismatch_fails_before_mutation(self):
        with production_adapter() as (module,_):
            a=module.UnifiedQDQ(object(),'W',LAYERS);a.current_token=1
            with self.assertRaises(RuntimeError):a.after_token(cache())
            self.assertFalse(LOG)

    def test_10_multitoken_helper_rejected_and_hooks_restored(self):
        with production_adapter() as (module,Q):
            a=module.UnifiedQDQ(object(),'W',LAYERS);original=Q.torch_chunk_gated_delta_rule
            q=Tensor((1,2,32,128),[0.])
            with a:
                with self.assertRaises(RuntimeError):Q.torch_chunk_gated_delta_rule(q,q,q)
                self.assertFalse(any(e[0]=='native_helper' for e in LOG))
            self.assertIs(Q.torch_chunk_gated_delta_rule,original)

    def test_11_static_omega_nonfinite_and_outside_clamp_rejected(self):
        for bad in (float('nan'),float('inf'),0.0,1/32,32.0):
            with production_adapter(omega_value=bad) as (module,_):
                with self.assertRaises(RuntimeError):module.UnifiedQDQ(object(),'RW',LAYERS)
        self.assertFalse(any(e[0] in ('cat','encode','copy') for e in LOG))

    def test_12_unfrozen_row_parameter_pair_rejected(self):
        with production_adapter(selected_pair=(.2,.25)) as (module,_):
            with self.assertRaises(RuntimeError):module.UnifiedQDQ(object(),'R',LAYERS)
        self.assertFalse(any(e[0] in ('cat','encode','copy') for e in LOG))

    def test_13_aime_derived_omega_rejected(self):
        with production_adapter(aime_used=True) as (module,_):
            with self.assertRaises(RuntimeError):module.UnifiedQDQ(object(),'W',LAYERS)
        self.assertFalse(any(e[0] in ('cat','encode','copy') for e in LOG))


def child():
    p=ROOT/'src/unified_adapter.py';before=hashlib.sha256(p.read_bytes()).hexdigest()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AdapterIntegration)
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stderr).run(suite)
    after=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(status='PASS' if result.wasSuccessful() and before==after else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256_before=before,source_sha256_after=after,
        mocked=['torch tensors/profiler','canonical Hadamard context','native helper',
                'codec encode/decode/CPU fields','frozen-calibration JSON reads'],
        torch_import='virtual CPU-only mock, no actual torch imported',
        gpu_calls=0,model_calls=0,remote_calls=0,
        scope='real production adapter call-order/ownership integration with explicit CPU mocks; NOT GPU QDQ or actualmodel consumption proof')


if __name__=='__main__':
    if '--child' in sys.argv:
        v=child();print(json.dumps(v));raise SystemExit(v['status']!='PASS')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    runs=[]
    for index in range(2):
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child'],
                            text=True,capture_output=True,env=env,timeout=45)
        print(proc.stderr,end='',file=sys.stderr)
        try:run=json.loads(proc.stdout)
        except json.JSONDecodeError:run=dict(status='FAIL',stdout=proc.stdout,returncode=proc.returncode)
        run['fresh_process_index']=index+1;runs.append(run)
    same_source=len({r.get('source_sha256_after') for r in runs})==1
    v=dict(status='PASS' if all(r['status']=='PASS' for r in runs) and same_source else 'FAIL',
        fresh_processes=runs,source_consistent_between_processes=same_source,
        independent_gpu_proof=False,tests_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=ROOT/'operations/ADAPTER_CPU_INTEGRATION.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(v,ensure_ascii=False));raise SystemExit(v['status']!='PASS')
