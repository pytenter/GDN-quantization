"""Real Torch CPU diagnostic entry coverage; kernels/models/devices are forbidden.

CPU_PAYLOAD supplies exact pinned source text via stdin. Codec encode and the
legacy baseline are explicit scalar-reference CPU fixtures only. This checks
production Encoded/summary/exact/_verify_panel plumbing, not GPU mathematics.
"""
import ctypes, hashlib, io, json, os, struct, sys, types, unittest
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
assert not torch.cuda._initialized
forbidden_calls=[]
def forbidden(*args,**kwargs):
    forbidden_calls.append('GPU_DRIVER_KERNEL_MODEL_FORBIDDEN')
    raise AssertionError('CPU-only entry fixture attempted a GPU/driver/kernel call')
for name in ('_lazy_init','init','is_available','device_count','current_device',
             'get_device_properties','set_device','synchronize'):
    setattr(torch.cuda,name,forbidden)
if hasattr(torch._C,'_cuda_init'):torch._C._cuda_init=forbidden
torch.Tensor.cuda=forbidden

def source_module(name,key):
    m=types.ModuleType(name);m.__file__='/CPU_STDIN/'+key
    sys.modules[name]=m
    exec(compile(CPU_PAYLOAD[key],m.__file__,'exec'),m.__dict__)
    return m

F=source_module('actual_byteview_field_module','src/gpu_codec_fields.py')
C=source_module('actual_byteview_codec_module','src/unified_codec.py')
M=source_module('unchanged_byteview_scalar_reference','reference/math_reference.py')
O=source_module('unchanged_byteview_group32_cpu_module','inherited/group32/src/group_codecs.py')
C._kernels=forbidden
ROOT=Path('/CPU_FIXTURE_BYTEVIEW_ROOT')
PANEL_CALLS=[]

def raw_memory(t):
    assert t.device.type=='cpu'
    t=t.detach().contiguous()
    return ctypes.string_at(t.data_ptr(),t.numel()*t.element_size()) if t.numel() else b''

def actual_encoded(method,*,noncontiguous=False):
    shape=(1,2,128,128)
    def layout(dtype,shape,value):
        if noncontiguous and len(shape)>1:
            reverse=(*shape[:-2],shape[-1],shape[-2])
            return torch.full(reverse,value,dtype=dtype,device='cpu').transpose(-1,-2)
        return torch.full(shape,value,dtype=dtype,device='cpu')
    e=C.Encoded(method,shape,
        layout(torch.int8,(1,2,4,32,128),0),
        layout(torch.float32,(1,2,4,128),.125),None,
        layout(torch.bool,(1,2,4,32,128),True),
        layout(torch.float32,(1,2,128),1.),
        layout(torch.uint8,(1,2,4,128),3),
        layout(torch.float32,(1,2,4,128),0.),
        torch.tensor(0,dtype=torch.int32,device='cpu'),
        torch.zeros(shape,dtype=torch.float32,device='cpu'))
    return e

def reference_fixture_encode(x,condition,omega,a=None,b=None):
    """Explicit CPU mock of encode; deliberately never calls production kernel."""
    PANEL_CALLS.append(dict(condition=condition,device=str(x.device),shape=list(x.shape)))
    pieces=[]
    for bidx in range(x.shape[0]):
        sx=M.Tensor32((1,x.shape[1],128,128),x[bidx:bidx+1].reshape(-1).tolist())
        w=omega[bidx] if omega.ndim==3 else omega
        sw=M.Tensor32((x.shape[1],128),w.reshape(-1).tolist())
        q=M.quantize(sx,sw,C.METHODS[condition],a=a,b=b)
        e=C.Encoded(condition,q.shape,
            torch.tensor(list(q.codes),dtype=torch.int8).reshape(1,x.shape[1],4,32,128),
            torch.tensor(q.scales.values,dtype=torch.float32).reshape(1,x.shape[1],4,128),None,
            torch.tensor(list(q.negative_zero),dtype=torch.bool).reshape(1,x.shape[1],4,32,128),
            torch.tensor(q.row_factors.values,dtype=torch.float32).reshape(1,x.shape[1],128),
            torch.tensor(list(q.candidate_indices),dtype=torch.uint8).reshape(1,x.shape[1],4,128),
            torch.tensor(q.losses.values,dtype=torch.float32).reshape(1,x.shape[1],4,128),
            torch.tensor(0,dtype=torch.int32),
            torch.tensor(M.decode(q).values,dtype=torch.float32).reshape(q.shape))
        pieces.append(e)
    if len(pieces)==1:return pieces[0]
    cat=lambda attr:torch.cat([getattr(e,attr) for e in pieces],dim=0)
    return C.Encoded(condition,tuple(x.shape),cat('codes'),cat('scales'),None,
        cat('negative_zero_flags'),cat('row_factors'),cat('lambda_indices'),
        cat('weighted_losses'),torch.tensor(0,dtype=torch.int32),cat('_decoded'))

class EncodedCPUChecks(unittest.TestCase):
    def test_01_complete_actual_Encoded_cpu_and_summary_all_methods(self):
        for method in ('base','W','R','RW'):
            e=actual_encoded(method);cpu=e.cpu();s=F._field_summary(cpu)
            self.assertIsNone(cpu._decoded);self.assertIsNone(cpu.zero_points)
            self.assertEqual(s['method'],method);self.assertEqual(s['invalid_flags']['shape'],[])
            self.assertEqual(s['invalid_flags']['dtype'],'torch.int32')
            self.assertEqual(s['invalid_flags']['bytes'],4)
            self.assertEqual(s['invalid_flags']['first_values'],[0])
            for name in ('codes','scales','negative_zero_flags','row_factors','lambda_indices','weighted_losses','invalid_flags'):
                value=getattr(cpu,name);self.assertEqual(s[name]['sha256'],hashlib.sha256(raw_memory(value)).hexdigest())
                self.assertEqual(s[name]['bytes'],value.numel()*value.element_size())
            self.assertTrue(C.validate_encoded(cpu))
            json.dumps(s,allow_nan=False)

    def test_02_full_schema_dtype_set(self):
        e=actual_encoded('RW')
        expected={'codes':torch.int8,'scales':torch.float32,'negative_zero_flags':torch.bool,
            'row_factors':torch.float32,'lambda_indices':torch.uint8,
            'weighted_losses':torch.float32,'invalid_flags':torch.int32}
        for name,dtype in expected.items():self.assertEqual(getattr(e,name).dtype,dtype)
        self.assertEqual(set(F._field_summary(e))-{'method','shape'},set(expected))

    def test_03_all_actual_fields_exact_gate_real_Torch(self):
        for method in ('base','W','R','RW'):
            e=actual_encoded(method).cpu()
            for name in ('codes','scales','negative_zero_flags','row_factors','lambda_indices','weighted_losses','invalid_flags'):
                value=getattr(e,name);expected=raw_memory(value)
                F._exact(value,expected,name,dict(case='CPU_FULL_ENCODED'))
                broken=bytearray(expected);broken[0]^=1
                context=dict(case='CPU_REAL_FIELD_CORRUPTION')
                with self.assertRaises(AssertionError):F._exact(value,bytes(broken),name,context)
                self.assertEqual(context['mismatch']['field'],name)

    def test_04_actual_fields_noncontiguous_and_clone(self):
        for method in ('base','W','R','RW'):
            e=actual_encoded(method,noncontiguous=True)
            self.assertFalse(e.codes.is_contiguous());self.assertFalse(e.scales.is_contiguous())
            s=F._field_summary(e.cpu())
            for name in ('codes','scales','negative_zero_flags','row_factors','lambda_indices','weighted_losses','invalid_flags'):
                value=getattr(e,name)
                self.assertEqual(F._bytes(value),raw_memory(value))
                self.assertEqual(s[name]['sha256'],hashlib.sha256(raw_memory(value)).hexdigest())

    def test_05_actual_Encoded_empty_diagnostic_fields(self):
        for method in ('base','W','R','RW'):
            e=actual_encoded(method)
            for name in ('codes','scales','negative_zero_flags','row_factors','lambda_indices','weighted_losses','invalid_flags'):
                setattr(e,name,torch.empty((2,0,3),dtype=getattr(e,name).dtype))
            s=F._field_summary(e.cpu())
            for name in ('codes','scales','negative_zero_flags','row_factors','lambda_indices','weighted_losses','invalid_flags'):
                self.assertEqual(s[name]['bytes'],0);self.assertEqual(s[name]['first_values'],[])
                F._exact(getattr(e,name),b'',name,dict(case='EMPTY_DIAGNOSTIC_ONLY'))

    def test_06_negative_zero_decode_byte_gate_and_optional_fields(self):
        e=actual_encoded('base').cpu();decoded=C.decode(e)
        self.assertEqual(F._bytes(decoded)[:4],struct.pack('=f',-0.))
        self.assertNotEqual(F._bytes(decoded),F._bytes(torch.zeros_like(decoded)))
        with self.assertRaises(AssertionError):F._exact(decoded,raw_memory(torch.zeros_like(decoded)),'decode sign',dict(case='SIGNEDZERO'))
        e.weighted_losses=None;e.invalid_flags=None
        s=F._field_summary(e);self.assertNotIn('weighted_losses',s);self.assertNotIn('invalid_flags',s)

    def test_07_scalar_invalid_flags_still_reject_and_other_scalar_types(self):
        e=actual_encoded('W');e.invalid_flags=torch.tensor(1,dtype=torch.int32)
        self.assertEqual(F._field_summary(e)['invalid_flags']['first_values'],[1])
        with self.assertRaises(FloatingPointError):C.validate_encoded(e)
        for dtype,val in ((torch.bool,False),(torch.int8,-1),(torch.uint8,255),
            (torch.int16,-17),(torch.int64,1<<40),(torch.float16,-0.),
            (torch.bfloat16,1.25),(torch.float32,-0.),(torch.float64,1.25)):
            e.invalid_flags=torch.tensor(val,dtype=dtype)
            self.assertEqual(F._field_summary(e)['invalid_flags']['sha256'],hashlib.sha256(raw_memory(e.invalid_flags)).hexdigest())

    def test_08_actual_verify_panel_CPU_fixture_all_four_methods(self):
        original_load=F._load;original_encode=C.encode;legacy_encode=O.encode
        exact_calls=[];original_exact=F._exact
        def track_exact(actual,expected,name,context):
            exact_calls.append(name);return original_exact(actual,expected,name,context)
        O.encode=lambda x,method:legacy_encode(x,method,cpu_semantics='CUDA_LEGACY_EMULATED')
        F._load=lambda path,name:O if str(path).endswith('inherited/group32/src/group_codecs.py') else original_load(path,name)
        C.encode=reference_fixture_encode;F._exact=track_exact
        try:
            x=torch.tensor([-0. if i%2 else 0. for i in range(16384)],dtype=torch.float32).reshape(1,1,128,128)
            omega=torch.ones((1,1,128),dtype=torch.float32)
            for method in ('base','W','R','RW'):
                context={};result=F._verify_panel(ROOT,Path('/CPU_TARGET'),dict(id='CPU_'+method,condition=method),C,M,x,omega,.5,.25,'CPU_MOCK_ZERO_SIGNED_'+method,context)
                self.assertEqual(result['status'],'PASS')
                self.assertEqual(context['actual_fields_summary']['invalid_flags']['shape'],[])
                self.assertEqual(context['actual_fields_summary']['invalid_flags']['bytes'],4)
            required={'original baseline codes','original baseline scales','original baseline negative_zero_flags','original baseline decoded','selected signed codes','selected scales','signed-zero flags','candidate choice/tie','selected weighted losses','CPU decode of actual fields','GPU decode vs CPU-field decode'}
            self.assertEqual(set(exact_calls),required)
            self.assertEqual(len(exact_calls),4*len(required))
        finally:
            F._load=original_load;C.encode=original_encode;O.encode=legacy_encode;F._exact=original_exact

    def test_09_actual_verify_panel_CPU_fixture_failure_enters_exact_gate(self):
        original_load=F._load;original_encode=C.encode;legacy_encode=O.encode
        O.encode=lambda x,method:legacy_encode(x,method,cpu_semantics='CUDA_LEGACY_EMULATED')
        F._load=lambda path,name:O if str(path).endswith('inherited/group32/src/group_codecs.py') else original_load(path,name)
        def corrupt(x,condition,omega,a=None,b=None):
            e=reference_fixture_encode(x,condition,omega,a,b)
            if condition=='W':e.codes.reshape(-1)[0]=1
            return e
        C.encode=corrupt;context={}
        try:
            with self.assertRaises(AssertionError):F._verify_panel(ROOT,Path('/CPU_TARGET'),dict(id='CPU_BAD_W',condition='W'),C,M,torch.zeros((1,1,128,128)),torch.ones((1,1,128)),.5,.25,'CPU_MOCK_CORRUPT',context)
            self.assertEqual(context['mismatch']['field'],'selected signed codes')
            self.assertEqual(context['actual_fields_summary']['invalid_flags']['bytes'],4)
            self.assertIn('candidate_loss_evidence',context)
        finally:F._load=original_load;C.encode=original_encode;O.encode=legacy_encode

suite=unittest.defaultTestLoader.loadTestsFromTestCase(EncodedCPUChecks)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not forbidden_calls and not torch.cuda._initialized
print(json.dumps(dict(status='PASS_REAL_TORCH_ENCODED_AND_ACTUAL_PANEL_CPU_FIXTURES' if result.wasSuccessful() else 'FAIL',
    tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    torch_version=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_initialized=False,
    source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in CPU_PAYLOAD.items()},
    actual_panel_entry=True,encode_and_legacy_baseline_explicit_CPU_fixture_only=True,
    methods=['base','W','R','RW'],actual_panel_CPU_mock_encode_calls=PANEL_CALLS,
    GPU_calls=0,driver_calls=0,kernel_calls=0,model_loads=0,compile_calls=0,
    remote_file_writes=0,GPU_qualification_credit=0,AIME_samples=0)))
raise SystemExit(not result.wasSuccessful())
