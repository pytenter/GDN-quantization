"""CPU checks of bounded failure evidence and explicit-target IR audit gates.

Model/device execution is never imported. These checks grant no GPU credit.
"""
from pathlib import Path
import importlib.util,json,struct,sys,tempfile,types,unittest
R=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
F=load('actual_field_evidence_module',R/'src/gpu_codec_fields.py')
P=load('actual_CPU_compile_audit_module',R/'reference/compile_sm86_cpu_only.py')
M=load('unchanged_scalar_reference_field_evidence',R/'reference/math_reference.py')
K=load('new_synthetic_panels_field_evidence',R/'reference/reduction_repair_cases.py')
class BytesTensor:
 def __init__(self,raw,width=4):self.raw=raw;self.width=width
 def element_size(self):return self.width
class Flat:
 def __init__(self,x):self.x=list(x)
 def reshape(self,*args):return self
 def tolist(self):return list(self.x)
class EvidenceChecks(unittest.TestCase):
 def test_01_exact_gate_preserves_signed_zero_difference(self):
  old=F._bytes;F._bytes=lambda t:t.raw
  try:
   c={'case':'CPU_SIGNED_ZERO'}
   with self.assertRaises(AssertionError):F._exact(BytesTensor(struct.pack('<f',-0.)),struct.pack('<f',0.),'selected weighted losses',c)
   self.assertEqual(c['mismatch']['first_differences'][0]['actual_hex'],'00000080')
  finally:F._bytes=old
 def test_02_evidence_is_bounded_and_identical_bytes_pass(self):
  old=F._bytes;F._bytes=lambda t:t.raw
  try:
   F._exact(BytesTensor(b'a'*200,1),b'a'*200,'codes',{'case':'CPU_IDENTICAL'})
   c={'case':'CPU_BOUNDED'}
   with self.assertRaises(AssertionError):F._exact(BytesTensor(b'a'*200,1),b'b'*200,'codes',c)
   self.assertEqual(len(c['mismatch']['first_differences']),24)
  finally:F._bytes=old
 def test_03_first_failure_persists_and_no_overwrite(self):
  original_run=F._run;prior=sys.modules.get('execution_control');fake=types.ModuleType('execution_control')
  def atomic_new(p,x):
   with Path(p).open('x') as out:json.dump(x,out,allow_nan=False)
  fake.atomic_new=atomic_new;sys.modules['execution_control']=fake
  def failed(root,target,unit,context):context['case']='CPU_MOCK_FAILURE_NO_GPU';raise AssertionError('CPU_ONLY_EXPECTED')
  F._run=failed
  try:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'CODEC_FIELDS_FIRST_FAILURE_SUMMARY.json'
    with self.assertRaises(AssertionError):F.run(R,Path(d),dict(id='CPU_TEST',condition='W'))
    first=p.read_bytes();x=json.loads(first);self.assertFalse(x['scientific_credit']);self.assertFalse(x['raw_model_or_tensor_dump'])
    with self.assertRaises(AssertionError):F.run(R,Path(d),dict(id='CPU_TEST',condition='W'))
    self.assertEqual(p.read_bytes(),first)
  finally:
   F._run=original_run
   if prior is None:sys.modules.pop('execution_control',None)
   else:sys.modules['execution_control']=prior
 def test_04_actual_near_tie_failure_summary_has_five_oracle_losses(self):
  sx=M.Tensor32((1,1,128,128),K.all_group_tile_near_tie_values());sw=M.Tensor32((1,128),K.all_group_tile_near_tie_weights());e=M.quantize(sx,sw,'wscale')
  choices=list(e.candidate_indices);self.assertEqual(choices[3],3);choices[3]=4
  actual=types.SimpleNamespace(lambda_indices=Flat(choices),weighted_losses=Flat(e.losses.values))
  r=M.Tensor32((1,1,128),[1.]*128);out=F._candidate_evidence(M,sx,sw,r,e,actual)
  self.assertEqual(out[0]['column'],3);self.assertEqual(out[0]['reference_five_candidate_argmin'],3)
  self.assertEqual(out[0]['reference_candidate_loss_hex'][3],out[0]['reference_candidate_loss_hex'][4])
  self.assertEqual(out[0]['actual_selected_index'],4);self.assertEqual(out[0]['reference_best_runner_up_margin'],0.)
 def test_05_row_ULP_summary_keeps_worst_even_beyond_first24(self):
  distances=[1]*128;distances[111]=17
  x=F._row_factor_evidence([1.]*128,[1.]*128,distances)
  self.assertEqual(x['max_ulp'],17);self.assertEqual(x['worst_differences'][0]['row'],111)
  self.assertEqual(len(x['worst_differences']),8);self.assertEqual(len(x['first_differences']),24)
 def test_06_strict_IR_shape_acceptance(self):
  x=dict(ttgir='arith.addf %a, %b : tensor<16xf32>',llir='%v = fadd float %a, %b',ptx='.target sm_86\nadd.f32 %f0, %f1, %f2;')
  self.assertEqual(P.ir_checks(x)['generic_float_sum_regions'],0)
 def test_07_old_generic_float_reduction_is_rejected(self):
  # Exact operation form from the old cached SM86 TTGIR float reducer.
  old='\"tt.reduce\"(%65) <{axis = 0 : i32}> ({\n^bb0(%arg10: f32, %arg11: f32):\n%360 = arith.addf %arg10, %arg11 : f32\ntt.reduce.return %360 : f32\n})'
  with self.assertRaises(RuntimeError):P.ir_checks(dict(ttgir=old,llir='%v = fadd float %a, %b',ptx='.target sm_86'))
 def test_08_fast_reassociation_fusion_wrong_target_are_rejected(self):
  for llir,ptx in [('%v = fadd fast float %a, %b','.target sm_86'),('%v = fadd reassoc float %a, %b','.target sm_86'),('%v = fadd float %a, %b','.target sm_86\nfma.rn.f32 %f0,%f1,%f2,%f3;'),('%v = fadd float %a, %b','.target sm_80')]:
   with self.assertRaises(RuntimeError):P.ir_checks(dict(ttgir='arith.addf %a,%b : f32',llir=llir,ptx=ptx))


 def test_09_only_declared_log_libdevice_FMA_with_row_debug_scope_is_permitted(self):
  ir='%v = fadd float %a, %b\n%x = call float @llvm.nvvm.fma.rn.f(float %a,float %b,float %c), !dbg !35\n!7 = distinct !DISubprogram(name: "_row_kernel")\n!35 = !DILocation(line: 116, column: 50, scope: !7)'
  asm=dict(ttgir='arith.addf %a,%b : tensor<16x16xf32>',llir=ir,ptx='.target sm_86\nfma.rn.f32 %f0,%f1,%f2,%f3;')
  x=P.ir_checks(asm,{116},{'16x16':1});self.assertEqual(x['permitted_libdevice_log_FMA_calls'],1)
  with self.assertRaises(RuntimeError):P.ir_checks(asm,{117},{'16x16':1})
  with self.assertRaises(RuntimeError):P.ir_checks(asm,{116},{'8x16':1})

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceChecks);r=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps(dict(status='PASS_BOUNDED_FIELD_EVIDENCE_AND_CPU_IR_GATES' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),GPU_calls=0,model_loads=0,compile_calls=0)))
 raise SystemExit(not r.wasSuccessful())
