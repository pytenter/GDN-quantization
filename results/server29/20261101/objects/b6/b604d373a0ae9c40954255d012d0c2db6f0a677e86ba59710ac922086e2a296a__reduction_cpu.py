"""CPU proofs of the actual new helper graph, not GPU qualification.

Only reshape/permute/split and FP32 add are simulated. The numerical oracle
and its bytes remain the original frozen independent scalar reference.
"""
from contextlib import contextmanager
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
PACK=struct.Struct('<f')


def f32(value):return PACK.unpack(PACK.pack(float(value)))[0]
def bits(value):return PACK.pack(value)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


CODEC=load(ROOT/'src/unified_codec.py','repaired_codec_CPU_graph_only')
REF=load(ORIGINAL/'reference/math_reference.py','original_frozen_scalar_reference')


class Tensor:
    def __init__(self,shape,values,add):
        self.shape=tuple(shape);self.values=tuple(values);self.add=add
        assert math.prod(self.shape)==len(self.values)
    def __add__(self,other):
        assert self.shape==other.shape and self.add is other.add
        return Tensor(self.shape,(self.add(a,b) for a,b in zip(self.values,other.values)),self.add)


class LogicalShapeCPU:
    """Explicit logical indexing only; no fake CUDA/Triton execution claim."""
    @staticmethod
    def static_assert(value,message):
        if not value:raise AssertionError(message)
    @staticmethod
    def reshape(value,shape):return Tensor(shape,value.values,value.add)
    @staticmethod
    def permute(value,order):
        assert sorted(order)==list(range(len(value.shape)))
        shape=tuple(value.shape[i] for i in order);out=[]
        for flat in range(math.prod(shape)):
            rem=flat;index=[0]*len(shape)
            for i in reversed(range(len(shape))):index[i]=rem%shape[i];rem//=shape[i]
            old=[0]*len(shape)
            for i,axis in enumerate(order):old[axis]=index[i]
            offset=0
            for axis,extent in enumerate(value.shape):offset=offset*extent+old[axis]
            out.append(value.values[offset])
        return Tensor(shape,out,value.add)
    @staticmethod
    def split(value):
        assert value.shape[-1]==2
        return (Tensor(value.shape[:-1],value.values[0::2],value.add),
                Tensor(value.shape[:-1],value.values[1::2],value.add))


CODEC.tl=LogicalShapeCPU


def fp_add(a,b):return f32(f32(a)+f32(b))
def symbolic_add(a,b):return ('FP32_ADD',a,b)


def actual_helper_columns(matrix):
    n=len(matrix);width=len(matrix[0]);assert all(len(row)==width for row in matrix)
    value=Tensor((n,width),(f32(x) for row in matrix for x in row),fp_add)
    return CODEC._halves_sum_first(value,n,width).values


def helper_tree_sum(values):
    assert len(values) in (32,128)
    return actual_helper_columns([[v] for v in values])[0]


def ordered_reference_tree(n,column):
    # Halves-first pairing is a balanced tree whose leaves are in bit-reversed
    # index order. This construction does not run the new helper algorithm.
    width=(n-1).bit_length()
    leaves=[('INPUT',int(format(i,'0%db'%width)[::-1],2),column) for i in range(n)]
    def balanced(items):
        if len(items)==1:return items[0]
        half=len(items)//2
        return ('FP32_ADD',balanced(items[:half]),balanced(items[half:]))
    return balanced(leaves)


def layout_roundtrip(matrix,rows_per_warp):
    """Pack/unpack independent logical indices through virtual GPU layouts."""
    n=len(matrix);columns=len(matrix[0]);column_lanes=32//rows_per_warp
    assert columns%column_lanes==0 and n%rows_per_warp==0
    registers=columns//column_lanes;physical={}
    for row in range(n):
        warp=row//rows_per_warp;row_lane=row%rows_per_warp
        for col in range(columns):
            lane=row_lane*column_lanes+col//registers;register=col%registers
            key=(warp,lane,register);assert key not in physical
            physical[key]=matrix[row][col]
    restored=[]
    for row in range(n):
        warp=row//rows_per_warp;row_lane=row%rows_per_warp
        restored.append([physical[(warp,row_lane*column_lanes+col//registers,col%registers)]
                         for col in range(columns)])
    return restored


def all_group_tile_near_tie_values():
    """32 known sensitive columns: every Key group and every Value tile.

    Only synthetic structured input and powers of two are used. Each target
    column is the original group0/column3 counterexample, scaled exactly by a
    positive power of two; no AIME/calibration/answer-dependent construction.
    """
    values=[0.]*16384;omega=[]
    for group in range(4):
        omega.extend(2.**((row%9)-4) for row in range(32))
        for row in range(32):
            for column in range(128):
                original=row*128+column
                values[(group*32+row)*128+column]=f32(math.sin(original*.731)*
                    (.01+(row%29)/9)*2.**group)
            tie=f32(math.sin((row*128+3)*.731)*(.01+(row%29)/9))
            for tile in range(8):values[(group*32+row)*128+3+16*tile]=f32(tie*2.**(group+tile))
    return values,omega


@contextmanager
def reference_with_actual_new_helper():
    original=REF.tree_sum;REF.tree_sum=helper_tree_sum
    try:yield
    finally:REF.tree_sum=original


def encoded_equal(left,right):
    return (left.codes.tobytes()==right.codes.tobytes() and left.scales.bytes()==right.scales.bytes()
        and left.negative_zero==right.negative_zero and left.row_factors.bytes()==right.row_factors.bytes()
        and left.candidate_indices==right.candidate_indices and left.losses.bytes()==right.losses.bytes()
        and REF.decode(left).bytes()==REF.decode(right).bytes())


class ReductionProofs(unittest.TestCase):
    def test_01_actual_Key32_graph_is_independent_bit_reversal_tree_for_every_column(self):
        width=16
        symbols=Tensor((32,width),(('INPUT',i,j) for i in range(32) for j in range(width)),symbolic_add)
        actual=CODEC._halves_sum_first(symbols,32,width)
        self.assertEqual(actual.shape,(width,))
        for col,tree in enumerate(actual.values):self.assertEqual(tree,ordered_reference_tree(32,col))
    def test_02_actual_Value128_graph_is_independent_bit_reversal_tree_for_every_row(self):
        width=32
        symbols=Tensor((128,width),(('INPUT',i,j) for i in range(128) for j in range(width)),symbolic_add)
        actual=CODEC._halves_sum_first(symbols,128,width)
        for col,tree in enumerate(actual.values):self.assertEqual(tree,ordered_reference_tree(128,col))
    def test_03_signed_zero_and_dynamic_range_vectors_have_exact_original_bytes(self):
        for n in (32,128):
            panels=[[-0.]*n,[(-0. if i%2 else 0.) for i in range(n)],
                [f32((-1. if i%3==0 else 1.)*2.**((i%61)-30)) for i in range(n)],
                [f32(1e-37*(1+i%7)) for i in range(n)],
                [f32(math.log(.0625+((i*19)%128)/9)) for i in range(n)]]
            for values in panels:self.assertEqual(bits(helper_tree_sum(values)),bits(REF.tree_sum(values)))
    def test_04_random_nonnegative_and_signed_columns_exact_original_bytes(self):
        rng=random.Random(20261025)
        for n,width in ((32,16),(128,32),(32,1)):
            for signed in (False,True):
                matrix=[[f32(rng.uniform(-1 if signed else 0,1)*2.**rng.randrange(-45,45))
                         for _ in range(width)] for _ in range(n)]
                expected=[REF.tree_sum([row[col] for row in matrix]) for col in range(width)]
                self.assertEqual(b''.join(map(bits,actual_helper_columns(matrix))),b''.join(map(bits,expected)))
    def test_05_logical_tree_survives_four_distinct_physical_packings(self):
        matrix=[[f32(math.sin(i*.31+j*.71)*2.**((i+j)%23-11)) for j in range(16)] for i in range(32)]
        expected=tuple(map(bits,actual_helper_columns(matrix)))
        for rows_per_warp in (4,8,16,32):
            restored=layout_roundtrip(matrix,rows_per_warp)
            self.assertEqual(restored,matrix)
            self.assertEqual(tuple(map(bits,actual_helper_columns(restored))),expected)
    def test_06_actual_old_tie_failure_is_preserved_and_explicit_tree_selects_original_candidate(self):
        x=REF.Tensor32((1,1,128,128),[math.sin(i*.731)*(.01+(i//128%29)/9) for i in range(16384)])
        w=REF.Tensor32((1,128),[2.**((i%9)-4) for i in range(128)])
        original=REF.quantize(x,w,'wscale')
        with reference_with_actual_new_helper():repaired=REF.quantize(x,w,'wscale')
        self.assertTrue(encoded_equal(original,repaired));self.assertEqual(repaired.candidate_indices[3],3)
        tree=REF.tree_sum
        def old_warp_graph(values):return tree([tree(values[i:i+8]) for i in range(0,len(values),8)])
        REF.tree_sum=old_warp_graph
        try:old=REF.quantize(x,w,'wscale')
        finally:REF.tree_sum=tree
        self.assertEqual(old.candidate_indices[3],4)
        self.assertEqual(sum(a!=b for a,b in zip(old.codes,repaired.codes)),24)
    def test_07_near_ties_cover_every_Key_group_and_Value_tile_without_relaxed_gate(self):
        values,omega=all_group_tile_near_tie_values();x=REF.Tensor32((1,1,128,128),values);w=REF.Tensor32((1,128),omega)
        original=REF.quantize(x,w,'wscale')
        with reference_with_actual_new_helper():repaired=REF.quantize(x,w,'wscale')
        self.assertTrue(encoded_equal(original,repaired))
        for group in range(4):
            for tile in range(8):self.assertEqual(repaired.candidate_indices[group*128+3+16*tile],3)
    def test_08_all_four_codec_methods_retain_exact_frozen_CPU_fields(self):
        values=[math.sin(i*.37)*(.1+(i//128%23)/7) for i in range(16384)]
        x=REF.Tensor32((1,1,128,128),values);w=REF.Tensor32((1,128),[2.**((i%9)-4) for i in range(128)])
        for method in ('base','wscale','rowbal','rowbal_wscale'):
            kwargs={'a':.5,'b':.25} if method.startswith('rowbal') else {}
            original=REF.quantize(x,w,method,**kwargs)
            with reference_with_actual_new_helper():repaired=REF.quantize(x,w,method,**kwargs)
            self.assertTrue(encoded_equal(original,repaired),method)
    def test_09_key32_and_value128_declared_shape_scope_is_strict(self):
        with self.assertRaises(AssertionError):CODEC._halves_sum_first(Tensor((64,1),[1.]*64,fp_add),64,1)
        with self.assertRaises(AssertionError):CODEC._halves_sum_first(Tensor((32,2),[1.]*64,fp_add),32,1)
    def test_10_original_reference_and_other_codec_math_source_bytes_are_unchanged(self):
        self.assertEqual((ROOT/'reference/math_reference.py').read_bytes(),(ORIGINAL/'reference/math_reference.py').read_bytes())
        old=ast.parse((ORIGINAL/'src/unified_codec.py').read_text());new=ast.parse((ROOT/'src/unified_codec.py').read_text())
        old_functions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
        new_functions={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for name in set(old_functions)-{'_row_kernel','_group_kernel','_kernels'}:
            self.assertEqual(ast.dump(old_functions[name]),ast.dump(new_functions[name]),name)
        for name,removed in (('_row_kernel',{'mean_sq','gm_m','gm_w','sum_log_m','sum_log_w'}),('_group_kernel',{'loss'})):
            class RemoveChangedAssignments(ast.NodeTransformer):
                def visit_Assign(self,node):
                    if any(isinstance(t,ast.Name) and t.id in removed for t in node.targets):return None
                    return self.generic_visit(node)
            before=RemoveChangedAssignments().visit(old_functions[name]);after=RemoveChangedAssignments().visit(new_functions[name])
            self.assertEqual(ast.dump(before),ast.dump(after),name)
    def test_11_helpers_have_no_generic_reduction_and_all_four_float_sites_use_them(self):
        tree=ast.parse((ROOT/'src/unified_codec.py').read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('_halves_add_first','_halves_sum_first'):
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sum' for n in ast.walk(functions[name])))
        row_calls=[n for n in ast.walk(functions['_row_kernel']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_halves_sum_first']
        group_calls=[n for n in ast.walk(functions['_group_kernel']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_halves_sum_first']
        self.assertEqual(len(row_calls),3);self.assertEqual(len(group_calls),1)
    def test_12_import_and_helper_proofs_have_no_device_or_model_dependencies(self):
        self.assertFalse(any(m=='torch' or m.startswith(('torch.','triton.','transformers.','numpy.')) for m in sys.modules))
    def test_13_field_fixture_independently_preserves_all32_sensitive_vectors(self):
        cases=load(ROOT/'reference/reduction_repair_cases.py','independent_check_of_field_panels')
        values=cases.all_group_tile_near_tie_values();weights=cases.all_group_tile_near_tie_weights()
        self.assertEqual(len(values),16384);self.assertEqual(len(weights),128)
        for group in range(4):
            for tile in range(8):
                for row in range(32):
                    original=f32(math.sin((row*128+3)*.731)*(.01+(row%29)/9))
                    self.assertEqual(bits(values[(group*32+row)*128+3+16*tile]),bits(f32(original*2.**(group+tile))))
            for row in range(32):self.assertEqual(bits(weights[group*32+row]),bits(f32(2.**((row%9)-4))))
        x=REF.Tensor32((1,1,128,128),cases.value128_dynamic_range_values())
        w=REF.Tensor32((1,128),cases.value128_dynamic_range_weights())
        factors=REF.row_factors(x,w,.5,.25)
        self.assertEqual(min(factors.values),.25);self.assertEqual(max(factors.values),4.)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReductionProofs)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS_CPU_EXPLICIT_HALVES_FIRST_OPERATION_GRAPH' if result.wasSuccessful() else 'FAIL_CPU_ENGINEERING_PROOF',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        codec_source_sha256=hashlib.sha256((ROOT/'src/unified_codec.py').read_bytes()).hexdigest(),
        frozen_reference_sha256=hashlib.sha256((ORIGINAL/'reference/math_reference.py').read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual production helper body with CPU logical tensor API; symbolic DAG and original scalar FP32 oracle. No compiler/device/qualification claim.',
        GPU_calls=0,model_loads=0,compile_calls=0)))
    raise SystemExit(not result.wasSuccessful())
