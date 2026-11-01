"""Explicit-target SM86 compilation only: never launches or loads a GPU kernel.

Run after CPU-only staging by the parent. Production dependency admission is
retained; no source hotpatch or fallback is performed. Source DAG CPU proof
and actual-device field qualification remain separate required evidence.
"""
import argparse
import ast
import hashlib
import importlib.util
import importlib.abc
import json
import os
from pathlib import Path
import re
import sys
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def dump(p,x):write_new(p,(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def req(x,s):
    if not x:raise RuntimeError(s)


def ir_checks(asm,allowed_log_lines=(),expected_stages=None):
    ttgir=asm['ttgir'];llir=asm['llir'];ptx=asm['ptx']
    sums=[]
    for block in re.findall(r'"tt\.reduce".*?\}\)',ttgir,re.S):
        if 'arith.addf' in block:sums.append(block[:1200])
    req(not sums,'Compiler reintroduced generic floating sum reduction')
    bad=[line for line in llir.splitlines() if re.search(r'\bf(?:add|sub|mul|div)\s+(?:fast|reassoc|contract|afn|arcp|nnan|ninf|nsz)\b',line)]
    req(not bad,'FP operation has unapproved LLVM fast/reassociation flags')
    # Pinned libdevice log polynomials contain explicit round-to-nearest
    # FMAs. They belong to the preregistered transcendental implementation,
    # not a reassociation/fusion of the fixed sum or pointwise QDQ graph.
    # Allow only those LLVM calls whose debug scope is this row kernel and
    # exact source line contains the declared tl.log call; helpers/group and
    # missing/unrecognized debug provenance remain rejected.
    row_scopes=set(re.findall(r'!(\d+) = (?:distinct )?!DISubprogram\(name: "_row_kernel"',llir))
    locations={k:(int(line),scope) for k,line,scope in re.findall(r'!(\d+) = !DILocation\(line: (\d+),.*?scope: !(\d+)',llir)}
    approved_fma=[]
    for entry in llir.splitlines():
        if 'call' not in entry or not re.search(r'@llvm\.[^\s]*(?:fmuladd|fma)',entry):continue
        debug=re.search(r'!dbg !(\d+)',entry);where=locations.get(debug.group(1)) if debug else None
        req('@llvm.nvvm.fma.rn.' in entry and where is not None and
            where[0] in set(allowed_log_lines) and where[1] in row_scopes,
            'FMA outside declared row-log implementation or with unknown provenance')
        approved_fma.append(dict(source_line=where[0],scope=where[1]))
    ptx_fmas=len(re.findall(r'\bfma(?:\.[a-z]+)*\.f32\b',ptx))
    req(not ptx_fmas or approved_fma,'PTX FMA without approved row-log LLVM provenance')
    stages={}
    for height,width in re.findall(r'arith\.addf[^\n]*tensor<(\d+)x(\d+)xf32',ttgir):
        key=height+'x'+width;stages[key]=stages.get(key,0)+1
    if expected_stages is not None:req(stages==expected_stages,'Compiled explicit-add stage shapes/counts differ: '+repr(stages))
    req('.target sm_86' in ptx,'Explicit SM86 target missing')
    req(re.search(r'\bfadd\s+(?:float|<)',llir) is not None,'Explicit FP32 additions missing')
    return dict(status='PASS_COMPILED_IR_STRUCTURE_NOT_NUMERICAL_GPU_PROOF',
        generic_float_sum_regions=0,unapproved_fast_flags=0,fused_QDQ_or_sum_operations=0,
        permitted_libdevice_log_FMA_calls=len(approved_fma),permitted_libdevice_log_FMA_source_lines=sorted({x['source_line'] for x in approved_fma}),PTX_FMA_instructions=ptx_fmas,explicit_add_stage_shapes=stages,
        target='sm_86',LLVM_fadd_count=len(re.findall(r'\bfadd\b',llir)),
        PTX_add_f32_count=len(re.findall(r'\badd(?:\.[a-z]+)*\.f32\b',ptx)),
        caveat='No-fast flags plus explicit source DAG preserve required operation dependencies; actual GPU oracle must still pass')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();out=a.out.resolve()
    req(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU compile requires explicitly empty CUDA_VISIBLE_DEVICES')
    req(not out.exists(),'New CPU compile evidence directory required; preserve prior attempts')
    out.mkdir(parents=True);os.environ['TRITON_CACHE_DIR']=str(root/'engineering/triton_cache')
    sys.path.insert(0,str(root/'src'))
    pins={n:sha(root/n) for n in ['src/unified_codec.py','src/triton_runtime_guard.py','analysis/TRITON_RUNTIME_FRESH_SHA.json','reference/math_reference.py','reference/reduction_repair_cases.py']}
    touched=[];torch_attempts=[]
    class NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch' or fullname.startswith('torch.'):
                torch_attempts.append(fullname);raise RuntimeError('CPU target compile forbids torch import')
            return None
    req('torch' not in sys.modules,'CPU target compile must start without torch')
    sys.meta_path.insert(0,NoTorch())
    report=dict(status='CPU_COMPILE_STARTED',GPU_calls=0,kernel_launches=0,model_loads=0,driver_context_calls=0,source_sha256=pins)
    try:
        from triton_runtime_guard import verify_before_import,verify_after_import,dependency_receipt
        verify_before_import(root)
        import triton
        verify_after_import(root,triton)
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        from triton.runtime import driver as driver_config
        import triton.compiler.compiler as compiler
        class NoDriver:
            def __getattr__(self,name):
                touched.append(name);raise RuntimeError('CPU compile attempted forbidden active GPU driver access: '+name)
        req(hasattr(driver_config,'set_active'),'Unexpected pinned Triton driver-config API')
        driver_config.set_active(NoDriver())
        def no_handles(*args,**kwargs):
            touched.append('CompiledKernel._init_handles');raise RuntimeError('CPU compile must never initialize GPU kernel handles')
        compiler.CompiledKernel._init_handles=no_handles
        spec=importlib.util.spec_from_file_location('actual_repaired_codec_cpu_compile',root/'src/unified_codec.py');codec=importlib.util.module_from_spec(spec);sys.modules[spec.name]=codec;spec.loader.exec_module(codec)
        row_kernel,group_kernel=codec._kernels()
        target=GPUTarget('cuda',86,32);options=dict(num_warps=4,enable_fp_fusion=False)
        selected=json.loads((root/'CALIBRATION_PARAMETER_SELECTION.json').read_text())['selected'];aa,bb=float(selected['a']),float(selected['b'])
        contexts=[(1,True),(2,True),(2,False),(32,True)];variants=[]
        for heads,batched in contexts:
            variants.append((f'row_h{heads}_batched{int(batched)}',row_kernel,
                {0:'*fp32',1:'*fp32',2:'*fp32',3:'*i32'},
                {4:heads,5:batched,6:aa,7:bb}))
            for method,use_row,search in [('base',False,False),('W',False,True),('R',True,False),('RW',True,True)]:
                variants.append((f'group_{method}_h{heads}_batched{int(batched)}',group_kernel,
                    {0:'*fp32',1:'*fp32',2:'*fp32',3:'*i8',4:'*fp32',5:'*i1',6:'*u8',7:'*fp32',8:'*fp32',9:'*i32'},
                    {10:heads,11:batched,12:use_row,13:search}))
        completed=[]
        for name,fn,signature,constants in variants:
            source=ASTSource(fn=fn,signature=signature,constants=constants)
            compiled=triton.compile(source,target=target,options=options)
            # Reading asm never accesses .run/.n_regs/kernel handles.
            assembly={k:compiled.asm[k] for k in ('ttir','ttgir','llir','ptx')}
            files={}
            for kind,content in assembly.items():
                p=out/name/(fn.__name__+'.'+kind);write_new(p,content.encode());files[kind]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
            if name.startswith('row_'):
                expected={str(n)+'x32':1 for n in (64,32,16,8,4,2,1)}
                expected.update({str(n)+'x1':2 for n in (16,8,4,2,1)})
                source_ast=ast.parse((root/'src/unified_codec.py').read_text())
                row_fn=next(n for n in source_ast.body if isinstance(n,ast.FunctionDef) and n.name=='_row_kernel')
                log_lines={n.lineno for n in ast.walk(row_fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.log'}
            else:
                expected={str(n)+'x16':5 if constants[13] else 1 for n in (16,8,4,2,1)};log_lines=set()
            checks=ir_checks(assembly,log_lines,expected)
            item=dict(name=name,status='PASS_CPU_EXPLICIT_TARGET_COMPILE_ONLY',signature=signature,constants=constants,checks=checks,files=files)
            dump(out/name/'RECEIPT.json',item);completed.append(item)
            print(json.dumps(dict(variant=name,status=item['status'],GPU_calls=0)),flush=True)
        req(not touched,'Active GPU driver was touched')
        req(not torch_attempts and 'torch' not in sys.modules,'Torch was imported in CPU target compilation')
        req(all(sha(root/n)==h for n,h in pins.items()),'Source changed during CPU compilation')
        report.update(status='PASS_EXPLICIT_SM86_CPU_COMPILATION_AND_IR_STRUCTURE_NO_GPU',variants=completed,variant_count=len(completed),target=dict(backend='cuda',arch=86,warp_size=32),options=options,driver_access_attempts=touched,torch_import_attempts=torch_attempts,torch_imports=0,triton_dependency_admission=dependency_receipt(),actual_GPU_qualification_credit=0,AIME_samples=0,not_a_GPU_numeric_or_performance_proof=True)
        dump(out/'CPU_TARGET_COMPILE_REPORT.json',report)
    except BaseException as exc:
        report.update(status='CPU_TARGET_COMPILE_FAILURE_NO_GPU_RETRY',error=repr(exc),traceback=traceback.format_exc(),driver_access_attempts=touched,torch_import_attempts=torch_attempts)
        dump(out/'CPU_TARGET_COMPILE_FAILURE.json',report)
        raise


if __name__=='__main__':main()
