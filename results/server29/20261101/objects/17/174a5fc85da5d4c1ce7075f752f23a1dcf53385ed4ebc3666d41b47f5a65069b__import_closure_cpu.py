"""One bounded pre-model import-closure audit in a fresh CPU-only interpreter.

This module never constructs Capture or calls model_step/gpu_guard, never loads
weights, and forbids CUDA initialization. Static dependencies are listed, not
executed. No installed-package tree is traversed or recursively hashed.
"""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

TASK='LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
W=Path(__file__).resolve().parents[1]
D=Path('/data/zypan/experiment_runs')/TASK
DIR=Path('/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path,scope):
    path=Path(path).resolve()
    assert path.is_file(), ('SOURCE_DEPENDENCY_MISSING',str(path))
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'scope':scope}
def publish(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(tmp,path);tmp.unlink()


def main():
    env={'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'LING_GATED_RMSNORM_FIXED_CONFIG':'BT16_W8','FLA_CACHE_MODE':'default','FLA_CONFIG_DIR':str(W/'frozen_fla_configs'),
         'PYTHONDONTWRITEBYTECODE':'1'}
    for key,value in env.items():assert os.environ.get(key)==value,(key,os.environ.get(key),value)
    assert sys.flags.dont_write_bytecode and not sys.flags.hash_randomization
    sys.path.insert(0,str(W))
    import torch
    from v2_runtime import initialize
    runtime=initialize(torch,0)
    assert not torch.cuda.is_initialized()
    blocked=[]
    def forbid_cuda(*args,**kwargs):
        blocked.append({'epoch':time.time(),'scope':'CUDA_INIT_FORBIDDEN_IN_CPU_IMPORT_AUDIT'})
        raise RuntimeError('CUDA_INIT_FORBIDDEN_IN_CPU_IMPORT_AUDIT')
    # Installed after the original initializer and before any scientific helper.
    torch.cuda.init=forbid_cuda
    torch.cuda._lazy_init=forbid_cuda
    result={'task_id':TASK,'scope':'fresh_interpreter_CPU_imports_only_no_Capture_no_model_no_forward',
            'epoch_start':time.time(),'environment':env,'initializer':runtime,'source':record(__file__,'audit_helper'),
            'CUDA_init_trap_calls':blocked,'GPU_model_constructions':0,'model_forward_count':0}
    try:
        spec=importlib.util.spec_from_file_location('ling_dual_axis_actual_forward_import_closure',DIR)
        direction=importlib.util.module_from_spec(spec);sys.modules[spec.name]=direction;spec.loader.exec_module(direction)
        transport,common,capture,core=direction.load_assets(6)
        assert hasattr(capture,'Capture') and hasattr(core,'qdq_r128') and hasattr(core,'update_pre_q')
        assert 'worker' in sys.modules and 'surrogate_statistics' in sys.modules
        assert not blocked and not torch.cuda.is_initialized()
        imports={}
        for name,module in list(sys.modules.items()):
            source=getattr(module,'__file__',None)
            if not source:continue
            path=Path(source)
            if path.suffix=='.pyc' and path.with_suffix('.py').exists():path=path.with_suffix('.py')
            text=str(path.resolve())
            if any(text.startswith(prefix) for prefix in ['/data/zypan/worktrees/','/data/zypan/experiment_runs/','/data/zypan/models/','/data/zypan/cleanroom/']):
                if path.is_file():imports[name]=record(path,'actual_import_origin')
        assert all(name in imports for name in ['common','capture','replay_core','transition','worker','surrogate_statistics'])
        static={}
        critical=[(common.V4/'scripts/v4_common.py','Capture.qualified_runtime extracts only apply_qualified_runtime AST'),
                  (common.PARENTS[2]/'scripts/phase2_final_hidden_512_smoke.py','Capture.__init__ P2 import; NOT executed here')]
        runtime_repo=common.V3/'inputs/runtime_repo'
        runtime_entry=runtime_repo/'experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py'
        critical.append((runtime_entry,'P2.load_runtime explicit entrypoint; listed only, no import'))
        for path,scope in critical:static[str(path.resolve())]=record(path,scope)
        # Follow only literal .py basenames reachable from this explicit legacy
        # forward-helper entrypoint, within two already-qualified script folders.
        # This is bounded source discovery, not a package or filesystem scan.
        index={}
        for directory in [runtime_repo/'experiments/rotation',runtime_repo/'experiments/ling']:
            for path in directory.glob('*.py'):
                assert path.name not in index, ('AMBIGUOUS_STATIC_BASENAME',path.name)
                index[path.name]=path
        pending=[runtime_entry];seen=set()
        while pending:
            path=pending.pop()
            if str(path) in seen:continue
            seen.add(str(path));assert len(seen)<=64,'BOUNDED_STATIC_FORWARD_SOURCE_LIMIT'
            static[str(path.resolve())]=record(path,'static_literal_forward_helper_closure_not_executed')
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):
                    base=Path(node.value).name
                    if base in index and str(index[base]) not in seen:pending.append(index[base])
        for path in Path('/data/zypan/models/Ling-3.0-tiny').glob('*.py'):
            static[str(path.resolve())]=record(path,'actual_model_source_listed_only_no_model_import')
        provenance=read_json(W/'frozen_runtime_provenance.json')
        fla={}
        for path,digest in provenance['installed_runtime_source_SHA256'].items():
            item=record(path,'explicit_14_frozen_FLA_sources')
            assert item['sha256']==digest,('FROZEN_FLA_SOURCE_CHANGED',path)
            fla[path]=item
        assert len(fla)==14
        from frozen_runtime_profile import validate_files
        configs=validate_files()
        versions={}
        for package in ['torch','transformers','triton','flash-linear-attention','fla-core','fla','numpy']:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]=None
        result.update(status='PASS_FRESH_CPU_FORWARD_IMPORT_CLOSURE',epoch_end=time.time(),python=sys.version,
            executable=sys.executable,torch=torch.__version__,compiled_CUDA_runtime=torch.version.cuda,
            package_versions=versions,actual_scientific_module_origins=imports,
            actual_scientific_module_count=len(imports),static_forward_dependencies=static,
            static_forward_dependency_count=len(static),frozen_FLA_source_records=fla,
            frozen_FLA_configurations=configs,source_sha256={**{r['path']:r['sha256'] for r in imports.values()},
                **{r['path']:r['sha256'] for r in static.values()},**{r['path']:r['sha256'] for r in fla.values()}},
            model_constructed=False,model_weights_loaded=False,CUDA_initialized=torch.cuda.is_initialized(),
            CUDA_init_trap_calls=blocked,worker_module_import_resolved=True,
            limitations='Static dependencies were not executed. No model constructor or tensor forward/GPU kernel was tested; runtime resource and numerical qualification remain NOT_RUN.')
        assert not torch.cuda.is_initialized() and not blocked
    except BaseException as exc:
        result.update(status='FAIL_CLOSED_CPU_IMPORT_CLOSURE',reason=repr(exc),traceback=traceback.format_exc(),epoch_end=time.time(),
                      CUDA_initialized=torch.cuda.is_initialized(),CUDA_init_trap_calls=blocked,
                      no_CUDA_fallback_attempt=True,no_model_construction_attempt=True)
        publish(D/'audits/CPU_IMPORT_CLOSURE.json',result)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));raise
    publish(D/'audits/CPU_IMPORT_CLOSURE.json',result)
    print(json.dumps({'status':result['status'],'actual_module_count':len(imports),'static_source_count':len(static),
          'FLA_source_count':len(fla),'CUDA_initialized':result['CUDA_initialized'],'output_sha256':sha(D/'audits/CPU_IMPORT_CLOSURE.json'),
          'source_sha256':sha(__file__),'versions':versions},allow_nan=False))


def read_json(path):return json.loads(Path(path).read_text())
if __name__=='__main__':main()
