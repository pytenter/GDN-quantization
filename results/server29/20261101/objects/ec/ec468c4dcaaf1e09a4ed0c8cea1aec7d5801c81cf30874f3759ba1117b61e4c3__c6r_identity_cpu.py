"""Read-only CPU provenance audit; JSON to stdout, no remote writes or CUDA calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast, hashlib, importlib.util, json, math, sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch

torch.set_num_threads(1)
BASE = Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1')
EXPORT = BASE/'rotations/C6R/frozen_rotation_fp32.pt'
MANIFEST = BASE/'configs/c6r_rotation_manifest.json'
RECOVERED = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1/provenance/recovered_anchors/matrix/C6R.pt')
CANONICAL = Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_GDN_C6R_C7_CANONICAL60_V1')
FORMAL = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')
OLD_FORMAL = Path('/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/formal_c128_runner.py')
SELECTION = Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/a800_stop_evidence/training/C6R/checkpoint_selection.json')
CORE = Path('/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/src/core.py')
EXPECTED = 'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6'
EXPECTED_MANIFEST = 'f365dbab44e31b127abc5f9cc37916e432c4f3c88cc7706c82886812686d1103'
LAYERS = tuple(i for i in range(32) if i % 4 != 3)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def asset(p):
    before=p.stat(); digest=sha(p); after=p.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(p),bytes=before.st_size,sha256=digest,stat_unchanged_during_hash=True)

def tsha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    assets={k:asset(p) for k,p in dict(export=EXPORT,deployment_manifest=MANIFEST,recovered_export=RECOVERED,formal_source=FORMAL,inherited_formal_source=OLD_FORMAL,checkpoint_selection=SELECTION,current_qualified_core=CORE,canonical60_runtime=CANONICAL/'provenance/canonical60_runtime_identity.json',canonical60_worker=CANONICAL/'scripts/canonical60_worker.py',canonical60_common=CANONICAL/'scripts/canonical60_common.py').items()}
    assert assets['export']['sha256']==assets['recovered_export']['sha256']==EXPECTED
    assert assets['deployment_manifest']['sha256']==EXPECTED_MANIFEST
    assert assets['formal_source']['sha256']==assets['inherited_formal_source']['sha256']=='77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
    m=json.loads(MANIFEST.read_text()); sel=json.loads(SELECTION.read_text()); rt=json.loads((CANONICAL/'provenance/canonical60_runtime_identity.json').read_text())
    assert m['condition']=='C6R' and m['file_sha256']==EXPECTED and tuple(m['layer_ids'])==LAYERS
    assert m['selected_update']==sel['selected']['update']==94 and sel['selected']['exposure']==752
    assert sel['selected']['frozen_delta_sha256']==EXPECTED and assets['checkpoint_selection']['sha256']=='b310c0cfe8c4908ef95cb8bb9e04964086e65359873d47386ee515b253bdb011'
    assert rt['rotation_files']['C6R']==dict(manifest_path=str(MANIFEST),manifest_sha256=EXPECTED_MANIFEST,path=str(EXPORT),sha256=EXPECTED)
    assert m['loading_rule']=='load frozen rotation directly; do not reconstruct R from theta on inference host'
    p=torch.load(EXPORT,map_location='cpu',weights_only=True)
    assert p['condition']=='C6R' and tuple(p['layer_ids'])==LAYERS and p['matrix_dtype']=='torch.float32' and p['matrix_shape']==[128,128]
    assert set(p['matrices'])=={str(l) for l in LAYERS}
    matrices=[]
    for l in LAYERS:
        x=p['matrices'][str(l)]
        assert x.device.type=='cpu' and x.dtype==torch.float32 and tuple(x.shape)==(128,128) and bool(torch.isfinite(x).all())
        digest=tsha(x); assert digest==m['per_layer_sha256'][str(l)]==sel['selected']['per_layer_sha256'][str(l)]
        matrices.append(dict(layer=l,sha256=digest,shape=list(x.shape),dtype=str(x.dtype),orthogonality_max_abs_FP64=float((x.double().T@x.double()-torch.eye(128,dtype=torch.float64)).abs().max())))
    tree=ast.parse(FORMAL.read_text()); names={'get_state','quantize_c128','hadamard','normalize_qk','QwenKeyHadamardPatch','FrozenRotationLayer','FrozenRotationBank','load_frozen_rotation','QwenFrozenRotationPatch','file_sha256','tensor_sha256'}
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    assert set(nodes)==names
    function_AST={n:hashlib.sha256(ast.dump(v,include_attributes=False).encode()).hexdigest() for n,v in nodes.items()}
    qsrc=ast.get_source_segment(FORMAL.read_text(),nodes['quantize_c128'])
    assert assets['current_qualified_core']['sha256']=='aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575'
    spec=importlib.util.spec_from_file_location('qualified_core_cpu_only',CORE);K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    f=K.formal(False);ns=vars(f).copy();ns.update(hashlib=hashlib,json=json)
    extra={'file_sha256','tensor_sha256','FrozenRotationLayer','FrozenRotationBank','load_frozen_rotation','QwenFrozenRotationPatch'}
    original=K.extract(OLD_FORMAL,extra,ns)
    bank,identity=original.load_frozen_rotation(EXPORT,MANIFEST,'cpu','C6R')
    assert identity['rotation_sha256']==EXPECTED and identity['deployment_manifest_sha256']==EXPECTED_MANIFEST
    assert identity['direct_frozen_load'] is True and identity['theta_reconstruction'] is False
    assert all(bank.layer(l).matrix().device.type=='cpu' and tsha(bank.layer(l).matrix())==m['per_layer_sha256'][str(l)] for l in LAYERS)
    cpu_loader=dict(status='PASS',loader='Exact original load_frozen_rotation through qualified core.extract, same extra namespace as entry; device=cpu',bank_layer_count=len(bank.layer_ids),bank_parameters=len(list(bank.parameters())),all_buffers_CPU_FP32=True,direct_frozen_load=True,theta_reconstruction=False,identity=identity,additional_namespace=['hashlib','json'],CUDA_api_calls=0)
    return dict(status='PASS',qualification_scope='CPU_IDENTITY_ONLY',GPU_calls=0,model_forwards=0,remote_writes=0,assets=assets,rotation_SHA256=EXPECTED,deployment_manifest_SHA256=EXPECTED_MANIFEST,selected_update=94,selected_exposure=752,theta_reconstruction=False,payload_metadata={k:v for k,v in p.items() if k!='matrices'},matrix_checks=matrices,original_loader_CPU_test=cpu_loader,formal_function_AST_SHA256=function_AST,quantize_c128_source_SHA256=hashlib.sha256(qsrc.encode()).hexdigest(),canonical60_same_export_manifest=True,C6R_minimal_original_rotation_path='VIABLE_WITH_GPU_GATES_PENDING',GPU_coordinate_runtime_gate='NOT_RUN',source_scope='CPU read-only identity and AST audit; not a new model/FP-H equivalence qualification')

if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2,allow_nan=False))
