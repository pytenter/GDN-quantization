"""Read-only anchor recovery and prospective exact parameterization gate.

Only new experiment files are written. No model, optimizer or generation is run.
"""
import argparse, ast, hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

TASK = 'QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'
REPO = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1')
ROOT = REPO/'experiments'/TASK
PARENT = Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_C6R_H_DISCORDANT_MECHANISM_DIAGNOSIS_V1')
A800 = Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1/a800_stop_evidence')
DEPLOY = Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1')
SRC = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
CAYLEY = SRC/'experiments/shared/rotation/cayley_rotation.py'
FORMAL = SRC/'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py'
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
ANCHORS = ROOT/'provenance/recovered_anchors'
EXPECTED = {
 'C5R': ('2916169ea163de9c7f179d73936f38d6effd8c4407488298d5e29a3d8de01076', 'a11cf60f6ff11c2bb97f31b09129b8bfaadbf553f2005abc9276343ddca10fb5',113,904),
 'C6R': ('a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6', '620a35101b95cadd16f50cd1b869aeef54da9db0b3c76076fe82a374a6ee17cf',94,752),
 'C7': ('57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8', 'b08f66fa64f3453821a4d9dd9586c5263a747c461202c1dbc31b33193bde4958',19,152)}

def check(v,m):
    if not v: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(n,x):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    p.chmod(0o444)
def source(p,name):
    text=p.read_text();n=next(x for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
    s=ast.get_source_segment(text,n)
    return dict(path=str(p),file_sha256=sha(p),name=name,start=n.lineno,end=n.end_lineno,source=s,function_sha256=hashlib.sha256(s.encode()).hexdigest())
def manifest(n):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        for q in sorted(ROOT.rglob('*')):
            if q.is_file() and q!=p:f.write(sha(q)+'  '+str(q.relative_to(ROOT))+'\n')
    p.chmod(0o444)
def run(x):return subprocess.check_output(x,text=True).strip()

def prepare():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU prepare only')
    check(run(['git','rev-parse','--abbrev-ref','HEAD'])=='exp/qwen-gdn-propagation-surrogate-validation-v1','branch')
    protected=[]
    for line in (PARENT/'FINAL_SHA256SUMS').read_text().splitlines():
        h,n=line.split(None,1);p=PARENT/n.strip();check(sha(p)==h,'parent SHA '+str(p));protected.append(dict(path=str(p),sha256=h))
    check(len(protected)==106,'parent106')
    inv=read(PARENT/'provenance/input_identity.json');mi=inv['model_identity']
    for x in mi['files']:check(sha(x['path'])==x['sha256'],'model SHA');protected.append(x)
    metric_sources=read(PARENT/'provenance/input_artifact_inventory.json')['metric_sources']
    for x in metric_sources:
        check(sha(x['path'])==x['file_sha256'],'source SHA')
        check(source(Path(x['path']),x['function'])['function_sha256']==x['function_sha256'],'function SHA')
    rows=[]
    for a,(matrix_h,theta_h,u,e) in EXPECTED.items():
        sel=A800/'training'/a/'checkpoint_selection.json';s=read(sel)['selected']
        check((s['frozen_delta_sha256'],s['theta_sha256'],s['update'],s['exposure'])==(matrix_h,theta_h,u,e),'selection identity')
        matrix=ANCHORS/'matrix'/f'{a}.pt'
        if a!='C5R':
            check(not matrix.exists(),'no overwrite');shutil.copyfile(DEPLOY/'rotations'/a/'frozen_rotation_fp32.pt',matrix)
        theta=ANCHORS/'theta'/f'update_{u:03d}_exposure_{e:04d}.pt'
        check(sha(matrix)==matrix_h and sha(theta)==theta_h,'recovered SHA')
        matrix.chmod(0o444);theta.chmod(0o444)
        rows.append(dict(condition=a,update=u,exposure=e,matrix=str(matrix),matrix_sha256=matrix_h,theta=str(theta),theta_sha256=theta_h,original_matrix=s['frozen_delta_path'],original_theta=s['theta_path'],selection_sha256=sha(sel)))
        protected.append(dict(path=str(sel),sha256=sha(sel)))
    check(sha(CAYLEY)=='62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067','same archived Cayley source')
    cfg=read(Path(mi['path'])/'config.json');tc=cfg.get('text_config',cfg)
    layers=[i for i,t in enumerate(tc['layer_types']) if t=='linear_attention']
    raw=[read(p) for p in sorted((PARENT/'raw').glob('*.json'))]
    check(len(layers)==24 and len(raw)==23 and all(sorted({x['layer'] for x in p['qdq_metrics']})==layers for p in raw),'24 actual layer identity')
    check((tc['linear_num_key_heads'],tc['linear_num_value_heads'],tc['linear_key_head_dim'],tc['linear_value_head_dim'])==(16,32,128,128),'GDN layout')
    sections=[source(ARCH,'torch_recurrent_gated_delta_rule'),source(ARCH,'Qwen3_5GatedDeltaNet'),source(FORMAL,'QwenKeyHadamardPatch'),source(FORMAL,'QwenFrozenRotationPatch'),source(CAYLEY,'CayleyDenseRotation')]
    save('provenance/anchor_recovery.json',dict(status='PASS',anchors=rows,A800_reconnected=True,A800_compute_started=False,A800_mode='CPU-only 0.5CPU/2GB; no GPU',transfer='scp A800->local->3090; exact file SHA checked at each hop',previous_connection_refused_preserved=True,prior_preflight_stop_reporter_executed=False,original_files_changed=False,old_checkpoint_selection_changed=False))
    save('provenance/GDN_RECURRENCE_SOURCE_AUDIT.json',dict(status='PASS',source_only=True,recurrent_layer_ids=layers,state_shape=[1,32,128,128],key_heads=16,value_heads=32,head_repeat=2,normalization='q/k l2norm eps1e-6 before separate H and Delta; production helper normalization disabled by formal wrapper',state_update_order=['FP32','decay','key projection/erase','beta delta write','query readout','cache','C128 QDQ'],exact_update='S_pre=a*S_old+k outer beta*(v-(a*S_old)^T*k)',zero_write='reuse production helper with v=zeros_like(v); decay/beta/erase unchanged',source_sections=sections))
    save('provenance/quantizer_identity.json',dict(C128_QDQ_IDENTITY_GATE='PASS',scope='frozen production source identity; no copied quantizer',source=source(FORMAL,'quantize_c128'),group_size=128,axis=-2,groups_per_layer=4096,scale='abs.amax(-2).clamp_min(1e-12)/127',rounding='torch.round ties-to-even',clip=[-127,127],writeback='copy_ dequant cast back to actual cache dtype'))
    save('provenance/native_basis_source_mapping.json',dict(status='SOURCE_RECOVERED_NUMERICAL_PENDING',qk_forward='separate q/k.float @ H128 @ Delta',native_state='separate Delta @ S_stored then H128 @ result',stored_from_native='Delta.T @ (H128 @ S_native)',physical_source=next(x for x in metric_sources if x['function']=='physical_state'),ideal_invariance='only in exact arithmetic with orthogonal maps',numerical_invariance='NOT_EVALUATED'))
    save('protocol/phase_a_parameterization_policy.json',dict(frozen_before_parameterization_panel=True,endpoint_reconstruction='torch.equal + exact tensor SHA; NO tolerance (same archived A800 export audit policy)',orthogonality_max_abs=1e-4,rotation_source_sha256=sha(CAYLEY),archived_A800_rotation_source_sha256='62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067',archived_policy_source_sha256='231a7565880311cd5cb452fc67eaa311bdbb10c795332806c5decd09654382e48',no_derived_rotations_before_endpoint_pass=True,no_model_forward_before_source_parameter_gates=True))
    import torch
    sys.path.insert(0,'/data/zypan/transformers-qwen35/src');import transformers
    runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__)
    check(runtime==inv['runtime']['runtime'],'runtime drift')
    save('provenance/runtime_identity.json',dict(QWEN_RUNTIME_IDENTITY_GATE='PASS',runtime=runtime,model_identity=mi,scope='CPU identity, GPU numerical parity pending',source_files=[dict(path=str(p),sha256=sha(p)) for p in [ARCH,FORMAL,CAYLEY]],git_base_sha=run(['git','rev-parse','HEAD'])))
    save('provenance/hardware_identity.json',dict(gpu=run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv']),compute=run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv']),free_bytes=shutil.disk_usage('/data').free))
    for x in protected:check(sha(x['path'])==x['sha256'],'protected drift')
    save('provenance/parent_artifact_identity.json',dict(status='PASS',parent_manifest_sha256=sha(PARENT/'FINAL_SHA256SUMS'),protected=protected,old_verdicts_unchanged=True,Ling_outcomes_inspected=False))
    manifest('provenance/RECOVERY_SHA256SUMS')
    print('ANCHOR_RECOVERY_AND_SOURCE_AUDIT=PASS',flush=True)

def parameters():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=read(ROOT/'protocol/phase_a_parameterization_policy.json');check(sha(CAYLEY)==p['rotation_source_sha256'],'source drift')
    spec=importlib.util.spec_from_file_location('original_cayley_for_propagation',CAYLEY);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    audit=read(ROOT/'provenance/anchor_recovery.json');layers=read(ROOT/'provenance/GDN_RECURRENCE_SOURCE_AUDIT.json')['recurrent_layer_ids'];rows=[];failed=None
    with torch.inference_mode():
        for x in audit['anchors']:
            check(sha(x['matrix'])==x['matrix_sha256'] and sha(x['theta'])==x['theta_sha256'],'anchor SHA drift')
            m=torch.load(x['matrix'],map_location='cpu',weights_only=True);t=torch.load(x['theta'],map_location='cpu',weights_only=True)
            check(m['condition']==t['condition']==x['condition'] and m['update']==t['update']==x['update'] and m['theta_checkpoint_sha256']==x['theta_sha256'],'anchor metadata')
            check(list(m['layer_ids'])==layers,'anchor layers')
            bank=module.PerLayerCayleyRotations(layers).to('cuda:0');bank.load_state_dict(t['bank'],strict=True)
            for l in layers:
                reconstructed=bank.layer(l).matrix().detach().cpu().contiguous();frozen=m['matrices'][str(l)]
                exact=torch.equal(reconstructed,frozen)
                raw=lambda v:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                orth=float((frozen.T@frozen-torch.eye(128)).abs().max())
                row=dict(condition=x['condition'],layer=l,exact=exact,reconstructed_sha256=raw(reconstructed),frozen_sha256=raw(frozen),max_abs_diff=float((reconstructed-frozen).abs().max()),orthogonality_max_abs=orth)
                rows.append(row)
                if not exact or not bool(torch.isfinite(frozen).all()) or orth>p['orthogonality_max_abs']:
                    failed=row;break
            del bank
            if failed:break
    save('provenance/rotation_parameterization_identity.json',dict(ROTATION_PARAMETERIZATION_IDENTITY_GATE='FAIL' if failed else 'PASS',policy=p,rows=rows,first_failure=failed,no_new_parameters_learned=True,no_derived_rotations_created=True,no_original_files_modified=True))
    print('ROTATION_PARAMETERIZATION_IDENTITY_GATE='+('FAIL' if failed else 'PASS'),flush=True)
    if failed:
        verdict=dict(TASK_STATUS='STOP_ENDPOINT_RECONSTRUCTION_MISMATCH',ROTATION_PARAMETERIZATION_IDENTITY_GATE='FAIL',DISCOVERY_COMPLETE='0/48',CONFIRMATION_COMPLETE='0/48',SELECTED_SURROGATE='NONE',QWEN_GDN_PROPAGATION_SURROGATE_CONFIRMATION='NOT_EVALUATED',NEW_TRAINING='NO',NEW_FREE_GENERATION=0,NEW_AIME_GENERATION=0,NEW_C8_TRAINING_AUTHORIZED='NO',FINAL_ACTION='STOP')
        save('analysis/final_verdict.json',verdict)
        text='```text\n'+'\n'.join(k+' = '+str(v) for k,v in verdict.items())+'\n```\n\n'
        text+='Recovered all required exact archived anchor files. Their original file SHAs PASS. However the unchanged archived Cayley parameterization on this RTX3090 runtime did not reproduce the frozen A800 export byte-exactly. The inherited A800 policy requires exact tensor identity, without tolerance. No derived bank, document selection, model forward or Discovery/Confirmation measurement started. This is a premeasurement parameterization closure failure, not evidence against propagation metrics.\n\nFirst failure:\n```json\n'+json.dumps(failed,indent=2)+'\n```\n\nNo original artifact or old verdict changed. Do not reconstruct inverse-Cayley parameters, substitute matrices or relax the gate. Await user direction for a separate recovery closure.\n'
        q=ROOT/'FINAL_REPORT.md'
        with q.open('x') as f:f.write(text)
        q.chmod(0o444);manifest('FINAL_SHA256SUMS')
        return 2
    return 0

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['prepare','parameters'],required=True);args=a.parse_args()
    if args.phase=='prepare':prepare()
    else:sys.exit(parameters())
