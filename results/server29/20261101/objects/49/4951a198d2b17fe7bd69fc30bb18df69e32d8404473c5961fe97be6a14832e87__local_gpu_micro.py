"""Single authorized local RTX3090 diagnostic; never full-model/formal training.

Importing this file or calling it without the root coordinator's authorization
does not import Torch, initialize CUDA, or create scientific artifacts.
"""
import argparse
from pathlib import Path
import gc, hashlib, json, math, os, random, subprocess, sys, time, traceback

TASK = 'QWEN_GDN_DUALM_AWARE_ROTATION_TRAINING_READINESS_V1'
ROOT = Path(__file__).resolve().parents[1]
UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
CHECKPOINT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/checkpoints/CURRENT.pt')
CP_SHA = '99ecd34378ca21ccfdf4d4a352804ae1a45d43e63d4cce9d74084149cddd282c'
CP_BYTES = 1507209134
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
INDEX_SHA = '26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'
SHARD = MODEL/'model.safetensors-00004-of-00004.safetensors'
SHARD_SHA = 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6'
NORM_KEY = 'model.language_model.layers.0.linear_attn.norm.weight'
PROJ_KEY = 'model.language_model.layers.0.linear_attn.out_proj.weight'
SEED = 20261017

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT/name).read_text())

def save(name,obj):
    with (ROOT/name).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def require(condition,text):
    if not bool(condition):
        raise RuntimeError(text)

def nvml():
    output=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    values=[x.strip() for x in output.split(',')]
    require(len(values)==4 and values[0]==UUID and '3090' in values[1], 'Physical GPU identity changed')
    return dict(UUID=values[0],name=values[1],memory_total_MiB=int(values[2]),memory_used_MiB=int(values[3]),unix=time.time())

def authorization(gpu):
    path=ROOT/'LOCAL_GPU_MICRO_AUTHORIZATION.json'
    if not path.exists():
        return None
    auth=json.loads(path.read_text())
    require(auth['status']=='AUTHORIZED_SINGLE_LOCAL_MICRO_ATTEMPT' and auth['task_id']==TASK, 'No explicit local micro authorization')
    require(gpu==auth['gpu']==0 and auth['GPU_UUID']==UUID, 'Only exact GPU0/UUID is allowed')
    require(0<auth['max_gpu_seconds']<=3600 and 0<=time.time()-auth['dispatch_unix']<auth['max_gpu_seconds'], 'No valid bounded GPU budget')
    require(auth['offline_gates']==dict(TRAINING_PROVENANCE_GATE='PASS',DUAL_M_FORWARD_PARITY='PASS',CPU_GRADIENT_GATE='PASS',DATA_ISOLATION_GATE='PASS'), 'Offline gates are not all admitted')
    mandatory={'GPU_SOURCE_MANIFEST.json','GRADIENT_QUALIFICATION_CPU.json','DUAL_M_FORWARD_PARITY.json','CPU_SOURCE_MANIFEST.json','FROZEN_C6R_PROVENANCE.json','PAIRED_TRAINING_PREREGISTRATION.json','DATA_SPLIT_MANIFEST.json','DATA_LEAKAGE_AUDIT.json'}
    require(mandatory<=set(auth['frozen_input_sha256']), 'Missing admission input bindings')
    for relative,digest in auth['frozen_input_sha256'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Admission relative path escaped root')
        require(sha(ROOT/relative)==digest, 'Admission input SHA changed: '+relative)
    manifest=read('GPU_SOURCE_MANIFEST.json')
    require({'source/local_gpu_micro.py','source/codec_training_adapter.py','source/training_state_io.py'}<=set(manifest['execution_sources']), 'GPU source manifest is incomplete')
    for relative,digest in manifest['execution_sources'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Source manifest path escaped root')
        require(sha(ROOT/relative)==digest, 'GPU micro source SHA changed: '+relative)
    require(read('FROZEN_C6R_PROVENANCE.json')['TRAINING_PROVENANCE_GATE']=='PASS', 'Training provenance not qualified')
    require(read('DUAL_M_FORWARD_PARITY.json')['status']=='PASS', 'Frozen codec CPU parity not qualified')
    require(read('GRADIENT_QUALIFICATION_CPU.json')['status']=='PASS_LOCAL_CPU_ONLY', 'CPU gradients not qualified')
    require(read('PAIRED_TRAINING_PREREGISTRATION.json')['frozen'] is True, 'Paired protocol is not frozen')
    return auth

def run(auth):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'Only physical GPU0 may be visible')
    for name in ('FIRST_LOCAL_GPU_FAILURE.json','LOCAL_GPU_MICRO_STARTED.json','LOCAL_3090_TRAINING_CANARY.json'):
        require(not (ROOT/name).exists(), 'Prior local GPU attempt; no retry')
    save('LOCAL_GPU_MICRO_STARTED.json',dict(status='STARTED_ONE_ATTEMPT',pid=os.getpid(),authorization_SHA256=sha(ROOT/'LOCAL_GPU_MICRO_AUTHORIZATION.json'),unix=time.time()))
    import torch
    import codec_training_adapter as A
    from training_state_io import save_legal_boundary_checkpoint,restore_legal_boundary_checkpoint
    from safetensors import safe_open

    def guard():
        require(time.time()-auth['dispatch_unix']<min(auth['max_gpu_seconds'],3300), 'Local GPU micro deadline reached; no next phase')
    def seed():
        random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        if 'numpy' in sys.modules:
            sys.modules['numpy'].random.seed(SEED)
    def tree_hash(x):
        def view(value):
            if torch.is_tensor(value):
                return ['tensor',str(value.dtype),list(value.shape),A.tensor_hash(value)]
            if isinstance(value,dict):
                return [[str(k),view(v)] for k,v in sorted(value.items(),key=lambda p:str(p[0]))]
            if isinstance(value,(tuple,list)):
                return [view(v) for v in value]
            return value
        return hashlib.sha256(json.dumps(view(x),sort_keys=True,allow_nan=False).encode()).hexdigest()

    guard()
    require(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Torch/CUDA runtime changed')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    components=A.QualifiedComponents()
    require(CHECKPOINT.stat().st_size==CP_BYTES and sha(CHECKPOINT)==CP_SHA,'Actual fixture checkpoint SHA/bytes changed')
    sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    require(payload['cursor']==32768 and payload['condition']=='C6R_DUAL_M','Actual fixture condition/cursor changed')
    captured=components.base.get_state(payload['cache'],0).detach().clone()
    fixture_identity=dict(path=str(CHECKPOINT),sha256=CP_SHA,bytes=CP_BYTES,condition=payload['condition'],doc_id=payload['doc_id'],cursor=payload['cursor'],layer=0,state_sha256=A.tensor_hash(captured),
        coordinate_scope='Stored historical C6R+DualM coordinates; diagnostic numerical fixture, NOT a clean native FP teacher',data_scope='Qualification only; excluded from new TRAIN/DEV/FUTURE test')
    del payload;gc.collect()
    require(tuple(captured.shape)==(1,32,128,128) and captured.dtype==torch.float32,'Full real GDN shape/dtype changed')
    require(sha(MODEL/'model.safetensors.index.json')==INDEX_SHA and sha(SHARD)==SHARD_SHA,'Frozen weight index/shard changed')
    weight_map=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    require(weight_map[NORM_KEY]==SHARD.name and weight_map[PROJ_KEY]==SHARD.name,'Frozen tensor-to-shard binding changed')
    with safe_open(str(SHARD),framework='pt',device='cpu') as handle:
        norm_weight=handle.get_tensor(NORM_KEY)
        proj_weight=handle.get_tensor(PROJ_KEY)
    require(norm_weight.dtype==proj_weight.dtype==torch.bfloat16 and tuple(norm_weight.shape)==(128,) and tuple(proj_weight.shape)==(4096,4096),'True model norm/out_proj shape/dtype changed')
    weight_identity=dict(shard_path=str(SHARD),shard_sha256=SHARD_SHA,index_sha256=INDEX_SHA,
        norm=dict(name=NORM_KEY,shape=list(norm_weight.shape),dtype=str(norm_weight.dtype),sha256=A.tensor_hash(norm_weight)),
        out_proj=dict(name=PROJ_KEY,shape=list(proj_weight.shape),dtype=str(proj_weight.dtype),sha256=A.tensor_hash(proj_weight)))
    guard();before=nvml()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    require(not any(line.split(',')[0].strip()==UUID for line in processes.splitlines() if line.strip()), 'GPU0 has another compute process; no takeover')
    torch.cuda.set_device(0)
    require(torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_properties(0).name,'Logical GPU map changed')
    state=captured.to('cuda:0');del captured
    norm=components.arch.Qwen3_5RMSNormGated(128).to(device='cuda:0',dtype=torch.bfloat16)
    projection=torch.nn.Linear(4096,4096,bias=False,device='cuda:0',dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(norm_weight.to('cuda:0'))
        projection.weight.copy_(proj_weight.to('cuda:0'))
    del norm_weight,proj_weight
    for p in list(norm.parameters())+list(projection.parameters()):p.requires_grad_(False)
    norms={0:norm};projections={0:projection};states={0:state}
    frozen_weights_sha=tree_hash(dict(norm=norm.state_dict(),projection=projection.state_dict()))
    (ROOT/'qualification').mkdir(exist_ok=True)
    results=[]
    for arm in ('A','D'):
        guard();seed()
        bank=components.cayley.PerLayerCayleyRotations((0,)).to('cuda:0')
        optimizer=torch.optim.Adam(bank.parameters(),lr=.001,weight_decay=0)
        require(sum(p.numel() for p in bank.parameters())==8128,'Only one original theta vector may train')
        context=dict(preregistration_sha256=sha(ROOT/'PAIRED_TRAINING_PREREGISTRATION.json'),data_split_sha256=sha(ROOT/'DATA_SPLIT_MANIFEST.json'),
            teacher_manifest_sha256=sha(ROOT/'GPU_SOURCE_MANIFEST.json'),source_manifest_sha256=sha(ROOT/'GPU_SOURCE_MANIFEST.json'),
            arm=arm,codec={'A':'CANONICAL_C128','D':'DUAL_M_INT8'}[arm],completed_updates=2,next_document_order_index=2,
            completed_captures=2,captures_per_update=1,tokens_per_update=64,checkpoint_kind='MICRO_QUALIFICATION_ONLY',
            boundary='AFTER_DOCUMENT_OPTIMIZER_STEP_CACHE_DISCARDED',layer_roster=[0],horizon=64,
            identity_scope='LOCAL_ORIGINAL_OPERATOR_ON_DIAGNOSTIC_INPUTS_NOT_A_FULL_MODEL_DOCUMENT')
        torch.cuda.reset_peak_memory_stats();rows=[]
        def update(index):
            guard();optimizer.zero_grad(set_to_none=True);probe=random.random()
            events=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
            torch.cuda.synchronize();started=time.monotonic();events[0].record()
            loss,info=A.local_diagnostic_loss(components,bank,states,norms,projections,arm,tokens=64)
            events[1].record();events[1].synchronize();forward=time.monotonic()-started
            require(bool(torch.isfinite(loss).item()),'Nonfinite local loss')
            started=time.monotonic();events[2].record();loss.backward();events[3].record();events[3].synchronize();backward=time.monotonic()-started
            params=list(bank.parameters())
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in params),'Nonfinite/missing theta gradient')
            grad_norm=torch.nn.utils.clip_grad_norm_(params,1.)
            require(math.isfinite(float(grad_norm)) and float(grad_norm)>0,'Zero/nonfinite theta gradient')
            started=time.monotonic();events[4].record();optimizer.step();events[5].record();events[5].synchronize();opt_seconds=time.monotonic()-started
            orth=bank.runtime_gate(1e-4);require(orth['status']=='PASS','Frozen orthogonality threshold failed')
            require(info['QDQ_calls']==64 and info['next_QDQ_consumption_checks']==63,'Persistent hard-QDQ consumption failed')
            require(all(not p.requires_grad and p.grad is None for p in list(norm.parameters())+list(projection.parameters())),'Frozen model weight acquired gradient')
            row=dict(update=index,loss=float(loss),state_loss=float(info['state_loss']),functional_loss=float(info['functional_loss']),
                gradient_norm=float(grad_norm),orthogonality_error=orth['max_abs_rt_r_minus_i'],python_rng_probe=probe,
                bank_sha256=tree_hash(bank.state_dict()),optimizer_sha256=tree_hash(optimizer.state_dict()),
                diagnostic_input_sha256=info['diagnostic_input_sha256'],teacher_target_sha256=info['teacher_target_sha256'],
                state_sha256=info['state_hashes'],tokens_per_optimizer_microstep=64,captures_per_update=1,QDQ_writebacks=64,next_QDQ_consumption_checks=63,
                forward_wall_seconds=forward,backward_wall_seconds=backward,optimizer_wall_seconds=opt_seconds,
                forward_CUDA_event_ms=events[0].elapsed_time(events[1]),backward_CUDA_event_ms=events[2].elapsed_time(events[3]),optimizer_CUDA_event_ms=events[4].elapsed_time(events[5]),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),NVML=nvml(),
                scope='Local single-layer H64 diagnostic; forward timing includes observation/CPU transfers; no full9B training or scientific gain claim')
            print(json.dumps(dict(arm=arm,update=index,status='LOCAL_MICRO_STEP_COMPLETE',loss=row['loss'])),flush=True)
            return row
        rows.append(update(1));rows.append(update(2));guard()
        checkpoint=save_legal_boundary_checkpoint(ROOT/f'qualification/LOCAL_GPU_MICRO_{arm}_AFTER2.pt',bank,optimizer,torch,context)
        rows.append(update(3));guard()
        restore=restore_legal_boundary_checkpoint(checkpoint['path'],bank,optimizer,torch,context,checkpoint['sha256'])
        replay=update(3)
        for key in ('loss','state_loss','functional_loss','gradient_norm','orthogonality_error','python_rng_probe','bank_sha256','optimizer_sha256','diagnostic_input_sha256','teacher_target_sha256','state_sha256'):
            require(rows[2][key]==replay[key],'Same-GPU checkpoint/RNG replay failed: '+key)
        require(frozen_weights_sha==tree_hash(dict(norm=norm.state_dict(),projection=projection.state_dict())),'Frozen real model weight mutated')
        results.append(dict(arm=arm,status='PASS_LOCAL_ONLY',updates=rows,replayed_update3=replay,checkpoint=checkpoint,restore=restore))
        del bank,optimizer;gc.collect();torch.cuda.empty_cache()
    require(all(a['diagnostic_input_sha256']==d['diagnostic_input_sha256'] and a['teacher_target_sha256']==d['teacher_target_sha256'] for a,d in zip(results[0]['updates'],results[1]['updates'])),'A/D inputs or diagnostic teacher differ')
    save('LOCAL_3090_TRAINING_CANARY.json',dict(TASK_ID=TASK,status='PASS_LOCAL_SINGLE_LAYER_ONLY',GPU_UUID=UUID,physical_gpu=0,seed=SEED,
        actual_state=fixture_identity,true_frozen_weights=weight_identity,parameter_dtypes=dict(theta='torch.float32',state='torch.float32',q_k_v_gate='torch.bfloat16',norm_out_proj='torch.bfloat16'),
        operator_backend='Original frozen Torch operator AST bodies; optional FLA/hub dispatch not used in this local operator test',
        LOCAL_GRADIENT_PASS=True,FULL_MODEL_TRAINING='NOT_TESTED',A800_TRAINING='NOT_TESTED',research_rotation_created=False,
        initial_NVML=before,final_NVML=nvml(),arms=results,source_manifest_sha256=sha(ROOT/'GPU_SOURCE_MANIFEST.json'),
        limitations=['Stored C6R coordinates are not a clean native FP teacher','Diagnostic q/k/v/g/beta/gate are not a fullmodel training trajectory','One layer64tokens/onecapture per microstep, not original24layer1024token/8capture document training','Loss changes are engineering observations, not32K or AIME benefit']))
    del state,norm,projection,components,states,norms,projections
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    return dict(status='PASS_LOCAL_SINGLE_LAYER_ONLY',FULL_MODEL_TRAINING='NOT_TESTED')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0],required=True)
    args=parser.parse_args()
    auth=authorization(args.gpu)
    if auth is None:
        print(json.dumps(dict(status='REFUSED_NO_EXPLICIT_LOCAL_GPU_AUTHORIZATION',Torch_imported=False,CUDA_initialized=False)))
        return 2
    try:
        result=run(auth);print(json.dumps(result));return 0
    except BaseException as exc:
        if not (ROOT/'FIRST_LOCAL_GPU_FAILURE.json').exists():
            save('FIRST_LOCAL_GPU_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),no_retry=True))
        raise

if __name__=='__main__':
    raise SystemExit(main())
