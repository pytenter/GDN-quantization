"""Phase 0A CPU-only audit. No model import, CUDA context, kernel launch or mutation of history."""
import hashlib,json,math,os,time
from pathlib import Path

TASK='LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
D=Path('/data/zypan/experiment_runs')/TASK
W=Path('/data/zypan/worktrees/ling-kda-rotation-dual-axis-functional-v1')
HIST=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1')
BASE=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1')
BRIDGE=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
DIR=Path('/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py')
LAYERS=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
SEED=202610160731

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY_VISIBLE_DEVICE_CONTRACT'
 # These rules are published before any matrix values or synthetic outcomes are computed.
 rules={'task_id':TASK,'phase':'0A_CPU_ONLY','rule_version':1,'seed':SEED,'rules_frozen_epoch':time.time(),
 'source_sha256':sha(__file__),'tensor_semantics':{'state':'[B, HV, Key, Value]','state_v_first':False,'actual_shape':[1,16,128,128],
 'rotation':'Value axis right multiplication only; codec-only inverse writeback to native coordinates','DualM_rows':'Key','DualM_columns':'rotated Value'},
 'rotation_ids':{'H':'872d4e45d21b03ccf8c3c11421331a07684f3b11ec6e5523ce7ced9254a44412','L6':'51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964'},
 'orthogonality_max_abs_threshold':1e-4,'orthogonality_threshold_source':str(BASE/'protocol/frozen_rotation_bank.json'),
 'orthogonality_threshold_source_sha256':'09707ffc0f76c512eae477ee06c453bd1511f134b09e7e899f32866dca9c1673',
 'floating_point_model':'IEEE round-to-nearest unit roundoff u=eps(dtype)/2; gamma(n)=n*u/(1-n*u); finite normal synthetic inputs',
 'roundtrip_bound':'E=abs(fl64(R R^T)-I)+gamma64(d)*abs(R)abs(R^T); B=abs(S)E+gamma_dtype(d)*(abs(S)abs(R))abs(R^T)+gamma_dtype(d)*abs(fl_dtype(SR))abs(R^T); residual<=max(B)*(1+gamma64(2048))+tiny',
 'recurrence_equivariance_rule':'At each of 128 steps and each real matrix: F(SR,vR) versus F(S,v)R; residual<=2*gamma64(8*d+32)*(1+maxabs((abs(D)abs(S)+abs(k)*beta*(abs(v)+abs(k)^T abs(D)abs(S)))abs(R)))',
 'small_matrix_tolerance':'2*gamma64(8*d+32)*(1+absolute-expression envelope), d=4',
 'counterexample_rule':'4x4 normalized Hadamard and unequal gates [0.1,0.3,0.6,0.9] must have maxabs(R^T D R-D)>0.1 and wrong-Key-gate state mismatch>1e-3',
 'synthetic_recurrence':{'steps':128,'K':128,'V':128,'heads_per_matrix':1,'matrices_per_condition':18,'decay_range':[0.60,0.95],'beta_range':[0.1,0.9],'qk':'unit normalized FP64 random vectors','value':'standard normal FP64'},
 'unquantized_FP32_roundtrip_drift':'record actual drift; no exact-native-logit parity assertion, no posthoc tolerance; structural bound is not a full-model forward error bound',
 'GPU_canary_requirements':'future independent canary must record FP_NATIVE versus unquantized H/L6 real logits using frozen FP32/native dtype; no GPU forward in this audit',
 'full_model_logit_tolerance_reuse':'No verified historical no-quantized-H/L6-vs-native full-model tolerance contract found. Existing fresh64 exact contract applies repeated same path, not native-vs-roundtrip.'}
 if not (D/'audits/MATH_EQUIVALENCE_RULES.json').exists():
  publish(D/'audits/MATH_EQUIVALENCE_RULES.json',rules)
 else:
  existing=json.loads((D/'audits/MATH_EQUIVALENCE_RULES.json').read_text())
  assert {k:v for k,v in rules.items() if k not in ('source_sha256','rules_frozen_epoch')}=={k:v for k,v in existing.items() if k not in ('source_sha256','rules_frozen_epoch')}, 'FROZEN_RULE_CHANGE_FORBIDDEN'
  rules=existing
 import torch
 torch.set_num_threads(1)
 assert not torch.cuda.is_initialized()
 g=torch.Generator(device='cpu');g.manual_seed(SEED)
 u64=torch.finfo(torch.float64).eps/2
 def gamma(n,dtype):
  u=torch.finfo(dtype).eps/2;return n*u/(1-n*u)
 def step(S,k,v,decay,beta):
  dec=S*decay.unsqueeze(-1)
  residual=v-(dec*k.unsqueeze(-1)).sum(-2)
  return dec+k.unsqueeze(-1)*(beta*residual).unsqueeze(-2)
 def envelope(S,k,v,decay,beta):
  dec=S.abs()*decay.abs().unsqueeze(-1)
  residual=v.abs()+(dec*k.abs().unsqueeze(-1)).sum(-2)
  return dec+k.abs().unsqueeze(-1)*(beta.abs()*residual).unsqueeze(-2)
 def roundtrip(S,R,dtype):
  X=S.to(dtype);A=X@R.to(dtype);Y=A@R.to(dtype).transpose(-1,-2)
  X64=X.double();R64=R.to(dtype).double();absR=R64.abs();dim=R.shape[-1]
  E=(R64@R64.transpose(-1,-2)-torch.eye(dim,dtype=torch.float64)).abs()+gamma(dim,torch.float64)*(absR@absR.transpose(-1,-2))
  B=X64.abs()@E+gamma(dim,dtype)*((X64.abs()@absR)@absR.transpose(-1,-2))+gamma(dim,dtype)*(A.double().abs()@absR.transpose(-1,-2))
  bound=float(B.max())*(1+gamma(2048,torch.float64))+float(torch.finfo(torch.float64).tiny)
  error=float((Y.double()-X64).abs().max());assert error<=bound,('ROUNDTRIP_ERROR_BOUND_FAIL',str(dtype),error,bound)
  return Y,{'dtype':str(dtype),'max_abs':error,'max_abs_analytical_bound':bound,'bound_pass':True}
 sources={}
 for p in [DIR,BASE/'scripts/capture.py',BASE/'scripts/transition.py',BASE/'scripts/replay_core.py',BRIDGE/'rotation_manifest.json',BASE/'protocol/frozen_rotation_bank.json',Path('/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py'),Path('/data/zypan/envs/ling-kda/lib/python3.11/site-packages/fla/ops/kda/fused_recurrent.py'),HIST/'SOURCE_CODE_FREEZE.json']:
  sources[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 assert sha(BASE/'protocol/frozen_rotation_bank.json')==rules['orthogonality_threshold_source_sha256']
 freeze=json.loads((HIST/'SOURCE_CODE_FREEZE.json').read_text())
 oldW=Path('/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1')
 for n in ['persistent_engine_v1.py','functional_worker.py']:
  assert sha(oldW/n)==freeze['files'][n]
  sources[str(oldW/n)]={'sha256':sha(oldW/n),'bytes':(oldW/n).stat().st_size,'historical_source_manifest_bound':True}
 manifest=json.loads((BRIDGE/'rotation_manifest.json').read_text())
 outputs={}
 for name in ['H','L6']:
  entry=next(x for x in manifest['bank']['conditions'] if x['condition_id']==name)
  p=BASE/'rotations'/f'{name}.pt'
  assert sha(p)==entry['artifact']['sha256']==rules['rotation_ids'][name]
  asset=torch.load(p,map_location='cpu',weights_only=False)
  assert asset['layer_ids']==LAYERS and asset['condition']==name
  R=torch.stack([asset['rotations'][l] for l in LAYERS]);assert R.shape==(18,128,128) and R.dtype==torch.float32
  R64=R.double();defect=(R64@R64.transpose(-1,-2)-torch.eye(128,dtype=torch.float64)).abs().amax((-2,-1))
  assert float(defect.max())<=rules['orthogonality_max_abs_threshold']
  S=torch.randn(18,128,128,generator=g,dtype=torch.float64)
  rt64,check64=roundtrip(S,R,torch.float64);rt32,check32=roundtrip(S,R,torch.float32)
  state=torch.zeros_like(S);op64=state.clone();op32=state.float();equiv_max=0.;equiv_bound_max=0.;codec_roundtrip_max=0.;codec_bound_max=0.
  for t in range(128):
   k=torch.randn(18,128,generator=g,dtype=torch.float64);k/=torch.linalg.vector_norm(k,dim=-1,keepdim=True)
   v=torch.randn(18,128,generator=g,dtype=torch.float64)
   decay=0.60+0.35*torch.rand(18,128,generator=g,dtype=torch.float64)
   beta=0.1+0.8*torch.rand(18,1,generator=g,dtype=torch.float64)
   native_next=step(state,k,v,decay,beta)
   rotated_next=step(state@R64,k,(v.unsqueeze(-2)@R64).squeeze(-2),decay,beta)
   expected=native_next@R64
   envelope_max=float((envelope(state,k,v,decay,beta)@R64.abs()).max())
   err=float((rotated_next-expected).abs().max());bound=2*gamma(8*128+32,torch.float64)*(1+envelope_max)
   assert err<=bound,('RECURRENCE_EQUIVARIANCE_FAIL',name,t,err,bound)
   equiv_max=max(equiv_max,err);equiv_bound_max=max(equiv_bound_max,bound)
   op64_raw=step(op64,k,v,decay,beta);op64,br=roundtrip(op64_raw,R,torch.float64)
   codec_roundtrip_max=max(codec_roundtrip_max,br['max_abs']);codec_bound_max=max(codec_bound_max,br['max_abs_analytical_bound'])
   op32_raw=step(op32,k.float(),v.float(),decay.float(),beta.float());op32,_=roundtrip(op32_raw,R,torch.float32)
   state=native_next
  outputs[name]={'artifact':{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size},'layers':LAYERS,'rotation_shape':list(R.shape),'rotation_dtype':str(R.dtype),'orthogonality_max_abs':float(defect.max()),'orthogonality_per_layer':[float(x) for x in defect],
  'one_step_roundtrip':[check64,check32],'synthetic_128_step':{'recurrence_equivariance_max_abs':equiv_max,'recurrence_equivariance_max_bound':equiv_bound_max,'each_step_bound_pass':True,'codec_FP64_max_single_step_roundtrip':codec_roundtrip_max,'codec_FP64_max_roundtrip_bound':codec_bound_max,
  'native_vs_unquant_codec_FP64_final_max_abs':float((op64-state).abs().max()),'native_vs_unquant_codec_FP64_final_relative_l2':float(torch.linalg.vector_norm(op64-state)/torch.linalg.vector_norm(state)),
  'native_vs_unquant_codec_FP32_final_max_abs':float((op32.double()-state).abs().max()),'native_vs_unquant_codec_FP32_final_relative_l2':float(torch.linalg.vector_norm(op32.double()-state)/torch.linalg.vector_norm(state)),
  'scope':'synthetic drivers only; no full-model, CUDA, learned outcomes or actual-data-forward'}}
 # A deliberate negative control: arbitrary Key rotation cannot keep a diagonal gate unchanged.
 R=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float64)/2
 gate=torch.diag(torch.tensor([.1,.3,.6,.9],dtype=torch.float64));S=torch.arange(16,dtype=torch.float64).reshape(4,4)/10
 k=torch.tensor([.5,.5,.5,.5],dtype=torch.float64);v=torch.tensor([.2,.3,-.1,.4],dtype=torch.float64);beta=.6
 native=step(S,k,v,gate.diag(),torch.tensor(beta,dtype=torch.float64))
 T=R.T@S;kp=R.T@k;transformed_gate=R.T@gate@R
 dec=transformed_gate@T;proper=dec+kp[:,None]*(beta*(v-kp@dec))[None,:]
 wrong=step(T,kp,v,gate.diag(),torch.tensor(beta,dtype=torch.float64))
 good_error=float((proper-R.T@native).abs().max());bad_error=float((wrong-R.T@native).abs().max());gate_noncommute=float((transformed_gate-gate).abs().max())
 assert good_error<=2*gamma(8*4+32,torch.float64)*(1+float(native.abs().max()))
 assert bad_error>1e-3 and gate_noncommute>0.1
 checks={'state_and_gate_axis_source_assertions':'PASS','historical_codec_only_inverse_writeback':'PASS','true_H_and_L6_SHA':'PASS','frozen_orthogonality_rule':'PASS','actual_H_L6_roundtrip_analytical_bound':'PASS','synthetic_128_step_value_equivariance':'PASS','nonuniform_Key_gate_negative_control':'PASS','no_CUDA_context':not torch.cuda.is_initialized()}
 assert all(v is True or v=='PASS' for v in checks.values())
 result={'task_id':TASK,'phase':'0A_CPU_ONLY','status':'PASS_STRUCTURAL_CODEC_ROTATION_WITH_OPERATIONAL_FINITE_PRECISION_CAVEAT','epoch':time.time(),'rules_sha256':sha(D/'audits/MATH_EQUIVALENCE_RULES.json'),'source_sha256':sha(__file__),'sources':sources,'checks':checks,'real_frozen_rotations':outputs,
 'Key_gate_negative_control':{'correct_conjugated_gate_error':good_error,'incorrect_unchanged_gate_error':bad_error,'gate_noncommutation_max_abs':gate_noncommute},'GPU_model_forward_count':0,'CUDA_initialized':torch.cuda.is_initialized(),'full_model_native_vs_no_quant_H_L6_logits':'NOT_RUN_PHASE1_REQUIRED','classification':'A_CODEC_ONLY_VALUE_ROTATION_NATIVE_RECURRENCE','pure_quantization_error_decomposition_claim_allowed':False}
 publish(D/'audits/MATH_CPU_TESTS.json',result)
 report=f'''# Phase 0A KDA旋转数学审计\n\n任务 `{TASK}`。结论：**A：只在量化codec坐标中做Value轴旋转，QDQ后逆变换回原生递归坐标**。本次CPU结构审计通过；真实模型无量化H/L6 logits差异尚未运行，不能把结构通过替代GPU资格。\n\n## 公式与轴\n\n实际state为 `[B,HV,K,V]=[1,16,128,128]`，FP32，`state_v_first=False`。Key在倒数第二轴，Value在末轴。令 `D_t=diag(exp(g_t))`，处理后的q、k按Key维L2归一化（q另乘K^-1/2），beta为每head标量。\n\n`S_minus=D_t S_prev`；`e=v_t^T-k_t^T S_minus`；`S_t=S_minus+beta_t k_t e`；`o_t=q_t^T S_t`。\n\n原native模型在o后执行原gated RMSNorm与o_proj。原persistent codec计算 `X=S_t R`，`X_hat=QDQ(X)`，`S_written=X_hat R^T`，下一token原生kernel实际消费S_written。未改gate、q/k/v、readout或FLA递归。\n\n原R128是对每个Key-row的128个Value坐标进行对称signed INT8：`scale=max(maxabs(row),1e-12)/127`，ties使用torch.round，code夹到[-127,127]，dequant乘scale，FP32 state写回。scale为原FP32；新Dual-M的row=Key、column=旋转Value，不能盲目将最后轴叫Key，不能改原R128为FP16 scale。\n\n对本次实际A，非均匀Key gate没有旋转坐标中的对角近似问题，因为每次都回到native。Value全变换亦满足 `F(SR,vR)=F(S,v)R`，无需改Key gate。但任意Key旋转T=R^T S若不把D改成R^T D R则不等价；负例已验证。\n\n## 源码证据\n\n- `{DIR}` 165–185：probe.begin('FP','native',None)，原full-model一步后取native cache；195–207：S@R→原R128→@R.T→state.copy_。\n- `{BASE/'scripts/capture.py'}` 36–49：q/g shape、state_v_first=False、FP32 state shape；调用原kernel不改递归。\n- `{BASE/'scripts/transition.py'}` 24–30：amax(dim=-1)、floor1e-12、/127、round、clamp、int8。该source仅用于qdq定义；其replay更新不能冒称本次数学worker。\n- `/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py` 823–830读取cache.keys；832–860原q/k/v卷积和gate/beta；880–894原fused调用；909–920更新nativecache及o_norm/o_proj。\n- `/data/zypan/envs/ling-kda/lib/python3.11/site-packages/fla/ops/kda/fused_recurrent.py` 174–197按Key逐元素gate、减去Key投影、写value update与readout；275–278给出两种state layout；本次明确使用Key-first。\n- 历史更早phase2 wrapper有另一路Value全坐标B（v@R、out@R.T），但当前model_step固定native，不执行该rotated分支；不能将两条路径混称。\n- 旧 `persistent_engine_v1.py` 92–153逐token检查真实initial_state与已QDQstate torch.equal，写回后未改MLA/conv；这一行为须在新四student资格实际再证，不能继承旧GPU标签。\n\n## 规则与实测\n\n规则先写入 `audits/MATH_EQUIVALENCE_RULES.json`，SHA `{result['rules_sha256']}`；随后才加载真实H/L6矩阵、计算误差。源/资产SHA详见 `audits/MATH_CPU_TESTS.json`，SHA `{sha(D/'audits/MATH_CPU_TESTS.json')}`。历史rotation-bank的正交maxabs门槛1e-4原样复用；没有发现可合法继承的native-vs无量化旋转full-model logits容差。\n\n小矩阵非均匀Key-gate负例：正确共轭gate误差 `{good_error:.17g}`，错误保留对角gate误差 `{bad_error:.17g}`，非交换残差 `{gate_noncommute:.17g}`。真实18层H/L6各完成128步synthetic recurrence与每步FP64矩阵表达式界检查。\n\n'''
 for name,row in outputs.items():
  t=row['synthetic_128_step']
  report+=f"- {name} artifact SHA `{row['artifact']['sha256']}`；orth maxabs `{row['orthogonality_max_abs']:.17g}`；128步Value递归等价残差最大 `{t['recurrence_equivariance_max_abs']:.17g}`；无量化codec循环FP32末态相对L2 `{t['native_vs_unquant_codec_FP32_final_relative_l2']:.17g}`。\n"
 report+='''\nCPU误差界只约束受测的有限正常synthetic输入的矩阵算术，不是full-model forward误差上界。冻结矩阵以FP32存储，R^T只是近似逆；逐token无量化roundtrip存在可测有限精度误差。后续即便结构正确，也必须称固定实际系统下的operational codec effect，不能声称严格纯量化误差分解。不得据此修矩阵、改dtype或FLA。\n\n初始CPU小矩阵负例fixture曾将β的一支误设为FP32、另一支FP64，导致无关舍入差异断言FAIL；原代码与失败JSON均保留在audits/MATH_CPU_INITIAL_FIXTURE_*。只将两支β显式同为FP64，未改冻结规则、阈值、模型、FLA或科学算法；随后原合同重测。

## 尚未完成的硬门禁\n\nFP_NATIVE与无量化H/L6真实canary logits差异、R128/Dual-M GPU parity、四student真实写回/next-kernel/fresh进程、8448-token五条件资源资格均NOT_RUN。CPU通过只允许继续Phase0B/0C准备，不能绕过这些门禁进入正式。若需要改变KDA递归或FLA才能等价则STOP；目前A路径没有该结构性需求。本审计GPU模型forward=0、CUDA initialized=false，未修改任何历史或生产源。\n'''
 with (D/'MATH_EQUIVALENCE_AUDIT.md').open('x') as f:f.write(report)
 print(json.dumps({'status':result['status'],'test_sha256':sha(D/'audits/MATH_CPU_TESTS.json'),'rules_sha256':result['rules_sha256'],'report_sha256':sha(D/'MATH_EQUIVALENCE_AUDIT.md'),'real_rotation_tests':outputs,'cuda_initialized':False},allow_nan=False))

if __name__=='__main__':main()
