# Phase 0A KDA旋转数学审计

任务 `LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1`。结论：**A：只在量化codec坐标中做Value轴旋转，QDQ后逆变换回原生递归坐标**。本次CPU结构审计通过；真实模型无量化H/L6 logits差异尚未运行，不能把结构通过替代GPU资格。

## 公式与轴

实际state为 `[B,HV,K,V]=[1,16,128,128]`，FP32，`state_v_first=False`。Key在倒数第二轴，Value在末轴。令 `D_t=diag(exp(g_t))`，处理后的q、k按Key维L2归一化（q另乘K^-1/2），beta为每head标量。

`S_minus=D_t S_prev`；`e=v_t^T-k_t^T S_minus`；`S_t=S_minus+beta_t k_t e`；`o_t=q_t^T S_t`。

原native模型在o后执行原gated RMSNorm与o_proj。原persistent codec计算 `X=S_t R`，`X_hat=QDQ(X)`，`S_written=X_hat R^T`，下一token原生kernel实际消费S_written。未改gate、q/k/v、readout或FLA递归。

原R128是对每个Key-row的128个Value坐标进行对称signed INT8：`scale=max(maxabs(row),1e-12)/127`，ties使用torch.round，code夹到[-127,127]，dequant乘scale，FP32 state写回。scale为原FP32；新Dual-M的row=Key、column=旋转Value，不能盲目将最后轴叫Key，不能改原R128为FP16 scale。

对本次实际A，非均匀Key gate没有旋转坐标中的对角近似问题，因为每次都回到native。Value全变换亦满足 `F(SR,vR)=F(S,v)R`，无需改Key gate。但任意Key旋转T=R^T S若不把D改成R^T D R则不等价；负例已验证。

## 源码证据

- `/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py` 165–185：probe.begin('FP','native',None)，原full-model一步后取native cache；195–207：S@R→原R128→@R.T→state.copy_。
- `/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1/scripts/capture.py` 36–49：q/g shape、state_v_first=False、FP32 state shape；调用原kernel不改递归。
- `/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1/scripts/transition.py` 24–30：amax(dim=-1)、floor1e-12、/127、round、clamp、int8。该source仅用于qdq定义；其replay更新不能冒称本次数学worker。
- `/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py` 823–830读取cache.keys；832–860原q/k/v卷积和gate/beta；880–894原fused调用；909–920更新nativecache及o_norm/o_proj。
- `/data/zypan/envs/ling-kda/lib/python3.11/site-packages/fla/ops/kda/fused_recurrent.py` 174–197按Key逐元素gate、减去Key投影、写value update与readout；275–278给出两种state layout；本次明确使用Key-first。
- 历史更早phase2 wrapper有另一路Value全坐标B（v@R、out@R.T），但当前model_step固定native，不执行该rotated分支；不能将两条路径混称。
- 旧 `persistent_engine_v1.py` 92–153逐token检查真实initial_state与已QDQstate torch.equal，写回后未改MLA/conv；这一行为须在新四student资格实际再证，不能继承旧GPU标签。

## 规则与实测

规则先写入 `audits/MATH_EQUIVALENCE_RULES.json`，SHA `c8061748347de208af551b24132c19e16a91f621256b5fb0dc5dae9b451605d4`；随后才加载真实H/L6矩阵、计算误差。源/资产SHA详见 `audits/MATH_CPU_TESTS.json`，SHA `795c1625973c86f768e350b0a4de30ab8046162bea4ec4cc949692257a20c6d9`。历史rotation-bank的正交maxabs门槛1e-4原样复用；没有发现可合法继承的native-vs无量化旋转full-model logits容差。

小矩阵非均匀Key-gate负例：正确共轭gate误差 `1.6653345369377348e-16`，错误保留对角gate误差 `0.40500000000000003`，非交换残差 `0.42500000000000004`。真实18层H/L6各完成128步synthetic recurrence与每步FP64矩阵表达式界检查。

- H artifact SHA `872d4e45d21b03ccf8c3c11421331a07684f3b11ec6e5523ce7ced9254a44412`；orth maxabs `3.4228545331416171e-08`；128步Value递归等价残差最大 `1.1102230246251565e-15`；无量化codec循环FP32末态相对L2 `4.7292648087154235e-07`。
- L6 artifact SHA `51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964`；orth maxabs `2.3231553025748752e-07`；128步Value递归等价残差最大 `1.3322676295501878e-15`；无量化codec循环FP32末态相对L2 `1.6800917016245014e-06`。

CPU误差界只约束受测的有限正常synthetic输入的矩阵算术，不是full-model forward误差上界。冻结矩阵以FP32存储，R^T只是近似逆；逐token无量化roundtrip存在可测有限精度误差。后续即便结构正确，也必须称固定实际系统下的operational codec effect，不能声称严格纯量化误差分解。不得据此修矩阵、改dtype或FLA。

初始CPU小矩阵负例fixture曾将β的一支误设为FP32、另一支FP64，导致无关舍入差异断言FAIL；原代码与失败JSON均保留在audits/MATH_CPU_INITIAL_FIXTURE_*。只将两支β显式同为FP64，未改冻结规则、阈值、模型、FLA或科学算法；随后原合同重测。

## 尚未完成的硬门禁

FP_NATIVE与无量化H/L6真实canary logits差异、R128/Dual-M GPU parity、四student真实写回/next-kernel/fresh进程、8448-token五条件资源资格均NOT_RUN。CPU通过只允许继续Phase0B/0C准备，不能绕过这些门禁进入正式。若需要改变KDA递归或FLA才能等价则STOP；目前A路径没有该结构性需求。本审计GPU模型forward=0、CUDA initialized=false，未修改任何历史或生产源。
