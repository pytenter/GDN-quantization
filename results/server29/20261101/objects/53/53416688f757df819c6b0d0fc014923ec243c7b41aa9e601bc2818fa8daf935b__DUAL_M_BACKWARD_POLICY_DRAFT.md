# Dual-M backward policy — 未冻结审查草案

状态：DRAFT_PENDING_AUTHORITATIVE_C6R_PROVENANCE。本文是准备方案，不是资格 PASS，不授权 GPU、A800 或正式训练。权威 A800 clean-room trainer、其真实调用依赖、配置和完成回执仍由 provenance 审计绑定；不以早期同名训练脚本替代。

## 前向合同

训练 Arm D 直接调用已经用于正式 32K 推理的冻结 Dual-M encode/decode。只包装其 autograd 边界，不重新实现 codec 公式。保留 INT8 codes、FP16 row/column scales、一次冻结 column LS、最终 re-encode、torch.round ties-to-even、原 clamp 和 decode 乘法次序。正式 codec 的 source() 已显式 detach().float()，直接替换旧训练函数会断开旋转梯度。

必须逐位比较 codes、row scale、column scale 和 reconstruction；浮点比较还检查 signed-zero/原始字节。不得用仅数值近似的重写公式冒充同一个 forward。codec、reference 和 wrapper 的源 SHA 与 AST 均需绑定。

候选最小实现为一个 Python torch.autograd.Function，无 CUDA backward、无 recurrence/kernel 改写：forward 获得冻结 codec 的 detached hard reconstruction，原样返回其数值；backward 仅将上游 cotangent 传给 FP32 state 输入。codes/scales 作为 detached 诊断，不允许第二条隐含 scale 梯度路径。

不能不经验证就用 `x + (hard - x).detach()` 宣称逐位 forward 相同：FP32 消减和重新相加可能改变末位。原共享 C128 helper 使用该 identity-STE 写法；是否属于权威最终 trainer 及实际允许的前向差异，必须待 provenance 核验。候选共同 exact-forward wrapper 要与权威旧 C128 的 input-gradient 逐值比较，且把任何 forward 差异明确记录，不隐藏修改。

## 候选固定 backward

- 对整个 QDQ：`dL/dx = upstream_gradient`，包括 rounding ties、clamp 饱和、FP16 scale floor/ceiling。此处是 identity STE surrogate，绝不宣称为 hard codec 的真导数。
- row scale 估计、初始 column scale、LS column fit、FP16 转换、clamp、最终 re-encode：全部 detach，无 scale-gradient 或 LS-gradient。
- 对旋转的连续矩阵乘、原 Cayley solve、原模型计算和真实 Loss：使用原 PyTorch autograd，不 detach 参数，不自创 backward。
- 原权威 C128 若确实采用整个 QDQ identity STE，则 A/D 共用同一 identity backward；两组仅 hard codec 不同。若权威 C128 使用不同 policy，先报告冲突，不擅自切换。
- 返回的 hard QDQ state 必须成为下一 token 唯一 student recurrent state。高精度 pre-QDQ state 可用于本 token backward graph，但不得作为下一步数值历史；不增加 hidden FP shadow cache。
- 保留权威 trainer 的 BPTT horizon、detach 边界、teacher/student construction 和 checkpoint 选择规则。不能通过本草案擅自设定或延长 horizon。

## CPU 资格设计（未执行）

### 1. 冻结前向及边界

对同一 FP32 输入，比较冻结推理 codec 与训练包装后的 hard forward：codes、row/column scales、reconstruction、dtype、shape 和全部原始字节。真实状态形状为 `[B,H,K,V]=[1,32,128,128]`；还测试单 head、多个 batch/head、非 contiguous 输入及正式按层批处理。实际 captured-state 如可安全读取，应只在服务器 CPU 上使用，不下载大 tensor。

输入至少覆盖：全零、全常数、正负对称、单点及稀疏 outlier、跨行/列不同动态范围、seed 固定随机；FP16 LOW=`2**-24` 和 HIGH=`65504` 的下方/恰值/上方（包含 nextafter）；half conversion 的相邻表示 midpoint；rounding 的正负 k+0.5 与 nextafter 邻点。直接 unit-test 冻结 levels/safe_half；全 codec 路径另外测试最终实际 levels。不能只构造名义 ties 后因 scale fitting 改变而假称最终 re-encode ties 已覆盖。

NaN/Inf 输入必须按冻结 source 检查拒绝。大 outlier 若触发原 forward 的非有限中间量或原 scale-risk STOP，测试应记录预期失败类别；不得为了让边界测试 PASS 修改 clamp/floor/LS 分母或安全阈值。边界测试成功不代表所有极端数值都被准许运行。

### 2. STE 局部规则

对任意固定上游梯度 g 验证 `autograd.grad((STE(x)*g).sum(), x) == g`，分别覆盖零值、正常区、ties 和饱和区。codes/scales 均不 requires_grad；正向与 input.requires_grad=False 时完全相同；backward 不改变 hard 输出，也不保留跨调用缓存。

同输入和同上游梯度下，与权威旧 C128 STE 的 input-gradient 做精确比较；必要时比较连接同一权威 Cayley 参数和原 Loss 后的参数梯度，分别披露 forward 末位差异。先测试非退化输入；全零 Loss 的梯度为零不是失效，不能把这一例强行要求为非零。

### 3. 原坐标及旋转 graph

仅提取权威原函数，不改写 recurrence。使用原 state recovery 和原 q/k 旋转方向：state 旋转轴 K（-2），q/k 轴最后一维；value 轴不新增旋转。Hadamard 后 Delta 保持两个原乘法，恢复顺序按原定义。

在无量化的连续部分检验 identity-coordinate readout/recovery 与有限差分；可用 double 的小维度纯连续 reference 做方向导数检查。该测试不跨越 hard rounding，不要求 hard codec 有限差分等于 STE。实际 128 维 FP32 Cayley 仍使用原正交阈值；不能把 double 检查替代原运行数值合同。

让真实 hard codec 输出通过 identity STE 连到原 Cayley theta，验证有限、非全零 gradient 和参数覆盖，且两 arm 参数集合相同。梯度非零只证明图连接，不证明预测改善。

### 4. recurrent 与更新

使用原权威 recurrent 函数的小 CPU 状态/短序列，核对每 token 确实消费上一 hard QDQ state 的值/身份；与脱离 autograd 的冻结推理 recurrence 逐 token 前向比较。禁止引入替代 recurrence。复用原 Loss 能力可验证时，再做三个连续小 micro-updates，记录 Loss、grad norm、参数/optimizer/RNG hash、正交性。

固定 seed 重复同样的 CPU micro-updates；checkpoint 保存和恢复 rotation、optimizer、Python/Torch RNG 及必要 student state，比较下一步和不中断控制流。若要验证 BPTT 中间恢复，必须明确 graph 无法由普通 checkpoint 序列化，选择原合法 detach/optimizer-step 边界，不伪称恢复了未保存计算图。

## 尚未完成的资格

所有前向 parity、C128 gradient parity、Cayley graph、三步更新、checkpoint 恢复以及 3090 micro-test 目前均 NOT_RUN。取得权威 provenance 后只选择一个有界实现，CPU 资格完成并冻结后再由主协调器执行最短授权 GPU 测试。局部资格不等于完整 9B training PASS，更不等于 Dual-M-aware 科学收益。
