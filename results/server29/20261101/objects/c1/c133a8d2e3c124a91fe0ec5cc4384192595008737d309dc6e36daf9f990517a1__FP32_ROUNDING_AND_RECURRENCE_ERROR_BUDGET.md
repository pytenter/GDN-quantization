# V3数值模型、已验证范围与未解决的功能接受范围

## 真实计算位置与结构

锁定真实modeling_qwen3_5.py SHA256为90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a。Q/K投影先经过因果卷积和SiLU，再reshape并repeat_interleave(2)；归一化在原BF16 dtype完成，之后转FP32并进行A32矩阵乘。旋转位于非线性之后。g来自FP32 A_log/softplus，alpha=exp(g)为逐head标量；beta来自sigmoid(b)，也是逐head标量。Native state布局为batch,32,Key128,Value128；readout沿Key轴归约，Query包含1/sqrt(128)。首个单token prompt走chunk helper（padding64），其后走recurrent helper。原运行使用绑定的Torch helper，不能把尚未核验的FLA fused kernel视为等价实现。

每层16个真实Q/K head共享给相邻32个recurrent/value head，映射h→floor(h/2)。同一共享对使用同字节U/A；4096个真实Query样本不会因复制成为8192个独立样本。全部24层、384独立头实际核验，无跨层/跨head池化或重拟合。

## 精确算术恒等式

令A=UH且A^T A=I，SA=A^T S、kA=A^T k、qA=A^T q。原公式

Snew=alpha(I-beta k k^T)S+beta k v^T

左乘A^T，与旋转坐标中的同一秩一delta更新一致；(SA)^T qA=S^T q。此恒等式仅要求同一q/k/v/alpha/beta输入、正交A及正确布局，不允许把旋转穿过SiLU/卷积，也不允许改变归一化输入。独立CPU使用稠密transition与秩一更新两种实现验证全部384矩阵的一步，以及三个固定矩阵各128连续步。

## 表示误差及矩阵算术

部署固定为A32=castFP32(U64 H64)，不根据哪条路径通过旧2e-6选择。所有现有U/A逐字节身份保持。对于DeltaA=A32-A64，有

||A32^T A32-I||2 <= 2||DeltaA||2+||DeltaA||2^2+FP64评估余量。

对C64的二次型，矩阵表示造成的绝对偏差受||C||2(2||DeltaA||2+||DeltaA||2^2)及原M0残差控制。C32转换和FP32检查器的两次矩阵乘另行计入。M0–M3全部384头统一扫描；没有重新eigen拟合或选择更有利的U。

FP32单位舍入u=2^-24，gamma_n=nu/(1-nu)。标准有限、正常范围、无TF32降精度的长度128内积具有componentwise gamma界。该模型不是“所有CUDA操作都已证明满足该界”：CUDA exp/rsqrt、归约方式和实际dtype必须另行核验；CPU结果不能替代CUDA资格。每个表达式的乘加数量、绝对值尺度、下溢绝对项和FP64参考评估误差分别处理，不能直接把通用最坏情况上界写成实测精度保证。

## 固定输入递归预算

对同一输入与同一初态，M=alpha(I-beta k k^T)。由于这是对称秩一修正，

||M||2=|alpha| max(1,|1-beta ||k||2^2|)。

使用实际归一化后Key范数，不能假设BF16归一化后恰好单位长度。若局部残差范数为eta，则固定输入的传播界为e_next<=||M||2 e_prev+eta。实际CPU工具逐步比较真实Torch helper与独立NumPy FP64图：decay乘法、128项Key读取、减法、beta乘法、外积及state相加分别传播componentwise预算；readout另作内积界。alpha以实际CPU FP32 exp(g)结果作为此局部参考的输入，未假装给出了CUDA exp的独立误差证明。

A32并非精确正交也要计入：忽略舍入时，变换后递归与直接变换原递归的额外项为-alpha beta A32^T k k^T(A32 A32^T-I)S；其范数可由|alpha beta| ||A32^T k|| ||k|| ||A32 A32^T-I|| ||S||控制。readout同理有||S|| ||A32 A32^T-I|| ||q||项。它们不应隐藏在“4倍H误差”常数中。

## 完整模型不是上述同一线性系统

完整9B比较中，各层之前的readout经过BF16舍入、归一化、投影、SiLU/MLP、attention和残差连接。输入q/k/v/g/beta也随hidden state改变。此时不能继续仅使用E_next=M E_prev+eta而忽略输入扰动。

例如令参数从(alpha,beta,k,v)变成带撇号的值，则还必须控制(M'-M)S和(beta' k' v'^T-beta k v^T)。其中Key外积扰动可用||delta_k|| (||k'||+||k||)控制；还需要delta_alpha、delta_beta、delta_v以及它们经过完整网络的传播。仅用实际观察到的这些扰动再去“解释”结果，是一致性审计，不是提前建立的功能正确性接受门限。

BF16 rounding cell边界会将很小的连续算术扰动变成一个离散步长；局部FP32矩阵误差小并不等于完整模型logits误差小。相反，最坏情况全网络Lipschitz界若巨大或无信息，也不能作为可信的功能等价证明。

## token22的证据结论

保存对象是layer14的32×128有效Key比较，保存dtype为float64；UH有效Key先FP32旋转，再以FP64回投原坐标。最大位置recurrent head18、Key坐标78，对应真实共享head9。Native值0.56640625，UH回投值0.5898437745114653。最大差0.023437524511465302，相对L2为0.013733534416246003。

UH回投值接近BF16值0.58984375，与Native之间为6个该数值范围的BF16步长（1/256）。这只是与原dtype结构一致的条件重建：实际旋转前Key未保存，不能将最近BF16格点冒称原始观测。回投的小数偏移2.45114653e-8可与矩阵表示/乘法预算一致；它不能证明是哪一个更早的layer/token首先产生了整体漂移。

44/4096个元素的误差大于1/256；最大元素约占总平方误差9.1099%。所以既不能只说“一个极值”，也不能把5.07e-8门限超出量当成实际模型误差。原门禁确实在4H边界附近，但停止点不是已知的首次非零差异。

缺失：匹配的Native/H state快照、此前各token完整轨迹、原始Q/K及归一化前输入、gate/beta/value、对应layer输出与full-vocab logits。保存的当前C1 state覆盖24层且有限，但没有对应原生state，不能据此唯一恢复传播原因。

## V3接受范围的最终判断

结构恒等式、全部矩阵及CPU固定输入算术已经取得独立证据。真实CUDA局部误差及完整9B 128-token功能等价尚未执行。现有证据无法给出一个既有独立依据、又对完整网络功能漂移有实际约束力的预登记接受范围。4H、0.024/0.025、任意额外epsilon、仅Relative L2或仅MaxAbs均不作为替代。

因此C3接受合同为UNRESOLVED，V3 GPU准入关闭，按用户规定一次性V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP。这里没有断言UH数学错误或不可能建立更好的模型，也没有把未完成的128-token真实模型测试标成PASS。
