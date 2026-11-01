# 数值审计

CPU prototype原跨轨迹screen仍FAIL：worst output NRMSE=2.42456077%，不改0.5%阈值、不把CPU数值路径当成GPU数值路径。正式CPU fallback被代码拒绝。

目标CUDA两fresh完整case与fast40case：output worst=0.09813022%，delta=0.09871565%，boundary=0.06652701%；原0.5%筛查数值不变。same-actual-state output=0.05607616%，小于原0.1%。无非有限值、错计数或tail隐式flush。

旧局部rounding heuristic的687个UNEXPLAINED标签仍在原工件；实际同窗口base替换反事实验证见LINEAGE_COUNTERFACTUAL_V1：1077个跨轨迹code差异在换为同base的sequential replay后消失，另2个是同base局部adjacent rounding；同base所有本地差异均可由原量化编码逐值重现且局部未解释数0。samebase u/boundary worst NRMSE=1.17116959e-07/7.61724367e-08。这是继承base分叉的运行证据，不是把所有不同code都要求处于同一个当步舍入边界。

独立FP64 oracle使用真实packed输入/实际u/g，BF16输出对同样BF16舍入oracle最坏约0.01915%；该证据说明局部算子，不保证80K生成等价。

真实32层/24GDN非AIME teacher-forced模型：prefill bitwise相等、marker相等；decode开始有差异，logits worst NRMSE=6.409093%，GDN module输出worst=16.253947%。下游逐层传播并非单个相同输入算子的0.5%筛查对象；未声明模型输出等价。新两配置作为独立FP32数值实现，历史21/30与0/30不复用、不继承。

CODE_VERIFIED / RUNTIME_VERIFIED见各脚本与原始JSON；80K稳定性、准确率和长上下文吞吐在真实formal完成前UNRESOLVED。科学准入最终判定另见validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json，不把CPU原FAIL改写为PASS。
