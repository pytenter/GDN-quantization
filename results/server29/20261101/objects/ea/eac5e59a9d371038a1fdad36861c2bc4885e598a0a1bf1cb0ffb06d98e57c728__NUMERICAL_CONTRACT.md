只读继承原407中qualification/GPU_PLAN.json及worker的事前合同，不修改容差。

两独立fresh使用同卡同模型/输入/seed/init，各32 token。原封存程序比较完整TRACE（含每Tensor raw SHA与metadata），首差异停止性能晋级；不能把该程序输出冒称显式torch.equal或全体直接memcmp统计。

激活驱动CPU FP64 oracle按实际q/k/v/g/b从token0连续递推。state误差门槛1e-5+1e-4*maxabs(reference_state)；core输出 .001+.01*maxabs(reference_core)；normalization .0001+.02*maxabs(FP64_normalization)。这些小prefix冻结拒绝门槛不是全模型FP64证明、32K/81K误差上界或本轮已验证数值。任何真实失败不得事后扩宽阈值或更换NEW数学。

本次更完整输出要求与封存程序仍有差距，详见analysis/RECOVERY_REQUIREMENTS_COMPATIBILITY_REVIEW.json：显式torch.equal、全程逐Tensor误差与首layer/head、成功轨迹所有关键激活raw、模型加载独立计时与持续NVML峰值等，不能编造已记录。本轮所有真实GPU数学与fresh结果NOT_RUN。
