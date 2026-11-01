参考为NEW_STRICT B0=:4096:8；候选B1=:16:8，均在Torch/CUDA初始化之前通过fresh进程设置。Qwen3.5-9B参数BF16、GDN state FP32、固定非AIME输入prefix256、seed20261015、deterministic/fill=True、matmul/cudnn TF32=False、Torch/OMP/MKL线程1/1。每个对照绑定同一原GPU UUID，逻辑cuda:0；四卡之间不互用字节参考。

四卡B0两fresh全256及全部保存logits/GDN/state/cache均逐字节一致，范围仅此短程FP基线。B1 GPU3 token0存在真实差异；其他三卡被共享停止，不判为数值失败。H和Dual-M未获得当前候选资格。32库路径/26实际库文件及241外部依赖当前SHA匹配，8个完整单位均有真实PREFLIGHT/POSTFLIGHT14库证据；失败B1只有PREFLIGHT，缺失POSTFLIGHT保留缺失，不用当前磁盘SHA追认它。逐份合同保存在remote_evidence/run/gpuN/analysis/STRICT_BASELINE_CONTRACT.json。
