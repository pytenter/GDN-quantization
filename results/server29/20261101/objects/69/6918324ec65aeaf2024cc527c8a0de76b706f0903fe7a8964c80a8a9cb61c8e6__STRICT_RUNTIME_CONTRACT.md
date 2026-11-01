Strict仍为新版B0 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；B1候选仅`:16:8`。模型Qwen3.5-9B，参数BF16，GDN recurrent carrier FP32；deterministic=True、fill=True、matmul/cudnn TF32=False、Torch/OMP/MKL线程1/1、固定非AIME输入、seed20261015，逻辑设备cuda:0映射到各自冻结物理UUID。

四份完整合同及237外部依赖的当前SHA/bytes复核通过。原输入完整32768 IDs raw/canonical SHA与冻结合同一致，1014条已提交输入/next/cache-cursor及记录Tensor描述/状态SHA谱系核验通过；这不是跨fresh数值比较。

实际执行库身份闭包未通过：`/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torchaudio/lib/libtorchaudio.so` 未列入28条prospective library路径或237条外部依赖。退出前完整/proc maps未保留；停止后该文件的当前2393848字节SHA为 `b7802c164ccf0911047a265ce489961615df2848fcca2403795cf854641ce073`，不能据此追认运行资格。它是排序遍历中首个被拒绝路径，不能保证只补这一个库就能通过。详见 `SOURCE_AND_AUDIT_BINDINGS.json`。
