# 快速 Direct Read 验证

C1/C3完全复用已封存AIME正式的src/pace_buffer.py、pace_runtime.py、pace_quantizer.py、lq_unroll_buffer.py、packed_direct_read.py及vendor；SHA见SOURCE_IDENTITY.json与provenance/VALIDATION_REUSE_ADMISSION.json。原目标CUDA40cases/FP64同state/lineage证据逐SHA复核复用，不重复门禁。原CPUprototype失败、Direct-vs-Replay完整模型非bit等价均保留，不改称PASS。

新RUNTIME_VERIFIED：四卡9B均完成16prompt+33decode四配置固定非HMMT序列；C0原生逐字一致。新引擎C0/C1/C3实际采样33上限、强制第2个sample为EOS，确认decode=sample-1、EOS不forward、preserve尾部、dP8/16边界重建数floor(decode/P)、无窗口内完整State常驻。C1/C3 backend为TRITON_PACKED_DIRECT，未退回逐token完整State replay；prefill沿原实现。

原误差门槛未改：crossReplay0.5%、sameactualstate0.1%。C2旋转Direct算子通过，但FP_H完整模型失败而禁入。
