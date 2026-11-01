# 实现审计

CODE_VERIFIED：类型B，低比特base + packed K4/FP32 u/FP32 log-decay。src/lq_unroll_buffer.py::step在decode每步调用packed_readout；src/packed_direct_read.py::_kernel融合INT4/INT8分组解码、base两个matvec、pending向量读出和本步u/output，没有全局dense State输出。正式生成只有UnrollEngine。replay8只作明确要求的固定工作量reference。

Token1…P−1：packed_base不变，追加actual u；每步临时批解码K，BF16 native norm后FP32向量。TokenP先算u和output，追加本条，然后_delta_state用实际u一次顺序重建FP32矩阵，原PackedAffineState.from_tensor提交并清空引用。未满窗口EOS/cap不补forward、不flush，preserve尾部。Prefill完全委托原PACEBuffer，256/prompt_end原语义。最后sampled token未forward，decode_forwards=generated−1。

没有持久dense FP32 State；None slot与Python引用无tensor payload。临时boundary State与allocator reserved必须分别解释。批处理K与逐条原codec/native BF16norm逐元素一致，验证和四卡真实9B smoke保留。

两配置json只差decode_period=8/16；量化器、K4、head mask、sampler、FrozenV4均复用冻结字节。非官方LeapQuant kernel复现声明：仅借鉴可核验公式，自研Triton实现，不声称官方实现相同。
