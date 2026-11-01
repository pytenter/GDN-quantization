# HISTORICAL_PORT_PERSISTENT_INT8_GROUP32_V1

三条件相同H128 q/k注入、FP native helper、BF16权重和输入，采用一次一个token的原生stepwise路径。首token forward结束后即对全部24层FP32 recurrent state QDQ并原位写回，此后每token相同操作。拒绝query sequence length>1的chunk；不能把chunk末尾一次QDQ称为逐token。

在下一native GDN helper入口读取真实initial_state，临时CPU解码上一token的整数codes/FP32 scales/整数zp或negative-zero flags，然后比较实际Tensor dtype/shape/torch.equal及完整字节。不是指针或hash替代。预期编码仅诊断使用，不参与模型输入；不存在未量化recurrent shadow。观察模式对全部head/全部层检查；CPU小模型已实际验证，9B GPU仍NOT_RUN。

canary保留全部相关preQDQ、整数fields、postQDQ/writeback、实际kernel initial_state、logits与全cache raw。无损重复字节可共用packet offset，须实际字节相等后去重。fresh0流式永久归档29、独立SHA/bytes原子发布后提交receipt；fresh1直接读取参考实际raw比较，首差异完整保留当前raw，禁止前缀重跑取证。临时CPU解码/观测不成为GPU recurrent buffer。

实际持久carrier仍为原生48MiB FP32 QDQ cache，不能称物理packed INT8。测量关闭观察/逐tokenhash/远端raw传输，保留必要codec和finite检查；128 warmup后同步计时128token，末尾再次同步。不沿用历史OLD首token漏QDQ或AIME整段未QDQ prefill协议。
