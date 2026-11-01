当前结论：CPU准备检查通过；9B/GPU数学、GPU fresh重复性、恢复吞吐均 NOT_RUN。当前不能批准任何正式GPU科研。

真实CPU检查使用已绑定90d929…架构中的 Qwen3_5ForCausalLM、native chunk/recurrent、SDPA、MLP 和 Cache，随机tiny双层权重、FP32/BF16、4 token，固定CPU seed。原native recurrent与独立FP64小例比较、finite、FP32 state dtype/shape、cursor/KV增长、实际next-state消费和相同进程内重新构造重复均通过。它不是9B权重运行，也不是两个GPU fresh进程。

观察器另用真实tiny CPU Cache检验dict形式recurrent_states/conv_states的完整遍历、attention KV、metadata、JSON roundtrip与元数据改动可见性；原始字节差分保留signed-zero字节不等，并能给出实际数值差。独立无引擎fake/AST测试及控制器合成测试另附。所有CPU入口均限制CUDA_VISIBLE_DEVICES为空，CPU fixture拒绝torch.cuda._lazy_init；没有GPU查询、初始化或dispatch。

当前可恢复的旧Torch chunk保留native g.cumsum，未安装NEW的compensated替换。历史helper装饰器可能选择hub/FLA，完整历史实际物理分派未保存。因此源码相同不代表历史kernel逐字节相同；未来必须记录当前真实环境和依赖选择。

未来最小数值组为两次独立fresh、同卡同输入的32-token前缀。完整state/conv/KV/logits与Cache元数据均比较，包含JSON序列化规范化；非法重叠state存储、缺层、错误cursor、非finite或首差异即停止。库身份在load后首次forward前及执行后检查，OLD各单位同阶段critical path/SHA签名必须与首fresh一致；不要求pre/post集合相同，以保留合法lazy loading。

独立FP64 state oracle使用实际模型的q/k/v/g/b激活，从token0持续递推。state gate=1e-5+1e-4×max(abs(reference_state))；core输出gate=.001+.01×max(abs(reference_core))；normalization gate=.0001+.02×max(abs(FP64_normalization))。这些是当前事前冻结的小前缀工程拒绝门槛，不是已验证的9B误差、论文通用容差、全模型oracle或32K/81K误差上界。BF16量化/归一化与FP32累加范围分开审计，不能事后加宽阈值。

历史没有足以与这次逐token比较的完整OLD字节参考，故不能宣称与原OLD输出byte-exact。NEW_STRICT byte parity不是本任务要求。若未来自身fresh不等，保留首tensor/element/raw byte、maxabs/relative及metadata差异，标记RUNTIME_REPRODUCIBILITY_FAIL并停止性能晋级。

内存安全结论限于静态路径审查、已核验依赖及未来动态不变量；finite不是所有未初始化读的普遍证明。本轮没有关闭fill保护、没有改dtype或state数学。128+128的性能轨迹也不自动证明长程稳定性。现有AIME完成历史仅为范围受限的历史运行事实。
