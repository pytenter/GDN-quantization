# 最终内存、性能口径与工程结论补充

状态：COMPLETE_BOTH_CONFIGS。全部实验/主交付与完整额外55/29/Mac来源审核已PASS；本件仅补齐比较表、权重与KV/CPU时间作用域，无新模型加载、GPU运行或评分。

|指标|原Replay dP8|快Direct dP8|快Direct dP16|
|---|---:|---:|---:|
|同GPU1固定输入完整模型decode TPS|5.081395084750404|6.6531946365133345|6.794332373139476|
|Packed base bytes|8896512|8896512|8896512|
|Update buffer容量bytes|1837056|3397632|6795264|
|实测各层峰值payload之和bytes|1837056|3397632|6795264|
|CUDA allocated peak bytes|17949048832|17950618112|17954518528|
|CUDA reserved peak bytes|17976786944|17978884096|17985175552|
|128计时forward全部24层State写回|384|384|192|
|fused readout GPU图中位µs|UNMEASURED|19.097599983215332|27.953920364379883|
|本次AIME实际pooled decode forwards/s|UNMEASURED|6.07756409391|6.441573664387069|

固定输入TPS是完整模型decode阶段（排除prefill/加载/编译），不是GDN微核TPS；实测写回以after-before计数扣除warmup和prompt。上表峰值各层求和不是全层同时live承诺；allocator峰值不等于payload。原Replay没有本轮新AIME轨迹，不用旧21/30轨迹的不同上下文TPS冒充严格对照。

## 权重与端到端footprint

METADATA_VERIFIED：只读4个safetensors头、未读取tensor数值，checkpoint文件共19306310880bytes，头内声明tensor payload共19306216416bytes、775个tensor。dtype元素计数：{'BF16': 9653100528, 'F32': 3840}。
这描述磁盘checkpoint，包含的命名空间/权重别名可能与AutoModelForCausalLM实际参数集合不同，不能声称是GPU live weight bytes。CODE_VERIFIED/RUNTIME_VERIFIED：冻结loader在src/pace_runtime.py:296实际检查所有已加载parameter为BF16/CUDA0；没有保存完整GPU parameter存储/别名inventory，故该项实际payload仍UNMEASURED。上表整模型allocated/reserved是真实端到端allocator footprint，另给State+marker/head-map的有效bpv，不把两者混成量化bit数。

INFERRED：模型config有8个full-attention层、4个KV heads、head_dim=256；若完整BF16 K/V追加，每context token为32768bytes。各题context=input_tokens+generated−1的推算见COMPARISON_DATA。没有实际KV hook，真实KV峰值/容量/allocator生命周期UNMEASURED。

## CPU、IO与校准时间

|原CPU阶段|真实owner起止区间s|
|---|---:|
|CPU_SCORE_FRESH1|26.039300382|
|CPU_SCORE_FRESH2|25.038271338|
|CPU_REPORT_FINISHER|2.004103675|

均用同55主机monotonic相减，来自原STARTED/OWNING_EXIT实际整数0；没有跨主机比较时钟。两fresh CPU评分只读取既有输出。每题模型加载/生成/prefill/decode/dispatch总值见原FINAL_MEMORY_RUNTIME_DATA；加载含部分IO/核验，纯IO和单元排他审计时间没有单独计时，不能从wall差额伪造。当前两配置未重新校准，继承冻结校准/mask；没有新校准耗时。GPU active/SM seconds未测，不拿dispatch-wall或CPU区间冒充。

## 对原任务八个问题的结论

1. 是实际packed-base LeapQuant-style Direct Read：Triton融合base/更新读出、每步临时批量原codec/BF16norm K；没有跨token持久完整FP32 State，只有周期末临时重建。
2. 两个正式配置逐字段核对，只差decode_period=8/16。模型/量化器/mask/校准/seed/协议/评分规则不变。
3. 目标CUDA跨轨迹最坏输出NRMSE0.09813%、同actual-state0.05608%，均通过预注册0.5%/0.1%工程界；300 FP64 oracle支持算子计算。CPU prototype原2.42% FAIL保留。full-model teacher-forcing logits最坏6.409%、GDN输出16.254%的差异也保留，因此仅是原工程准入可接受，不是bitwise/80K轨迹等价或普遍精度保证。
4. 同卡同输入快Direct8/16比Replay8分别+30.93%/+33.71%；Direct16比Direct8 +2.12%。完整正式自由生成TPS另见报告，不能当同token因果速度比较。
5. 本次AIME26 seed1 dP8=20/30、dP16=24/30；配对差为+4题，McNemar p0.289，不能称统计显著。dP8两条到长度上限，dP16全EOS；没有低分停止或答案选参。
6. dP16相对dP8多3397632bytes更新buffer容量；短固定输入整模型allocated峰值增加3900416bytes。正式峰值还受不同生成长度/KV影响，不把它归因于周期。更长pending向量会增加读取、准备和临时scratch成本；State写回次数减半不等于全模型写回总耗时已实测减半。
7. 在本次seed1和实现下，dP16以约3.4MB额外buffer容量换来较好正确数与小幅同输入吞吐优势，准确率—速度—State内存权衡更有利；结论限本次数据，不推广到其他模型/seed。
8. 可把本次冻结dP16作为下一版PACE的候选工程基线；它是独立数值轨迹，不自动替换原baseline或宣称官方LeapQuant实现等价。本任务已经封存并停止，不自动追加实验。
