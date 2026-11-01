# Group32资格报告：CPU_CODEC_QUALIFIED；GPU_NOT_AUTHORIZED

本次授权范围内的CPU工作已实际完成，B0/B1/B2均为CPU_CODEC_QUALIFIED。9B GPU、同卡fresh及吞吐全部NOT_RUN。没有启动32K、AIME、训练或重跑FP恢复。

1. **B0是否复现canonical？** 是，CPU真实验证24层×32heads，共12,582,912元素；重建值与字节完全一致、maxabs=0。原7项canonical AST不变。FP32 scale、safe floor、舍入、clip及负零均保留。CUDA标量除法语义另有预登记模拟参考，尚未获得GPU证明。
2. **B1是否通过GPU与真实QDQ？** 尚未；CPU Key轴连续4×32分组、独立参考与写回检查通过。GPU Codec和9B persistent均NOT_RUN。
3. **B2 scale/zp/codes/decode资格？** CPU全部字段通过独立NumPy参考、边界与真实样本检查，UINT8 codes/zp、FP32 scales、int16相减。GPU全部NOT_RUN。张量除法顺序已冻结，遇跨度溢出拒绝，不事后改数学。
4. **统一Runtime及时序？** 已实现同一适配器和H128，不修改FP helper/权重/Attention/MLP；三条件只变codec。小模型真实调用链通过，9B GPU合同待执行。
5. **首token persistent？** 小型真实Qwen CPU从token0执行，FP32/BF16权重下三条件均通过；没有沿用旧首token漏QDQ或整段prompt不量化。9B GPU尚未证明。
6. **下一kernel实际消费？** 两fresh CPU过程共192个合法层写回、168次实际helper initial_state值及字节比较通过；12个故意破坏真实cache负控全部检测。不是hash/指针代替；没有未量化递归shadow。规模仅2个GDN层/2heads/128x128。
7. **同卡fresh？** 同卡GPU NOT_RUN。CPU两独立进程的codec完整输出和真实Qwen类FP32/BF16轨迹完全相同。不能把CPU稳定性升级为GPU稳定性。
8. **INT8 TPS？** B0=NOT_RUN，B1=NOT_RUN，B2=NOT_RUN，显存峰同样NOT_RUN。未创建空壳INT8_THROUGHPUT_RESULTS.csv；没有借用9.064610或13.774973作为当前结果。
9. **Group32风险？** CPU全18类资格和额外13项边界检查通过；极小scale有1e-12安全floor，跨度溢出明确FAIL。全24层尺度元数据B0=384KiB、B1=1.5MiB，B2另加384KiB zero-point；三者codes均12MiB。Sym为保留canonical负零另有12MiB bool flags（临时编码/审计元数据）。实际FP32 recurrent carrier仍48MiB，不是packed INT8，也不据此声称显存加速。clipping/saturation只能作为编码计数解释，GPU发生率、实际峰值与长程误差未知。短资格不证明32K稳定或精度收益。
10. **谁可进入Future-KL？** 当前没有任何条件完成全部9B GPU门禁，三者均不能直接进入。后续方案已准备；必须新授权先完成短GPU资格，再独立授权32K。

CPU证据包括：35项主断言（两独立进程重复）、13项额外边界检查、27固定真实state小样本、独立NumPy逐字段参考、精确Fraction tie检查，以及两fresh真实Qwen类小模型。独立静态/负控复核38项通过。61项当前外部依赖已fresh SHA/bytes核验；实际GPU库/设备门禁没有执行。

| 条件 | CPU | GPU Codec | 9B persistent | 同卡fresh | INT8短Runtime/TPS | 32K准入 |
|---|---|---|---|---|---|---|
| B0 H+C128 Sym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |
| B1 H+C32 Sym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |
| B2 H+C32 Asym | CPU_CODEC_QUALIFIED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 未具备 |

没有NUMERICAL_QUALIFICATION_FAIL、PERSISTENCE_CONTRACT_FAIL或GPU_RESOURCE_FAIL科学事件；合成负控的预期拒绝不是正式GPU失败。GPU尚未授权是NOT_RUN，不是数值失败。整体仍为CPU阶段合法完成、完整GPU资格未完成。

前序finisher当时未封存，随后最终seal与三地核验已完成；本次从实际最终交付核验407源和3753索引，保留两阶段事实，没有修改或删除历史文件/raw。详见源码身份与provenance锚点。

附件D1明确：默认ALLOW_GPU_CANARY=NO，且“此提示词不代表用户已经授权GPU”。本任务尚无另行GPU grant/permit，因此本次GPU查询/Driver/NVML/dispatch/9B加载均0。准备的22有限单位不保证在0.5GPUh内全完成；预算不足必须保留NOT_RUN，不延长或借历史数值填表。
