# P1 与 L7 完整模型 Future-KL 验证

状态：COMPLETE；总体判定：MIXED。

Primary 是冻结的 384 个时间位置、完整 157184 维词表 FP32 KL(FP||student)，sum/384。采样从8192到32704；模型执行到32768。

|文档|P1 rotation KL|L7 rotation KL|Delta=P1−L7|胜者|
|---|---:|---:|---:|---|
|pg19-test-860|0.135980603178|0.153608581084|-0.0176279779063|P1|
|pg19-test-5956|0.0891569468634|0.0856499057082|0.00350704115517|L7|

预冻结数值分辨率 3.3300064494e-06；abs(Delta) 不超过此值时 UNRESOLVED。它是经验算术分辨率，不是统计显著性。

P1/L7 预测准确性由表中胜者判定。仅两篇文档，不作统计确认或普遍有效主张。

State damage 只使用 V4 原有 per-position L2/MSE，在8192、16383、24575、32767比较；不新增时间聚合 proxy。详细数值保存在 DOCUMENT_VERDICT.json。

pg19-test-860 的 state MSE 方向：t=8192:L7, t=16383:P1, t=24575:P1, t=32767:P1；完整模型 KL 方向：P1。

pg19-test-5956 的 state MSE 方向：t=8192:L7, t=16383:L7, t=24575:P1, t=32767:P1；完整模型 KL 方向：L7。

P1 对 L7 之外的信息：本轮不提供一致的、可推广的 P1 增量信息证据。

P1 gradient trainability：先扩展预注册的 functional 验证和机制分析，不由本轮直接进入训练。

下一步不需要自动进入 L8。L8 是独立训练问题，需另行设计；本轮无 optimizer、Cayley、AIME 或自由生成。

E_future 和 Hidden KL：NOT_FROZEN_NOT_COMPUTED，依据用户本轮澄清；未新增未来 functional proxy。

量化诊断仅记录续跑区间[8192,32768)的原 scale arithmetic、真实 saturation/clipping 和原 J_INJ QDQ 误差；不虚构未保存的前8192 scale历史。V4 未导出独立 layer-wise state-error metric，因此未另造公式。

资源：completed worker 7.746293 GPU-context wall-hours（含启动/加载，精确 teardown 不计）；student完成 4/4，FP reference完成 2/2。

V4 三缓存预估容量 STOP 原样保留。本轮使用单缓存串行执行，并重新验证新增的 checkpoint 存储／恢复接口；原数值引擎未修改。
