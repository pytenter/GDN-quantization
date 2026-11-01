# P1人工取消：部分科学交付

终态：`HUMAN_CANCELLED_P1_PARTIAL_15_OF_30_C1_S3_TECHNICAL_STOP_ZERO`。P1原计划30条，真实合法完成 **15/30**，其中答对 **11/15**。
两份独立fresh CPU原FrozenV4评分及同一token payload重新decode完全一致。
这是已完成子集的部分结果；**完整30题准确率未定义**，未完成题没有按错误填充，也没有补跑。
C1/S3此前数值资格技术STOP保留，各0条正式生成，不能将其写为0%准确率。

|题号|结果|生成token|结束原因|
|---|---|---:|---|
|aime26_01|正确|10716|eos|
|aime26_03|正确|16863|eos|
|aime26_04|正确|19704|eos|
|aime26_07|正确|20556|eos|
|aime26_08|正确|19315|eos|
|aime26_09|错误|81300|eos|
|aime26_13|错误|63393|eos|
|aime26_14|正确|29074|eos|
|aime26_15|错误|55992|eos|
|aime26_16|正确|13608|eos|
|aime26_18|正确|36547|eos|
|aime26_19|正确|22015|eos|
|aime26_22|正确|20992|eos|
|aime26_25|正确|24250|eos|
|aime26_27|错误|43210|eos|

三条当时运行的样本 `aime26_06_P1_seed1, aime26_10_P1_seed1, aime26_23_P1_seed1` 由人类明确要求停止。
这些部分轨迹及TOKEN_CHUNKS原样保留，不评分、不补EOS、不缩短cap、不续跑、不拼接。
另12条P1从未派发。取消是资源优先级决定，不是按正确率、State误差或答案内容筛选。

P1原622冻结源、旧U/FP资格失败、原许可/dispatch/STARTED/资源账本及所有真实技术失败均只读保留。
本交付不创建原SESSION_RESULT或FINALIZER_STARTED，不调用原finisher。
停止操作中最初的pidfd缺失、job-control干扰和控制工具失败均保留为独立操作证据；不掩盖为科学PASS。

模型State持久载体仍是QDQ后的FP32约48MiB，不是物理packed INT8。
seed1、小N、已观察AIME和不同GPU/运行时间点的限制保留，不能由本15题子集推断PBR整体因果效果。
实际成本使用原同主机monotonic账本和真实停止观测；未完成区间不伪造exit0，EXACT_GPUh仍UNRESOLVED。

本报告只说明CPU部分评分完成。独立29逐文件SHA/bytes、原子NOREPLACE归档和Mac非raw交付验收仍须真实执行后单独给PASS。
