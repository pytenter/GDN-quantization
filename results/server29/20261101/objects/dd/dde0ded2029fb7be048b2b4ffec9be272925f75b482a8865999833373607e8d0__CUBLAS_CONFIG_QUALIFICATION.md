判定：FAST_CANDIDATE_NUMERICAL_FAIL；执行终态FAIL_CLOSED_STOP。B0在四卡各两个fresh完整256均逐字节相同。B1在GPU3首次fresh第0个输入已失败，其他三卡在模型加载或PREFLIGHT阶段共享终止，没有已提交数值比较；不是四卡都观察到B1差异。

双方同token、next-input、cache cursor、codec、GDN层表与234项dtype/shape/stride/offset/bytes roster均相同。CPU从29只读流式重查原始数据：231/234 Tensor实际不同，所有468侧Tensor有限且逐Tensor SHA匹配；64项非Tensor metadata差异为下游cache哈希。首个保存边界GDN/0/q的第0元素由0.043701171875(333d)变为0.044189453125(353d)，不能把该边界当成已定位的内部首个算术kernel。该q最大绝对差0.03125；state最大0.08163261413574219，logits最大0.53125。完整每Tensor maxabs/maxrel在数值审计JSON，relative分母沿用max(abs(reference),1e-300)，近零参考可产生极大相对值，并非nonfinite。

| 当前资格 | B0 | B1 |
|---|---|---|
| 完整FP轨迹 | 每卡2fresh×256 PASS | GPU3 token0失败；另3卡共享中止 |
| fresh确定性 | 同卡逐字节PASS | 未建立 |
| logits/state逐字节等价 | 自基线 | GPU3 FAIL |
| 单卡TPS/峰值/新Profile | NOT_RUN | NOT_RUN |
| H+C128、H+Dual-M | 未测试当前候选适用性 | NOT_QUALIFIED_FOR_THIS_CODEC |

数值优先门禁未放宽，未进入性能、Profiler、H或D3；不存在提速率或负吞吐结论，也没有稳定新数值协议的资格。内部GEMM/库kernel机制未定位。
