# Ling/KDA 原 MIXED 的时序诊断与最小交叉面板

原正式P1-vs-L7结果永久保持：860支持P1，5956支持L7，Overall=MIXED、1/2。新增面板为事后探索，不是独立前瞻确认。

1. 原384点是否保存逐位置数值？是。旧4条student的实际primary_samples各384个有限KL标量，精确positions、文档、旋转、V=157184完整；重新Python sum/384与原均值逐项一致。不是从aggregate或SHA反推曲线。

2. 860的P1优势集中在哪段？Middle贡献最大。Early meanΔ=-0.0125111464503、Middle=-0.0434400863617、Late=+0.00306729909320；对原384点总Δ的贡献分别-0.00417038215011、-0.0144800287872、+0.00102243303107。

3. 5956的L7优势集中在哪段？Late贡献最大。Early meanΔ=+0.00396383643053、Middle=-0.00680282253450、Late=+0.0133601095695；总Δ贡献分别+0.00132127881018、-0.00226760751150、+0.00445336985649。最大正差值在25600，Δ=+1.17150065303；两篇各自最大的10个正差值均有6个落在Late。逐位置和大差值分布文件保留全部依据，峰值不替代原Primary。

4. State MSE和时间KL排序一致吗？并非总一致。两篇24575/32767时点均P1 state MSE更低，但Late平均KL均L7更低。860的Middle方向与相邻P1 MSE更低相符；5956的四个快照则无法代表区间MSE。对应表明确最近真实KL时点及时间距离，不插值。冻结实现先产生same-step logits，再做当前QDQ写回供后续kernel消费；post-QDQ state快照与same-step输出也是不同观测。上述定义差异有助于说明不能互相冒充，不能由四个快照认定具体因果。

5. 支持早期损伤主导吗？描述上不支持：最大绝对分段贡献分别Middle/Late。支持的有限观察是两篇分段KL符号反转、后期MSE与区间KL排序不一致、不同文档差异集中在不同段。H1的因果早期损伤主导仍NOT_IDENTIFIABLE，H2/H3/H4仅描述性支持。128点分段不套用原384点分辨率。

6. 补齐两条trajectory了吗？是，新增860×L7-a50、5956×L7-a25均完整32768，384评分齐全、finite/INT8写回/next-kernel消费/observer/resume门禁PASS。

7. 原FP能合法复用吗？能。实际387个payload/文档（384个真实FP32 full-V logit位置及原state快照），token/position/shape、每payload SHA、logit内容摘要、model/tokenizer/source/runtime、GPU6 UUID及原KL reducer匹配PASS。新增FP重跑0、旧student重跑0。旧V4三缓存STOP保留，成功资产来自最新COMPLETE串行pipeline。

8. 完整2×3 KL矩阵：

|文档|H_to_L6_a50|H_to_L7_a25|H_to_L7_a50|
|---|---:|---:|---:|
|pg19-test-860|0.135980603178|0.153608581084|0.135745766736|
|pg19-test-5956|0.0891569468634|0.10581077226|0.0856499057082|

9. 探索性交互信号：NO_ORDER_REVERSAL_OBSERVED_TWO_DOCUMENTS。具体六个contrast见DOCUMENT_ROTATION_INTERACTION_ANALYSIS.md。

10. 能解释原MIXED吗？三个相同旋转contrast在两篇中排序均一致；原pair使用不同L7候选可作为原MIXED的候选选择解释线索。但两篇不足以确认稳定候选效应，也不能据此改选旧L7候选。 时间分析另外显示两文档的主贡献段不同、优势沿时间反转。上述观察拆解原结果，但不构成机制因果识别。

11. 哪些解释无法识别？不能由四个state快照确定区间损伤、误差方向/Query敏感性/历史累积的因果路径；不能把两个文档的探索面板外推为稳定文档或旋转效应。E_future、Hidden KL未冻结、未计算。P1是局部K3 readout-aware，未直接算全部未来Query；原L7已H128 recurrent-aware。

12. 下一轮四篇规则：完整官方100篇输入身份已冻，20历史排除、22长度不足、58输入可用；内容及完整/前缀token去重、seed0哈希排序，最多40篇full16 calibration，windows256/1024 H128全部18层16heads、两窗等权，原P1和L7_TOTAL相反方向且双方midrank gap≥3，最大min-gap后lex pair。前四篇信息性文档及pair在全部长程outcome前共同冻结；不足四篇零32K启动，不放宽、不补超额。完整源SHA与规则见下一轮预注册。

13. 本轮实际GPU科学单元：完整新增student=2，正式student已启动=2；旧4 student和2 FP只复用。另有4个自身历史/独立恢复资格worker，不算额外独立科学unit。下一轮新文档GPU、训练、AIME、free generation均0。

14. 实际资源：已自然退出worker合计2.625407746 GPUh（worker launch→natural exit墙钟、含模型加载及GPU上下文退出，非利用率加权kernel时间）；记录的CUDA执行段allocated峰值18087691264 bytes=16.845 GiB，reserved峰值24901582848 bytes=23.191 GiB；NVML进程显存采样峰值23752 MiB。CUDA使用原loop开始reset后的peak统计，NVML为周期采样，并非连续瞬时高水位。GPU6单RTX3090 24GB串行单cache。全部6 worker均自然退出。

15. 有新证据支持P1稳定优势吗？没有可确认的稳定优势证据。原独立两文档仍1胜1负，新的两条仅补事后旋转contrast，没有新增独立文档。不能忽略5956、重选L7候选、合并平均Δ宣布总体P1优势或把384个位置当384样本。

16. 应启动P1 gradient trainability吗？本轮证据不足以据此推进或自动授权。这里没有梯度、优化或新训练实验；梯度可优化性与功能预测排序是不同问题。下一步先按已写好的独立前瞻扩展协议另行授权执行；即使四篇全支持P1，也保留小样本和选择条件限制。

原冻结AIME26背景仍FP44/60、Hadamard35/60、L6 36/60、L7 33/60；本轮未运行AIME。这些背景不作为交叉面板结论。

状态：

- EXISTING_DATA_DIAGNOSIS_STATUS = COMPLETE_SAVED384_SCALARS_AVAILABLE
- TEMPORAL_FUNCTIONAL_DIAGNOSIS_STATUS = COMPLETE_CPU_POSTHOC_DESCRIPTIVE_CAUSAL_PATH_NOT_IDENTIFIABLE
- CROSS_ROTATION_PANEL_STATUS = COMPLETE_POSTHOC_EXPLORATORY_CROSS_PANEL
- DOCUMENT_ROTATION_INTERACTION_STATUS = NO_ORDER_REVERSAL_OBSERVED_TWO_DOCUMENTS
- NEXT_PROSPECTIVE_READINESS = PREREGISTRATION_FULL100_INPUT_IDENTITIES_AND_RULES_READY_NOT_AUTHORIZED_TO_RUN
- OVERALL_SCIENTIFIC_VERDICT = ORIGINAL_FORMAL_MIXED_PRESERVED_NO_STABLE_P1_ADVANTAGE_CONFIRMED

本轮完成后停止。未自动执行下一轮四篇、L8/L9新训练、P1梯度优化、Cayley optimizer、AIME或free generation。
