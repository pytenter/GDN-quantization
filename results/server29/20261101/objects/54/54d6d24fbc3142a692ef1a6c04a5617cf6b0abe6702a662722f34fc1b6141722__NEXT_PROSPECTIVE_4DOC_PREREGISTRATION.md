# 下一轮四篇独立 PG19 文档：前瞻扩展预注册

协议和 CPU 输入身份已准备完成；本轮不执行这些文档的 calibration 或 32K。原两篇正式结论 MIXED 永久保留。

候选池为官方完整 PG19 test 的100篇，来源为[PG19官方仓库](https://github.com/google-deepmind/pg19)。官方对象 generation、MD5、原文 SHA256、冻结 tokenizer 编码、完整 token 与32769-token前缀身份均已实际核验。20篇历史文档被保守排除，另22篇不足32769 tokens，58篇输入资格可用。

完整逐文档身份、排序与排除理由见 PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json；规则见 DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json。CPU输入资格不表示短程分歧配对资格，更不表示已选出四篇。

历史排除：pg19-test-10146, pg19-test-26239, pg19-test-26493, pg19-test-28988, pg19-test-32761, pg19-test-33426, pg19-test-3754, pg19-test-37702, pg19-test-38106, pg19-test-3890, pg19-test-38955, pg19-test-45881, pg19-test-48693, pg19-test-49078, pg19-test-51410, pg19-test-5396, pg19-test-58553, pg19-test-5956, pg19-test-6412, pg19-test-860。该清单绑定真实服务器历史路径登记；未来启动前只核对身份新增，新增历史文档继续排除，不能恢复旧文档。

排序固定为 seed=0，按 SHA256(experiment + '|seed=0|' + document_id + '|' + raw_sha256) 升序，document_id打破相同哈希。历史文档优先占用去重身份；对完整原文、NFC加空白归一化原文、完整tokens和32769前缀tokens作精确哈希去重；归一化文本只用于去重，模型输入保持原文。

按冻结顺序最多筛选40篇输入合格文档，每篇先完成16个旋转的同参考短程评分。自然无分歧配对可在看到任何长程outcome前按预定顺序继续筛选；calibration身份、数值或机械失败则停止整轮。选前四篇信息性文档，所有文档和pair必须一起冻结后才启动任何长程Primary。40篇内不足四篇则零32K启动，保存原因，不放宽gap、不追加第41篇、不把部分文档当完整四篇。

冻结16旋转：DA_fromH, DA_fromL6, H, H_to_L6_a25, H_to_L6_a50, H_to_L7R_a25, H_to_L7R_a50, H_to_L7_a25, H_to_L7_a50, H_to_L8_a25, H_to_L8_a50, IDENTITY, L6, L7, L7R, L8。每项的路径、SHA和layer覆盖见规则JSON；已有L8候选仅作为冻结roster的一员评分，不进行任何新L8训练。

Calibration连续执行0→1152，windows starts=256/1024、H128、18个KDA层、全部16 heads、两窗等权。每个student从自身token0建立INT8-R128状态；教师为同GPU/runtime的匹配FP模型，所有16项共享本次同一短程参考。

P1保留原K3 Query-Readout-aware定义。原实现的aggregate.P1是两个start时点readout_energy的等权均值：post-QDQ native状态误差与原prepared FP query相乘后，对key维求和，再对18层、16 heads、128 value维求平方均值。prepared_query/query_readout及query归一化均直接使用绑定SHA的原helper，不改成P3_H128，也不宣称计算全部未来Query。

L7_FUNCTIONAL_ORIGINAL是每token、每层实际post-o_proj误差平方和除以FP输出平方和（clamp_min=1e-12），再对层、128 tokens、两窗等权。L7_STATE_ORIGINAL是每层每head native状态误差平方和除以FP状态平方和（clamp_min=1e-12），再对head、层、token、窗等权。L7_TOTAL_ORIGINAL保留 functional + 0.1*state。原L7训练已是H128 recurrent-aware；这里复用其原loss helper对匹配full-model部署tensor评分，不冒充重放原FP-conditioned训练采样。

完整16项低分更安全，严格数值相等才midrank并列。pair要求两指标分数差异符号相反，双方midrank gap≥3。按较大的min(gap_P1,gap_L7)优先，再按排序后的科学key pair字典序打破并列。next_prospective_pair_rule.py保留原三个helper的AST，不加其他偏好。

长程顺序：按冻结文档顺序，逐篇FP→P1预测旋转→L7预测旋转；总4 FP+8 student=12条完整32K科学轨迹。逐条件自身0→8192历史与完整checkpoint，blind fresh-process8192→8256恢复及observer ON/OFF必须exact。学生均every-token QDQ、写回、下一kernel消费，保留KDA/MLA/conv/position语义。FP完整32768执行并保存真实FP32 full-V logits及完整身份，CPU/disk streaming单cache串行配对。

Primary固定V=157184，KL(FP||student)，t=8192+64*k、k=0..383，same-step logits无shift，Python sum/384，模型实际执行32768。数值分辨率固定3.3300064494019496e-6；Δ=KL(P1预测旋转)-KL(L7预测旋转)，低于负阈值支持P1，高于正阈值支持L7，其余UNRESOLVED。该阈值仅是原合格contract的经验数值分辨率，不是显著性。新key受影响数值资格必须PASS，不能据新outcome改阈值；新contract不兼容则STOP或另立实验。

Secondary仅原四时点V4 state L2/MSE/norm及冻结量化诊断。E_future、Hidden KL仍NOT_FROZEN_NOT_COMPUTED。不得新增正式功能proxy。

每文档要两student及对应FP全部完整有限、384位置齐全、checkpoint/配对合格才是valid。任何缺失不填零、不按失败算输、不补文档；发生fail-closed后保存已完成证据，后续NOT_RUN。报告计划分母4与实际valid分母，不能声称完整四篇。

独立分析单位是document。主摘要是P1支持/L7支持/UNRESOLVED/MISSING的文档计数与逐文档Δ，不对384相关位置做p-value，也不凭合并平均Δ宣布稳定优势。本轮不做confirmatory p-value。四篇全支持P1也只称有筛选条件的小样本Prospective Extension；原两篇MIXED单列，不能与旧文档混成新的成功率。数据去重保障输入不重复，跨文档独立性仍是外推假设。

更大独立confirmation需要另行预注册、样本量设计与明确授权；本轮不自动进行。交叉面板不能回答P1梯度可优化性、新L8/L9或reasoning改善。本协议也不授权P1梯度、Cayley optimizer step、AIME或free generation。

执行准备：先把现有合格short-calibration adapter的输入路径/CLI白名单移植到新clean worktree，保持上述helper与数学AST，冻结code/runtime/source/旋转SHA并完成受影响GPU门禁。模型与未变历史engine资格可依法复用；此处不把协议和CPU选择函数称为已通过未来GPU执行资格。

资源计划：同GPU6历史16项短程约0.568 GPUh/文档；四篇最早全合格约17.77 GPUh（含正式轨迹和checkpoint资格），最多40篇筛选约38.23 GPUh。估算不设人为GPUh硬上限；实际OOM/无进展/非法cache/IO/数值失败STOP。细项见NEXT_PROSPECTIVE_RESOURCE_PLAN.json。
