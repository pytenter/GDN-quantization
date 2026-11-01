# QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1 最终科学报告

{
  "ENGINEERING_STATUS": "PASS",
  "NUMERICAL_QUALIFICATION_STATUS": "PASS",
  "PHYSICAL_RESOURCE_STATUS": "PASS_FULL32768_CONFIRMED_WITH_VERIFIED_REMOTE_ARCHIVE",
  "FORMAL_8DOC_STATUS": "COMPLETE",
  "REPLICATION_STATISTICAL_STATUS": "DIRECTIONAL_REPLICATION_SUPPORTED_IN_8DOC_PANEL",
  "FUNCTIONAL_CAUSAL_REPLICATION_VERDICT": "REPLICATION_DIRECTIONALLY_SUPPORTED",
  "NEXT_RECOMMENDED_EXPERIMENT": "QWEN_GDN_TEMPORAL_READOUT_RISK_CAUSAL_DIAGNOSIS_V1_RECOMMENDED_ONLY_NOT_EXECUTED"
}

本轮在既有兼容RepoBench-P / RepoBench-P_e缓存上冻结8篇新文本/token去重文档。所有文档先固定再运行，历史2-doc不进入新统计。原数据没有权威repository/project字段；跨项目独立性无法核实，符号检验与bootstrap区间的解释均依赖独立性假设。

## 20项回答

1. 历史时间收益：Doc1 [8192,16384) 平均改善=-0.000701210584503，对总体Delta贡献=-0.000233736861501; Doc1 [16384,24576) 平均改善=+0.0116502309788，对总体Delta贡献=+0.00388341032627; Doc1 [24576,32768) 平均改善=+0.00456643136905，对总体Delta贡献=+0.00152214378968; Doc2 [8192,16384) 平均改善=+0.160236185023，对总体Delta贡献=+0.0534120616745; Doc2 [16384,24576) 平均改善=+0.164906646663，对总体Delta贡献=+0.054968882221; Doc2 [24576,32768) 平均改善=+0.0429340679635，对总体Delta贡献=+0.0143113559878。Doc1第一窗口略恶化，后两窗口改善；Doc2三窗口改善。POSTHOC_DESCRIPTIVE_ANALYSIS。
2. outcome-blind：是。789个不同原文候选、21个长度合格新候选在任何新GPU outcome前冻结；seed20261013与SHA排序，未使用KL/G/激活/收益。额外缓存发现造成的执行前暂停及独立审计记录完整保留。
3. 长度/独立性：冻结8篇均提供32769原context tokens，排除真实历史registry 6个本域已用ID及150个raw SHA，并通过冻结文本/token近重复检查。repository/project身份UNKNOWN，不能宣称已证实8个独立项目。
4. 数值合同：PASS；原compensated scan、量化器、数学engine、全词表KL/reducer源文件不改；增量资格两新observer进程和8192旧anchor新进程next8逐字节一致；分辨率0.000690448106979602。
5. RTX3090完整轨迹：全部完成。
6. FP references：8/8 有完成回执。
7. 共同8192 anchors：8/8。
8. 学生完整continuations：16/16。
9. Parent Future-KL：见下面所有实际成对文档逐项数值与DOCUMENT_LEVEL_PAIRED_EFFECT.csv。
10. NoInflation Future-KL：同表；未测单位无虚构值。
11. signed Delta/relative gain：同表；Parent非正或非finite时UNDEFINED。
12. 可分辨改善：8/8；负向0/8；UNRESOLVED 0/8。
13. exact sign test：{"one_sided_p": 0.00390625, "n": 8, "positive": 8, "interpretation": "primary N8 test only noUNRESOLVED; otherwise resolved-only descriptive sign statistic, full8 denominator retained", "independence_assumption_verified_by_repository_metadata": false}
14. 反例/弱激活：{"negative_documents": [], "weak_activation_documents": []}；弱激活真实保留，不更换或删除。
15. 时间一致性：所有8篇三窗口见TEMPORAL_EFFECT_BY_DOCUMENT.csv；不将窗口或384个位置作为独立样本，不根据窗口结果修改Primary。
16. clipping与改善共存：min scale会收紧范围，从而增加source overflow或饱和；长期KL反映整个独立闭环传播的总效应，两者不矛盾，也未证明clipping是唯一中介。实际新文档触发/overflow/saturation见下表及INTERVENTION_ACTIVATION.csv。历史NoInf两doc触发率22.6853%/21.9178%，source overflow约0.7022%/0.7285%，Parent约0.0341%/0.0341%；同时总体KL下降。
17. 本轮实际GPUh=42.1481282347，peak allocated=19343843328 bytes，reserved=25056772096 bytes；已验证远端保留=90647764785 bytes；当前本地文件=54980128 bytes（封存前）。复制的旧资格GPUh不重复计入。cache/free VRAM/wall-clock见原单位回执、进度及GPU_COST_LEDGER。
18. 独立复制：REPLICATION_DIRECTIONALLY_SUPPORTED；只解释本模型/C6R/已有代码数据域/oracle干预/冻结协议，不是部署方法或全GDN模型结论。
19. 下一步：QWEN_GDN_TEMPORAL_READOUT_RISK_CAUSAL_DIAGNOSIS_V1_RECOMMENDED_ONLY_NOT_EXECUTED。仅建议，本次不会启动新实验；历史已存在状态与功能不同步及时间异质性，不能绕过新8-doc统计。
20. C8训练证据：不充分，NEW_C8_TRAINING=NOT_AUTHORIZED，未执行。

|Doc|Parent KL|NoInflation KL|Delta|RelativeGain|方向|触发率|Parent overflow|NoInf overflow|
|---|---:|---:|---:|---:|---|---:|---:|---:|
|1|0.0502448094723|0.0265721231173|+0.0236726863551|47.114690%|POSITIVE|21.978401%|0.034063%|0.689229%|
|2|0.0134690609391|0.00653683641909|+0.00693222451996|51.467764%|POSITIVE|20.667866%|0.034116%|0.726361%|
|3|0.0315084636742|0.0233828917589|+0.0081255719153|25.788537%|POSITIVE|21.624148%|0.034093%|0.635317%|
|4|0.057022846778|0.0328253230293|+0.0241975237487|42.434787%|POSITIVE|23.604223%|0.034046%|0.684610%|
|5|0.0946657507333|0.0667228170984|+0.0279429336349|29.517469%|POSITIVE|22.490806%|0.034085%|0.603366%|
|6|0.0131035470335|0.0105229993259|+0.00258054770762|19.693505%|POSITIVE|20.989326%|0.034084%|0.632025%|
|7|0.0235218670475|0.00935097188242|+0.014170895165|60.245622%|POSITIVE|21.328222%|0.034091%|0.557272%|
|8|0.0440167231088|0.0243230432017|+0.0196936799071|44.741359%|POSITIVE|22.701709%|0.034067%|0.598898%|

文档bootstrap区间与小N/项目相关性限制：{"method": "100000 whole-document bootstrap mean Delta, percentile linear interpolation", "seed": 20261013, "confidence": 0.95, "lower": 0.009820203126923666, "upper": 0.021837493752987873, "limits": "smallN; conditional on representative independent documents, within-project dependence unknown; diagnostic oracle/data/model/rotation scope only"}

科学决策：REPLICATION_DIRECTIONALLY_SUPPORTED。NoInflation为clean-FP oracle诊断干预，没有验证可部署scale cap、S1/S2训练Loss、旋转排名或AIME改善，也没有运行新NoDeflation以复制其方向特异性。
稀疏状态诊断仅16384/24576/32768三个预冻结终点；state-MSE不替代KL。所有大tensor保存在202.38.247.29的逐doc封存目录，位置映射及SHA见archive/DOC_*_CURRENT_ASSET_LOCATIONS.json。旧正式资产未删除或修改。
第一失败：null

