# 历史五文档多 GPU 协议审计

原任务附件最初要求一张 GPU；随后人类明确要求“我希望是使用这个服务器上面的4张GPU一起跑”。

该覆盖原文保存在旧根 HUMAN_4GPU_STEERING.txt，SHA256=0ad20672fb4490a63c299dd163647d78392f130a4da3eb3b38ecd6c187cefec5，并进入原冻结104源清单及最终seal。

原 preregistration 显式记录 GPU_COUNT_OVERRIDE_HUMAN_MESSAGE、四个设备 UUID 和 Doc1/5→GPU0、Doc2→GPU1、Doc3→GPU2、Doc4→GPU3。

本次逐条核查15个正式轨迹 receipt/ENV/ledger，三臂在各自预注册同一物理GPU执行。不存在无证据推定授权或静默追溯改写历史的情况。

原跨GPU128-token canary exact PASS，仅验证该canary范围；未建立32K full-model跨GPU轨迹等价或其数值误差上界。

本新任务明确仅授权一张GPU、总20GPUh、保护停止19.5GPUh；旧四卡许可不延伸到本轮。

本轮新GPU尚未在本审计中选择或执行。若新GPU为历史GPU0且当前数学/软件/runtime身份资格完整，Doc1和Doc5可归A同设备探索层；Doc2/3/4归B跨设备描述层。其他新GPU需按真实UUID逐doc重新分层；身份不全归C，不进入完整交互计算。

A也不表示32K exact repeat或确认性因果交互；B不能与A无区别合并为严格同环境2×2确认。

原五篇Hadamard结果已经被观察，本轮固定原顺序和完整五篇，范围为POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS；项目独立性UNKNOWN。

原FP_NATIVE/FP_H token1 relative-L2失败、旧BrokenPipe及后续授权完成历史、PHYSICAL_PACKED_INT8未资格均永久保留。

本审计只读历史文件，使用原reducer对全部384×2×5位置执行全248320词表FP32 KL和独立标量FP64复核，逐标量精确复现原结果；未做GPU forward、训练或新研究。

历史根：/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1

原最终manifest SHA256：9363b152c991aba56d5696b6db816b48d7940aef63300b24845d7f87b7de2106
