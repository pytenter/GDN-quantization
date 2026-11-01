# 本轮执行计划（CPU门禁前草案）

TASK_ID: QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1

1. 仅CPU核验历史五doc、模型/runtime、完整FP参考logits及H两臂，以原冻结FP32/独立FP64 reducer逐位置重算；任一缺失、身份或复算失败即FAIL_CLOSED_STOP，不重跑FP或缩样本。
2. 核验确切冻结C6R export/manifest/SHA、Key侧坐标、CanonicalC128及Dual-M数学复用边界。不能在当前已合格递归路径中最小接入即STOP；不重写kernel、不改变codec或训练矩阵。
3. 冻结五doc原顺序、384位置、模型/runtime、单一物理GPU0 UUID、canary计划、预算预测公式、I/O归档和全部源码后，再做最短GPU资格。两codec初始与fresh进程128token；任何FAIL保留首失败不重试。128不能充当32K forward误差界。
4. 全门禁PASS且按实际C6R canary保守预算可完成10条轨迹时，单GPU顺序doc1..5，C6R_C128后C6R_DUAL_M，各token0..32767独立历史。GPU计时涵盖初始化、checkpoint、IO及GPU释放前全部占用。protective19.5 GPUh、硬上限20 GPUh。
5. 离线tmux托管、单协调器、不自动重启。资源/数值/身份失败有界停止。保留全部真实部分进展。历史资产不变，新增临时大资产若归档后移除需先远端独立SHA/bytes封存并记录永久位置。
6. 全部或真实STOP后CPU审计Primary、2x2、A/B/C设备分层、三窗口、LOO/去Doc1描述敏感性、真实系统风险；输出13问中文报告及状态、SHA封存、小文件本地交付、核验GPU释放。未运行科学结果标NOT_RUN，不生成空壳科学CSV。

本次是POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS，N固定5，PROJECT_INDEPENDENCE=UNKNOWN。原2doc仅背景。Primary只有Delta_C6R；与Interaction及LearnedGain_DualM严格区分。历史FP/H token1失败和physical packed未资格永久保留。禁止新研究、AIME、C8、Dual-W、scale干预及push。
