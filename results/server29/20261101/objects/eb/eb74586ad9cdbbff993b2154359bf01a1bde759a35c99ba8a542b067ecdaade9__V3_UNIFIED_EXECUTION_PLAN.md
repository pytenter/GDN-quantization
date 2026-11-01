# V3统一执行与实际终态

任务：QWEN_GDN_OSCAR_SYM32_C1_S3_CANONICAL_RECOVERY_AIME26_V3。

本次为用户明确授权的独立统一任务。P1当前执行root、其622源、61实际外部依赖、17模型文件stat，以及固定题目UUID映射仅只读引用。旧GPU0不参与V3。授权GPU1/2/3仍由P1占用时，V3只执行单线程CPU归因，不加载9B模型，不创建CUDA上下文，不采集新的Query，不读取AIME答案选择合同。

Phase A已经执行：保全三类历史失败来源、全部Cq/U/A身份、真实源码与16→32共享映射；对384独立头统一计算M0–M3；独立FP64稠密/秩一递归比较；在CPU提取真实Torch helper并与NumPy FP64算术图比较128步；重新核验已保存的token22首差异。两个CPU工具加载错误及原脚本完整保存，已完成部分未覆盖，独立工具完成原先未执行部分并复核全部结果。没有GPU或科学门限重试。

Phase B的最终必要前提——完整9B功能误差的可信预登记接受范围——未能从当前局部误差预算及不完整历史轨迹建立。不能将4倍H噪声改为略大的常数，也不能以局部CPU一致性取代完整9B。

因此统一终态为 **V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP**。V3新增GPU派发0、C1/S3正式生成0/60。完整128-token真实9B、512 canary、持续INT8 GPU资格、AIME正式及评分均为NOT_RUN。未生成空的科学CSV、空生成JSONL或虚假FP资格报告。

V3_NUMERICAL_QUALIFICATION_CONTRACT.json明确记录结构及局部算术已验证范围和C3未解决的证明义务；它不是GPU许可证。FORMAL_JOB_MANIFEST.json仅为未派发的60条计划，以P1固定同题UUID映射过滤C1/S3，不能执行。HUMAN_GPU_AUTHORIZATION只表示用户的条件性授权，不意味着数学门禁已通过。

原V1、旧RAM失败V2、当前C1/S3 FP失败各保留原STOP。P1自己的30条和旧INT8自己的120条继续由其原协调器处理；本次终止不停止、重启、改写或删除这两个任务的自动跟进。不会再自动创建Recovery04、Recovery05或另一个V3尝试。

所有原始Tensor和矩阵保存在55/29，不下Mac。封存包括原始首差异、当前状态、完整冻结Cq及原U/A、源代码、两次CPU工具错误和真实通过回执。最终清单及独立逐文件SHA/bytes验证是交付依据。
