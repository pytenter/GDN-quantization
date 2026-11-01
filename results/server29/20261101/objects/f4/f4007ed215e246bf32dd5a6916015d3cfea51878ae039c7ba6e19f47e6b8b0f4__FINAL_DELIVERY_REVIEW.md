# Ling/KDA Clean-Room 最终交付复核

本轮在模型启动前因真实资源冲突终止：**FAIL_CLOSED_STOP**。没有产生新的32K科学结果。此复核补充原封存报告，原报告、STOP及封存manifest保持原字节；原报告中描述的canary、parent、正式分段恢复均为预定方案，实际未执行。

首STOP北京时间：2026-10-15T13:59:38.503662+08:00。首次空闲发现时GPU0/3通过，正式worker启动前再次采样时GPU0已被外部compute占用。GPU0 UUID为GPU-0c785495-3462-4b7f-5ee1-0d3ab8fe53bc，两个进程PID2360738和2360739均为UID1003、start_ticks44237269。三个正式样本显存均17715MiB，SMI利用率依次0%、100%、100%；NVML直接进程列表也记录两个compute进程。该失败属于真实资源占用，不属于1%/0%低利用率采样差异。GPU3的三个样本均15MiB、0%，无compute进程，单卡准入PASS，但全局依赖STOP阻止其模型启动。未触碰外部进程。

实际GPU worker启动数0，新增GPUh **0.0/36.0**；模型加载/8-token canary **NOT_RUN**；数值资格 **0/12**；正式FP **0/4**，student **0/8**，有效文档 **0/4**。四篇41603/2544/7987/26183的K_P1、K_L7、Delta和胜负全部NOT_RUN。P1胜0、L7胜0、数值无法分辨0、缺失4；均值和中位Delta为null。未测CUDA或自有模型NVML峰值，均为null。0GPUh由无任何GPU worker launch和空账本共同证明，未把未知峰值写成0。

CPU控制实现及15项测试通过；原25项源码身份和干净工作树再次核验通过，commit `d03bd1b1122e93f89c5bc06b212b0b52b4974eed`。模型、输入、旋转、预注册与冻结FLA JSON的CPU身份PASS；实际模型FLA/kernel运行profile未执行，不能声称G1、数值资格或完整工程运行通过。历史26183 FP8192 checkpoint只有本轮SHA注册，尚未完成本轮fresh64，因此不计作合格科学轨迹。

原始完整证据见 `../FINAL_SCIENTIFIC_REPORT.md`、`../final_scientific_results.json`、`../qualification_matrix.json`、`../gpu_resource_ledger.json`、`../SOURCE_DIFF.patch` 和本目录JSON。原封存90文件与含manifest自身的91条SHA全部再次核验。另用本目录交付manifest绑定原SHA清单、原manifest、稳定controller.log及本复核；大型checkpoint/模型不下载。

科研问题的回答：

1. P1在四篇上是否更好：**尚无答案**。四篇均未执行，不能以资源失败判断selector效果。
2. 与旧两篇MIXED 1:1的关系：旧结果和posthoc身份永久保留。本轮没有新增证据，不合并为n=6。
3. 增量价值是否值得继续：本轮未测到增量价值，既不支持也不否定P1；不能据此作方向性胜负判断。
4. 是否停止当前selector路线：当前证据不足以作科学否定。本次执行按用户第10/13节资源安全与单次尝试边界结束，不能用工程停止替代科研否定，也没有新训练授权。
5. 如果支持P1，下一步如何验证：本轮没有满足该前提；一般只有取得合法完整四篇数据后，才可讨论预注册的独立样本验证。此处仅是条件性建议，未启动任何后续实验。

`MIXED_OR_INCONCLUSIVE`在本轮仅表示**缺失导致无法判断**，不是观察到新的混合科学胜负。文档才是统计单位；未计算384点评分、均值Delta或确认性p值。

所有自有GPU worker本来就未启动，CPU控制器PID2371103已退出、tmux不存在。旧execution01至05及所有历史STOP未改。本任务终止；不迁移设备、不自动开启第二次Clean-Room、不训练或启动其他评测。
