# 常驻旋转资格任务交接

当前本次合法交付是RESOURCE_BLOCKED_NO_DISPATCH，不是GPU/正式成功。CPU实施、两独立数学和实际GDN接入检查通过；全部GPU项目NOT_RUN，AIME0/60。FINAL_PREFLIGHT_REPORT.md、TASK_RESULT.json和operations真实证据优先于ROOT最初准备阶段字段。

仅GPU1 GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3、GPU2 GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31、GPU3 GPU-30b07aba-a728-7649-4899-e09d33f8bc75；GPU0未授权。P1仍有自己的原UUID队列。仅瞬时NVML空闲不够，原P1该UUID全部10题完成、精确进程退出/释放且无foreignjob后才有可用性。禁止停止/迁移/修改P1及旧任务。

Compute55 zypan@202.38.247.55:22；master /tmp/qwen_gdn_replication_20261013.sock；Python /data/ydai/miniconda3/envs/bitdecode/bin/python。Remote /data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1。55→29 master /tmp/qwen_replication_archive29.sock。凭据不写文件/脚本/命令/输出。

生产适配器src/resident_rotation.py，SHA4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95；一次CPU检查器修正后scripts/qualify_cpu_math.py SHAfd2ad95b4618049ae9b5b1091b8250909217a9d458157f0eb3d019190b4127aa。原CPU首次失败/source/STARTED不重写。两数学4033项及两接入310项的实际回执均在qualification。该CPU证据不能冒充完整9B/CUDA资格。

生产路径直接来自原生Q/K，在原BF16归一化后旋转，再复制头。State常驻UH/UHP，不需要H过渡T；不进入旧H wrapper。S3的P包含于矩阵列排列，QDQ直接B1，不重复P。Core/readout先计算，随后State QDQ，原cache写回不改变readout语义。没有after_token二次QDQ调用。

GPU_PERMIT当前NO；qualify_gpu.py单单位入口在Torch前检查许可，无自动启动器/正式派发器。QUALIFICATION_JOB_MANIFEST58项全部计划，FORMAL_JOB_MANIFEST60条全部未派发；不得凭清单宣称启动。实际GPU单元执行还需完整身份、source/permit/plan绑定、物理UUID、61actual freshSHA、17modelstat、存储/库/引用pair调度和进程释放审查。原run_offline/coordinator/finisher均未复用或重启。

用户允许资源不足时交付READY/RESOURCE_BLOCKED。因此本轮封存本次CPU进展，未增加后台GPU跟进或占卡等待器。现有qwen-oscar-sym32-unified-v2仍只服务原P1合法30和旧C1S3 STOP分支交付，不被本任务删除或改派。未来若明确继续本任务，需重新读取本root、当前P1真实状态和源冻结；已封存本轮资源阻断证据保持只读，不能覆盖或假装GPU已运行。
