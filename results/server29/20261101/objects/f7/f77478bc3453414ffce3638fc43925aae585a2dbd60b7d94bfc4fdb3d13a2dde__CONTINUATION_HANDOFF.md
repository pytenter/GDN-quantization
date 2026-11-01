# dP32 吞吐专项延续规则

完整授权见同目录 HUMAN_REQUEST.txt。唯一 local 为 `/Users/pytenter/Documents/临时项目1/PACE48_LQUNROLL_DP32_THROUGHPUT_V1/releases/ASSET_FIX_V2`；55 为 `/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1/releases/ASSET_FIX_V2`。只使用原 UUID GPU-30b07aba-a728-7649-4899-e09d33f8bc75 对应物理 GPU3，单 CVD→cuda0、TP1。旧 AIME dP8/dP16 和可能的 PureINT4 属于其他执行，禁止停止、迁移、改源、改输出、改优先级或重派。只有实际空闲才启动本测试；禁止抢占。

先读 CONFIG.json、SOURCE_IDENTITY.json、SOURCE_MANIFEST.json、runtime/UNIQUE_PARENT_STARTED.json 和 _local/monitoring 最新状态。不要重复启动 parent 或任何 GPU 单元。脚本顺序为两 fresh dP32 原门禁 CUDA 验证→同原 kernel 诊断→同卡三配置短基准三轮→同固定输入流16K长区间三轮→CPU报告。全部脚本 source 冻结后不热改。若科学门禁失败，不启动性能测量；CPU/包装故障保留真实首因，不能冒充数值或硬件失败。

所有模型、codec、mask、K4/norm、FP32 u/log-decay、Triton kernel 与原固定工作量函数保持原字节；只扩展独立副本构造器的 period 参数准入。新脚本是共同测量/诊断控制，不给 dP32 加优化。所有配置使用相同原非AIME16-token固定输入、sampler/checks、BF16、32 warmup。短测128步原 fixed_run，长测真实逐tokendecode至16384，区间2048–8192、8192–16384；每配置3轮同卡轮换，独立请求cache。长测试需要数小时，无人为总wall/GPUh停止预算。1800只用于真实phase不进展，120只用于own exit/release。不得追加AIME、评分、dP64、PureINT4、Compensator、Smoothing、调参或优化。

唯一前台 bounded helper：`env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B <local>/operations/monitor_dp32.py --deliver-if-ready`。先检查真实 local reader、`_local/READER.lock`、`_local/independent_delivery/CLAIM.json`、FIRST_FAILURE.json 和 RESULT.json；不能重复活跃 reader/已claim审核传输。可能进行终态CPU审核/下载，不设外层短 timeout，工具 yield 等真实进程。审核失败保留 claim/firstfailure，不自动重复。

终态 helper 要求原parent与Bash实际退出、真实Bash wait整数0、完整parent CPU报告，实际GPUowning整数退出与release收据；失败数值测试的非0是合法诊断工件，不改写成0。CPU self-test的exit0/7/143在冻结前完成，receipt不伪绑未来SOURCE，独立审核单独认定其0GPU scope。独立 fresh source、模型17、全部实际regular mapped文件SHA检查后，封存55 MANIFEST，逐文本SHA交付到Mac。不得下载Tensor/model/env/compiledbinary。

编译/磁盘cache加载wall单独记录，不能称纯编译时间；Kernel内部base/delta/pending融合，无独立排他时间则明确UNMEASURED，不改kernel强行拆分。边界诊断的stream elapsed可能含host空隙，不能称SM active；allocator reserved、live buffer、容量和FP32/FP64临时scratch分开。长测试其他GPU任务正常运行，温度/功耗/时钟/利用率telemetry是环境观察，不把NVML利用率当occupancy。

健康推进静默；不重复开始通知。只在真实故障、需人操作或最终完整交付时通知。完成全部报告、真实退出/释放和55/Mac SHA封存后，删除本任务自己的跟进；保留 pace-lqunroll 和其他任务，不归档聊天。最终严格使用用户要求的 COMPLETE / COMPLETE_NO_CLEAR_SPEEDUP / STOP_NUMERICAL_VALIDATION_FAILED / STOP_NO_IDLE_GPU / STOP_RUNTIME_BLOCKED，并中文回答速度/写回/缓存/内存/后续是否值得测准确率，不能自动启动准确率评估。


V2仅修复独立打包漏带的冻结资产AIME26_DATASET_FREEZE.json和CPU报告glob误读.compile.json。V1 SOURCE0237保留，原parent/测速worker/report/Bash实际1，旧local交付claim和failure保留不重试。两fresh CUDA gates实际0/release且数值源/输入/kernel在V2完全同SHA，经过独立CPU VALIDATION_REUSE_ADMISSION逐SHA/真实owning验证后复用资格，不重跑门禁、不复用任何benchmark轨迹。V2测速仍全新同卡同输入三配置三轮，原数学与kernel字节不改。
