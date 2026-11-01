# 五文档前瞻复制：四卡执行计划

任务 QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1。人类后续明确要求使用 .55 四张GPU，覆盖原提示词GPU_COUNT=1；新增总预算仍为四卡累计24 GPUh。

CPU身份：先核验完成Pilot的289项seal、56冻结输入/源码、实际运行依赖、14模型/tokenizer/config文件的当前流式SHA；保留旧FP/H失败、BrokenPipe失败与授权续跑。四份科学源逐字节复用。物理设备断言与UUID映射仅在独立entry控制适配；数学、seed、dtype、TF32、模型、kernel、Hadamard及QDQ次序保持原样。

数据：789唯一context，760长度不足、18历史排除、11合法；在任何GPU前冻结seed20261016的SHA确定性排序，选前5。原始字段/缓存缺可靠repository映射，PROJECT_INDEPENDENCE=UNKNOWN。两进程鲜读source并fresh tokenize核验5篇raw/full/32769/32768及384评分语义；固定manifest SHA为6b73f4df5171e028bbf0138b65d8cd853038cdebfa8f44127d505c38ade1e167。

资格：每卡原128初始canary及独立fresh进程，核验CanonicalC128原参考、DualM all8/ones参考、逐token写回/下token消费、checkpoint64恢复、数值安全及真实设备/环境/显存。四卡128字节一致另核验。资格文档不使用新5篇调试。旧FP/H relativeL2失败不重作硬门禁、不改判。

调度：GPU0依次doc1、doc5；GPU1 doc2；GPU2 doc3；GPU3 doc4。四lane并行，每doc三臂N/A/D顺序且同卡，各自从token0走完整32768。每卡一次只驻留一个worker；不共享模型state、prefix或student历史。

预算：旧真实每condition最慢完整reservation×1.10，共22.287422 GPUh；四卡资格预留1.30h，checkpoint锁排队360GPU秒，退出900GPU秒，计划总23.937422 GPUh。不是已证明上界。真实资格替代预测；并发OFF median若显著慢于旧同条件，预测取max(旧1.10,新旧median比)单向提高；完整32K observed超额再只增不减。所有active卡同时计费、所有加载/IO/等待/释放占用均计入24；不足则真实failclosed，不减少N或改协议。关键路径历史约8.4h，计划约9.3h，CPU审计归档另计。

存储：metadata及全任务一个原子rollingCURRENT在/data，15score与四卡canary scores在/tmp固定注册目录；完整原checkpoint函数由host flock串行，CATALOG保留所有已退役版本身份，只保留最后全局CURRENT，不宣称四lane均可恢复。/tmp仅暂存。永久29目录独立逐文件Python流SHA及sha256sum/bytes核验后封存；旧资产不动。

统计：每doc384点full248320词表FP32 KL，原独立FP64核对与math.fsum/384；主单位doc，N5固定；原2doc只单列。5/5 resolvedD且meanDelta>0为POSITIVE_PROSPECTIVE_REPLICATION，4/5为DIRECTIONALLY_SUPPORTIVE_MIXED；3及以上resolvedA且meanDelta<0为负面；其他混合/未分辨按冻结规则；任何缺失为INCOMPLETE_NOT_CONFIRMATORY。算术界不约束全部32K前向不确定性；UNKNOWN项目独立性不得包装正式显著性。

交付：先释放全部GPU，再CPU独立解析真实score、审计384位置/源/模型/设备/文档/FP参照、按9问生成中文报告，下载所有非大tensor小文件并逐SHA核验。真实失败同样交付实际报告；释放未核验仅封存静态小失败包，tensor记NOT_VERIFIED/POSSIBLY_ACTIVE，绝不对活跃tensor宣称完整seal。完成seal后STOP，不训练、不AIME、不C6R/C7、不DualW、不自动git push或启动下一实验。
