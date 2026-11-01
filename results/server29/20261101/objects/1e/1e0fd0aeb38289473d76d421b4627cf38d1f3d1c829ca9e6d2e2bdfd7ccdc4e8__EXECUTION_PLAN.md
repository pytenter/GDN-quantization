本任务恢复历史 OLD 的执行身份及完整 Python 路径，建立隔离候选和最小未来 GPU 资格程序。当前 ALLOW_GPU_RESTORE=NO；本轮只完成 CPU 工作并封存。以前的四卡、无时间上限授权不用于本任务。

13.742036038730332 TPS 的目标来源是历史调和任务 OLD_FP 三次固定窗口中位数，不是原始 AIME 自由生成速度，也不是 INT8 速度。历史 H+C128 中位数为 9.064609924852187 TPS。本轮不重复已完成的广泛根因调查、不搜索开关或线程、不开发 CUDA kernel。

恢复模式为 HISTORICAL_PORT：原 runner 9ff56d…、架构 90d929…、matched worker 08b628… 均逐字节复制；新增适配仅是隔离路径、外部依赖绑定和计时外身份检查。历史完整动态库/driver/物理 dispatch 身份缺失，不宣称历史二进制 EXACT。当前依赖只读复用，不覆盖或安装正式环境。

CPU 阶段依次核验来源、权重及小依赖身份、OLD/NEW 协议差异，建立 case-safe 原路径 SHA 分目录副本；运行真实 Qwen 类的 tiny FP32/BF16 CPU 检查、观察器/缓存回归、授权/资源控制合成测试；最后封定源码、计划和交付。所有历史文件、seal 和 GPU 账本只读；CPU 开发工具错误和修正保留。

未来 GPU 计划仅为预登记，不是授权：单 GPU0 / RTX3090 / UUID GPU-a78176ed-f97f-c54f-6201-5a01b570f422。硬预算 1800 秒=0.5 GPUh，保护 1650 秒，清理预留150秒，单 dispatch 上限240秒，清理观察120秒。任何派发前必须有新的明确人类授权、独立授权 JSON、ROOT 开关 YES、冻结文件和模型/依赖当前 SHA、显卡独占、至少22GiB显存与10GiB磁盘可用。不能因存在四张卡自动并行。

固定顺序为两次独立 fresh OLD FP 数值进程，各32个原固定输入前缀；通过后优先三次 OLD_FP 128 warmup+128 measured，无 Profiler、无计时窗逐token导出/hash；预算许可时最后补一次 NEW_FP 参考。每次派发预留240+150秒。最少两次数值进程不能因预算缺少而跳过；性能重复只能按冻结剩余预算减少并降级报告，不改任何参数。32-token数值资格与256-input性能窗、32K/81K及INT8资格分别标记。

数值预登记包括完整32层/24 GDN、FP32 state、全部8路KV/cursor、逐层状态写回和下一helper消费、完整dict/list/tuple Cache+metadata+alias结构、finite、state存储区间不重叠、同卡两fresh逐字节重复、当前实际critical library同阶段选择签名一致。FP64 state oracle从token0沿实际q/k/v/g/b激活递推，native normalization另作有界检查；不是全模型FP64或长程误差上界。阈值在资格计划 math_gate 中事前固定，不能根据GPU结果调整。

首次身份、资源、内存安全或数值失败保存原始首差异并 FAIL_CLOSED_STOP；不热修、隐式重试、换卡或重复前缀。进程退出和 NVML 释放单独观测，资源下界用最后确认存活时间，上界用首次确认释放时间，物理精确GPUh不能由轮询伪造。未获GPU授权本轮不查询NVML、不启动任何恢复/性能单位。

未来所有原始tensor留远端；两次数值raw仅GDN部分已超过3GiB，总cap8GiB。删除新tensor前必须29独立逐文件SHA/bytes+原子封存；历史资产不删。当前没有产生新GPU raw。

INT8本轮仅静态工程/科学协议审计，不运行H、Dual-M、Group32、AIME、32K Future-KL或训练。即使未来FP短窗通过，也不能直接宣称INT8或长程科研已资格。最终CPU状态为 CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED。
