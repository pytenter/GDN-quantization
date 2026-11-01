# 常驻坐标实现与资格计划

这是独立的新任务；原V1/V2/V3终态及P0/P1源码、动态库、队列、答案和封存均不改变。此目录源码和remote-only矩阵是独立副本，不使用指向旧工作目录的可写链接。

原生入口：QKV投影→原Conv/SiLU→Q/K16头、V32头→原BF16 L2Norm一次→FP32 Q/K批处理旋转→repeat_interleave2→原Torch GDN helper（关闭其重复归一化）→原helper执行Query缩放一次→原cache保存旋转坐标State→直接原B1 Sym32 QDQ/writeback。首prompt单token仍走原chunk helper；后续单token走recurrent。Gate/Beta/Value保持原图和dtype，不旋转Value轴。

C1的行向量接口是q@A、k@A，其中A=U64@H64后一次转FP32；与列向量A.T@q一致。S3为q@A后Key坐标P排列，State持久坐标是P.T@A.T@S_native。P在初始化/更新坐标定义内，只应用于Q/K准备及离线矩阵；S3直接使用与C1同一B1 Codec，不在QDQ中再排列或逆排列。初态零自动处于任何正交坐标，无运行时全State初始化变换。生产路径禁止State矩阵乘回Native/H。

旧H wrapper不进入此路径，原生Q/K未做Hadamard，因此直接A路径不需要T。H基线通过同一准备接口q@H；不重复H。每token每层QK两类向量共16uniquehead的批处理旋转，在复制前完成；矩阵构建、资产读取与head索引不进入token循环。每层计数、归一化标记、helper原函数身份、Q scaling及forward改写AST受结构门禁约束。

CPU阶段：全部384头U/A矩阵身份、FP64正交与递归/readout、P合法性、同U共享、实际BF16归一化与FP32准备、原Torch helper局部128步、真实B1 CPU编码/解码/写回消费、缓存坐标及fresh-reset、编译与AST变更范围。固定输入FP64参考的舍入预算按操作数量及数据尺度推导。不同完整模型浮点图的误差只记录，不使用4H或2e-6硬门禁。CPU通过不代替CUDA、9B或81K证明。

GPU准入前：核CVD实际UUID/ownerPID/command/starttick、P1所有权与目标卡空闲、model17stat、实际61外部freshSHA、冻结源码/矩阵SHA和库身份。先前GPU0释放不等于获得本任务许可；仅1/2/3可用。无授权卡空闲时以RESOURCE_BLOCKED_NO_DISPATCH交付本次CPU进度，不创建后台GPU竞争者或占卡等待实例。

GPU单元顺序：原CPU/GPU B1精确字段算术；128固定输入NATIVE/H/C1/S3无QDQ9B诊断；C1/S3各32和128逐tokenQDQ；各512两fresh；原Sampler/EOS/RNG独立非AIME free64两fresh；4096持续QDQ与稀疏完整值/byte next-consumption；活模型82293slotKV/nativeSDPA资源压力。每个完整sample独立process/cache/RNG。重型全值参考R0/R1 pair全局串行，预留80GiB参考+20GiB物理余量；不使用旧参考或拼接轨迹。故障保留首差异、精确退出和释放证据。

性能单元：在同一已释放UUID上独立freshprocess比较H+原Sym32/C1/S3；同一冻结doc05，固定token逐步输入，warmup128，短窗128，4096后段128；另记录prompt计时与原Sampler的非AIME free生成decode计时。torch.profiler记录CUDA kernel次数，record_function/事件分别QK、GDN、QDQ；单次同步在窗口边界，不做每head计时同步。包含原Codec安全检查的总TPS和显存必须同时报告，不能用事件和替代端到端墙时。成本用各UUID实测同任务TPS，列8k/20k/40k/81920条件场景，不保证81K速度。

性能通过要求实际完成全部可复核测量、无State正逆matmul/重复P/重复归一化/重复32头旋转、无未解释同步或显存不可执行；不事后编造最小TPS阈值或在无预算约束时宣称某个小时数失败。尚未实测一律NOT_RUN，不能声称高效已证实。

只有全部真实CPU/GPU/QDQ/资源/性能门禁PASS并退出释放后，才生成正式ALLOW_GPU许可及60条一次性dispatch；计划manifest不是派发。题目/seed/cap/原Sampler/FrozenV4固定；同题C1/S3用P1冻结UUID，卡内串行，三卡不同题并行。无GPUh/总wall/单条elapsed硬上限；1800秒仅真实phase/token无进展watchdog，120秒仅自身释放观察。

正式完成后双独立CPU FrozenV4+同payloadtoken重decode。P1/P0只读引用合法已有封存；不存在完整配对时明确commonN/PARTIAL，不补跑。I=(S3-C1)-(P1-P0)只作探索性描述；跨UUID、runtime、小N单seed和历史已观察测试限制保留。性能及数学局部证据不冒充最终准确率或81K数值界。
