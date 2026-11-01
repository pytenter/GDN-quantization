# cuBLAS Workspace 资格：CPU 准备完成，GPU 未授权

当前实际结论：**CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED**。B0/B1 的真实 logits、state、GDN 中间量、Codec、fresh determinism、TPS、显存与资源准入均为 **NOT_RUN**。本文件与可执行脚本不构成 GPU 授权，也不表明真实 GPU 工作负载已 PASS。

## 范围与唯一变量

- B0=`CUBLAS_WORKSPACE_CONFIG=:4096:8`；B1=`:16:8`。
- 两者均保留 global deterministic=True、fill=True、TF32=False、原 BF16 模型和 FP32 state、intraop/interop=1、OMP/MKL=1、原 compensated chunk/原 recurrent 数学、全部 safety、原 QDQ/writeback 时机、固定非 AIME token 与 seed20261015。
- 环境在 fresh 子进程创建时传入。worker 在任何 torch/model 导入之前验证启动变量、授权、完整源/权重/fixture 身份和控制器 dispatch。Python版本与 NVIDIA_TF32_OVERRIDE、TORCH_ALLOW_TF32_CUBLAS_OVERRIDE、CUDA_LAUNCH_BLOCKING 三个已知强制开关也在 engine 导入前检查；禁止值1并记录环境。原 core 顶层硬编码 `:4096:8` 仅在内存 AST 副本中改变一处字符串常量；反向恢复后 AST 必须与原源码一致，`configure`/数学函数不改。
- 不关闭 fill、不关闭 deterministic、不改线程/TF32、不增加 workspace 搜索。这次只准备同一 cuBLAS 假设，fill GPU 优化不在此授权或矩阵中。

原 core SHA 为 `aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575`；原 baseline worker SHA 为 `0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41`。source/输入/环境的完整绑定由 `analysis/STRICT_BASELINE_CONTRACT.json` 与新 source manifest 冻结。历史吞吐值仍只是对应原任务的参考，不能冒充本轮 B0 新测量。

## 已实现的入口

|文件|实际职责|
|---|---|
|`scripts/cublas_worker.py`|一次被准入的 fresh 单位；默认 NO 拒绝；数值、裸性能及短栈 profile 模式|
|`scripts/cublas_harness.py`|纯 CPU 的固定 schedule、fresh argv/env、组级数值与性能 verdict；不自行 dispatch|
|`scripts/cublas_compare.py`|纯 CPU 实际 raw-byte 比较、完整快照身份核验、首差数值分析、性能归约|
|`scripts/test_cublas_cpu_controls.py`|合成标量/文件/AST/门禁测试；不调用模型或 CUDA|
|`scripts/run_cublas_controller.py`|独立协调器负责未来授权、预算、单卡、精确进程与退出/释放观察；本轮未运行 GPU|

worker CLI 为 `--root --authorization --unit-id --mode qualify|benchmark|profile --condition B0|B1 --codec FP|H_C128`；repeat 只由固定 unit 身份导出，不能用 CLI 改动。直接调用 worker 不能绕过控制器：要求 isolated run copy、正确 future root、immutable DISPATCH SHA、直接父进程 PID/starttick/cmdline 与冻结 controller 相符、auth/source/budget 绑定，以及匹配的前置 verdict。执行源/启动身份/数值失败不重启单位。

## 数值先行：四条 fresh 256-input 轨迹

固定顺序：`B0_FP_Q0`、`B0_FP_Q1`、`B1_FP_Q0`、`B1_FP_Q1`。每条从 token0 独立开始，全部 256 个输入，不只测量后128。比较：

1. 实际 input tensor、当前 input id、下一固定输入 id。
2. 每个 GDN helper 实际接收 q/k/v、tensor args/kwargs（含 g、beta、initial_state）、返回 readout 与 updated state。H 模式捕获 H 注入后的 helper 输入。它们是明确命名的边界中间量，**不是所有内部 ATen 临时量的全覆盖证明**。
3. 每 token logits。
4. 完整 cache 对象递归 tensor 字节，以及原 `describe` 的类/属性/非 tensor 元数据；包含 attention KV、conv/recurrent state 和真实 cursor。
5. 原 state/logits finite 安全检查、每 token 每层恰1调用、256×24=6144次 GDN 调用、255×24=6120次下一 token 初始状态 pointer+bytes 消费检查。资格中增加的读取/CPU检查不用于 TPS。

`B0_Q0` 保留每 token 完整 raw reference；其余三条 fresh 进程逐 token 对同一个 reference 逐个实际 byte 比较，而非只比较浮点值。每个 token 保存 dtype/shape/stride/bytes/SHA/offset 与深 cache 元数据。B0重复等于基线；B1两条 fresh 都逐 byte 等于相同基线，故其自身重复性也得到严格、可审计的传递证明。CPU reducer 再核验全部256个 token receipt、整体 digest，并逐 tensor 重新读取 reference raw 检查实际 bytes/SHA。

任何首差立即保留最早 token、tensor 名、实际 raw 与 reference receipt，停止后续 GPU。进程退出/释放后，CPU `compare_failure` 计算首 byte、max absolute/relative error、nonfinite；signed-zero 等 byte 差不会因 max-abs=0 被放行。相对误差分母固定为 `max(abs(reference),1e-300)`，溢出以字符串 `Infinity` 保留，不能使严格 JSON 失败。nonfinite 在发现时即保存该 tensor raw/元数据并 STOP，不等到下一 token。

快照只留 `.55` 的新永久实验 root，不传 Mac。完整 reference 估计约40GiB级，但这是规划估算，非实际测量；准入至少留100GiB保护空间。CPU copy/hash/IO 成本尚未实测，可能使资格组无法在预算内完成；此时应真实 STOP，不能缩短覆盖、降低阈值或追加预算。

## 数值全部 PASS 后的性能

最多三组匹配重复，固定顺序：B0P0、B1P0、B1P1、B0P1、B0P2、B1P2。每条 fresh process 使用原 baseline worker 的 `replay` AST：128 warmup 保留 cache，位置128–255测128，窗口起止同步，不增加逐 token synchronize、hash 或 Profiler。保留原 Observe(on=False)/finite safety。

CPU reducer 要求全部六单位正确：mode/config/repeat/seed、同 GPU UUID、contract/source/prereg/fixture、actual loaded library digest、auth 和 fresh PID/starttick，TPS/wall 正有限且 `TPS×wall=128`，原 replay 256 forwards/6144 core/无 QDQ。有效收益事前固定为：

`median(B1 TPS) >= 1.05 × median(B0 TPS)` **且** `min(B1 TPS) > max(B0 TPS)`。

未完成六条、无可信收益、B1更慢或波动覆盖均不允许挑选更有利重复。performance 通过也不能证明已解释所有历史退化原因。

## 有条件的短栈 profile 与 Codec 范围

只有 FP 严格数值和可信性能收益都 PASS，且预算能完整容纳两条，才可选择 B0/B1 各一次128 warmup+32 measured profile；record_shapes=True、with_stack=True、profile_memory=False，仅修改原 replay 中的 instrumentation kwargs，数学代码不变。保留全部原 trace/摘要，profile TPS 不能混入裸 TPS。

之后如剩余预算能完整容纳四条180s allowance的组，才可运行原 H+C128 的同样四条256-input数值资格：原 H128/归一化、INT8 code 物化、scale、dequant、pre/post state、下一 token 消费。原 `quantize_c128` 数学 AST 不改，资格用 Python line observer 读取原 writeback 前局部量。它仍是原 FP32 QDQ carrier 的科研实现，不代表物理 packed INT8 常驻推理。

H 数值组未执行即 `NOT_QUALIFIED_FOR_THIS_CODEC`；此矩阵没有 H 性能组，也不能声称 H 获得了同等速度收益。Dual-M 没有注册执行单位，始终未获本轮 Fast 资格。Group32/其它 Codec 不能继承 FP 或 C128 资格。短程256输入不保证32K/81K路径等价。

## 环境、库选择与资源边界

未来 worker 对加载后的 `/proc/self/maps` 中实际 libtorch/libc10、cuBLAS、cuDNN、CUDA runtime 与 driver 路径计算 SHA/bytes。非 driver 必须命中事前绑定；pip/conda 两套文件同时存在不等于实际选择身份。组间真实 library digest 必须相同。driver 路径/SHA是未来实际观察，不能倒填历史 driver 身份。nvidia-smi 调用只位于授权后 runtime，独立5秒 timeout；本轮没有调用它。

单卡总硬限1800s /0.5GPUh，1650s保护停止，150s清理保留；四条数值各180s allowance，六条裸性能各90s，optional profile2×240s、H4×180s均须整组剩余容量足够才能开始。allowance 是保守调度额度，不是已知运行时间。控制器记录不可覆盖的首次 process exit 与 NVML release，无法精确测量的实际GPUh保持UNKNOWN或上下界。身份/数值/资源失败后不追加实验、不 patch-and-retry。

## CPU 资格证据和未通过项

`qualification/CUBLAS_CPU_TEST_RESULTS.json`：**15项合成 CPU 控制测试 PASS**，含默认NO、Python与TF32/同步ambient override拒绝、parent/dispatch防绕过、AST唯一变量、原 replay globals闭包、profile仅改观测控制的反向AST一致、固定 fresh schedule、合成/proc/maps库路径与SHA阻断、signed-zero/BF16/nonfinite/极端相对误差、四fresh完整256 toy字节证明、首差保留、token/raw篡改阻断、六条性能身份及≥5%/非重叠门槛。测试过程中没有导入 torch。合成数值不是 Qwen/Codec 科学证据。

CPU准备中发现并修正原 replay 抽取 namespace 的 stdlib `json` 绑定缺失；静态失败/修复历史保留在 `CPU_PREPARATION_FAILURE_HISTORY.jsonl`。没有因此启动或重试任何 GPU 单位。

**尚未通过**：真实 GPU 数值、fresh模型重复性、cuBLAS收益、真实 profile、H数值、Dual-M数值、显存峰、真实IO时长及资源准入。没有合法 Fast 配置，没有生成 `fast_validated_runtime.sh`。下一步只有在本任务显式 GPU 授权后，才能进入上述有限矩阵。
