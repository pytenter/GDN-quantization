本次自行修复已完成两个已定位工程缺陷，并以独立 execution_02 执行 GPU 资格。结果为 FAIL_AND_STOP：第一篇文档第1个 token 的 FP_NATIVE 与 FP_H 全词表 logits 不符合冻结相对L2上限。所有后续量化器试验停止，GPU 已释放；这是未量化的前置资格失败，不能判定 Dual-M 有效或无效。

VERIFIED_ENGINEERING_RESULT：原封存、权重、源码和两篇 token 身份通过；12项最小 codec CPU 回归、4项工程检查（含24层原正式 C128 写回逐字节）通过；GPU0 准入通过；新启动器确实在已有 CPU PASS 后派发 GPU，并在真实门禁失败后自动停止。

EXPLORATORY_SCIENTIFIC_RESULT：NOT_RUN。没有通过资格的真实 Hadamard state 分布、A/B/C/D 局部 MSE 或两文档 8/32/128 KL 结果；不生成空 CSV，不用旧 CPU 数值填充 GPU 结果。

NOT_ESTABLISHED：FP/H full-model 等价性未建立，canonical C128 GPU parity、GPU fresh-process、observer OFF/ON、persistent QDQ、physical INT8 均未资格；Long-horizon Future-KL、AIME、C6R互补性及新 rotation 训练目标均未验证。

```text
TASK = QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1
EXECUTION_ATTEMPT = execution_02
GPU_CANARY_AUTHORIZATION = YES
CPU_V1_SEAL = PASS
PREREG_AND_ADDENDUM_IDENTITY = PASS
FROZEN_DOC_TOKEN_IDENTITY = PASS
GPU_RESOURCE_ADMISSION = PASS
GPU_DEVICE_UUID = GPU-a78176ed-f97f-c54f-6201-5a01b570f422
FP_NATIVE_H_EQUIVALENCE = FAIL
REAL_HADAMARD_STATE_CAPTURE = NOT_RUN
REAL_STATE_CODEC_COMPARISON = NOT_RUN
FP16_SCALE_SAFETY = NOT_RUN
CANONICAL_C128_GPU_PARITY = NOT_RUN
FRESH_PROCESS_DETERMINISM = NOT_RUN
PERSISTENT_QDQ_RUNTIME = NOT_RUN
PHYSICAL_INT8_RUNTIME = NOT_RUN
SHORT_CANARY_128 = NOT_RUN
DUAL_M_REAL_STATE_MSE = NOT_RUN
DUAL_M_SHORT_KL = NOT_RUN
GPU_HOURS_USED = 0.019832093492150305
MAX_GPU_HOURS = 2.0
FULL_MODEL_2K = NOT_RUN
FULL_MODEL_32K = NOT_RUN
AIME = NOT_RUN
C8 = NOT_AUTHORIZED
FINAL_CLASSIFICATION = FAIL_AND_STOP
```

1. 硬件与 runtime：.55 physical GPU0，RTX3090 24GB，UUID GPU-a78176ed-f97f-c54f-6201-5a01b570f422；CUDA_VISIBLE_DEVICES=0→cuda0，未更换设备，未用其余三张卡。Torch2.5.1+cu121、CUDA12.1、Transformers 5.16.0.dev0、tokenizers 0.23.1、Triton 3.1.0；FLA distribution metadata 未检出，不能据此断言运行时没有对应 kernel。live architecture/recurrent 原源码检查和 compensated chunk 绑定已执行；没有引擎、模型参数dtype或 device-map改造。模型 BF16 loader 来源及14文件逐流 hash 见 PARENT_AND_MODEL_CONTENT_VERIFICATION.json；原参数dtype详细计数在失败前未保存完整成功回执，不补造。当前仓库 git HEAD 见 NUMERICAL_FIRST_FAILURE_CPU_AUDIT.json。

2. FP/H 数值资格：Doc1 token1 的248320维 BF16 logits 全有限。max_abs=0.09375（cap=0.11750000000000001，通过），relative_L2=0.006718461729707917（cap=0.002，失败），floor-stabilized max_rel=0.04521276595744681（cap=0.1，通过）；三个条件必须同时满足。exact_equal_fraction=0.26016833118556704，byte_exact=false。CPU 保存张量重算每个字段与 GPU first-failure 一致。尚未比较后续 final_hidden/state/readout，尚未第二进程复现；数学等价性不能用接近或 max_abs 通过替代。内部最早算子未定位；坐标变换/归约/转换舍入只是待检验假设。

3. 真实 H state row/column 分布：NOT_RUN。原定两文档×1/8/32/128×24层×32头，在 A0 首比较失败前没有保存任何完整 H state snapshot。不能把保存的 logits 当 state 或将 native state 事后旋转冒充合格轨迹。

4. A/B/C/D 局部误差：NOT_RUN。无 Delta_MSE，无 per-layer/head 收益，没有正面、负面或混合探索结论。

5. FP16 scale/clipping：NOT_RUN。原冻结 affected-energy≥1e-4 且 affected relative-MSE>1e-2 的停止定义未改；极端 CPU stress 历史失败保留。尚不能证明真实 GPU state 安全。

6. Canonical C128：原正式 AST 对新 adapter 的 CPU 24层 codes/scales/dequant/writeback exact PASS，包括负零恢复；GPU full-model B0 未运行，不将 CPU PASS 升格为 GPU资格。HISTORICAL_END_TO_END_REPRODUCTION=NOT_RUN。

7. 逐 token QDQ：本次 A0 两条件各消费 Doc1 的1个 token，无 QDQ；GPU encode/writeback 调用次数0。下一 token 读取 QDQ、每层每 token24次写回、observer OFF/ON 等均未运行。修复的 canonical 零符号 I/O 不在本次 A0 调用链。

8. 两文档短程 KL：NOT_RUN。Doc2、8/32/128评分点以及所有学生条件均未执行，无 scientific CSV 和显著性检验。raw文档不同，repository/project独立性仍 UNKNOWN；Ling已知四ID排除记录保持，未伪造原manifest路径核验。

9. GPU 性能：codec encode/decode/writeback median、次数、full-model token/s、同步计数、checkpoint恢复代价均 NOT_RUN；失败前未形成性能回执，不能由36秒占用估算codec吞吐或2K成本。原 Python reference 的速度也不能代表未来 fused kernel。

10. 存储显存：本 attempt NVML目标间隔0.5秒（实际含查询开销）采样峰值 17588MiB，非连续精确峰值；Torch peak allocated/reserved 未在失败时保存，为 NOT_AVAILABLE。原定公平比较使用 FP32 QDQ cache，物理 INT8 未资格。仅保存约1MiB失败 logits；无整模型权重或完整FP state 下载，raw_snapshot目录为空；bytes审计在 final seal 回执。当前cache实际 bytes、完整临时FP分配未测，不能用理论shape替代测量。

11. GPUh：execution_01 0.009828903468118773 GPUh（35.3840524852秒），execution_02 0.010003190024031532 GPUh（36.01148408651352秒），累计 0.019832093492150305 GPUh（71.3955365717411秒），上限2.0 GPUh。包含模型加载、初始化、失败前forward、持有与释放，不重置预算。原账本逐样本保存。最终只读NVML核对全部4张卡各使用3MiB、GPU0利用率0%、无compute进程；无本次 worker/coordinator。

12. 失败历史：见 failure_history.md；保留历史native cumsum与CPUstress FAIL、原GPU_NO STOP、execution01工程失败、本轮两项pre-GPU错误和真实A0数值FAIL。独立GPU attempt只有01和02，未在02里修补重跑。

13. 身份封存：source_manifest.json 在GPU前冻结10个文件；父140文件seal、execution01 seal和冻结执行源码在GPU后再次核验。保存的 reference/candidate 张量SHA分别为 d4629d7af39dd74e4bddfcec894aaaa1116d2b1cfe5ccf5b863aa21406b4e4b1 / f39191b98bdea42fe744e8b3a310e4a297c1e7e312f9f31e17b88354e664ca72。两篇固定token SHA分别为e8b4bd1b698e1f77d24f8a705c6cc8b1b7a703c1d472c4c2fbfb1a5a960e1d5b / d59307230ee4f81a922ab0d3e62092643227cc2ea29ff33ffd39b6c06db29c3c。逐产物SHA见 hashes/artifact_sha256.txt；本地和远端字节独立比对见 hashes/DELIVERY_VERIFICATION.json。清单/末回执为避免循环哈希互相排除，receipt自身另行核对一致。

14. 后续建议：核心runtime资格未完成，不生成 NEXT_2K_EXPLORATORY_PREREG_DRAFT.md。下一项必要工作是定位 FP_NATIVE/FP_H 首个内部数值差异，并依原协议处理新的独立资格；当前证据不足以推荐Dual-M长程实验、预测2K GPUh或判断8K/32K价值。未运行2K/32K/AIME/训练，未push，未启动后续后台实验。
