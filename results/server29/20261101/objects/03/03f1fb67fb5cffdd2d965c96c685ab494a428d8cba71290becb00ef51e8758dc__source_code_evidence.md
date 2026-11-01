# 源码和静态边界证据

| 路径 | HEAD | branch | dirty 条目数 |
|---|---|---|---:|
| `/data/zypan/GDN-quantization` | `8f631858767c369829133d88c769f84019677c1e` | `research-sync-2026-09-02` | 188 |
| `/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2` | `a979a9f678f4ba0d6b9510a56760d36728929b8c` | `exp/qwen-gdn-c7-persistent-final-hidden-v1` | 157 |
| `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen` | `fa7918d82c144a30ba3459114abb53e70df73d98` | `exp/hadamard-init-dense-orthogonal-oracle-v1-qwen` | 1339 |

三处历史工作树都不干净。完整逐项 porcelain 状态及 origin URL 保存在 `source_manifest.json`；没有 reset、clean、历史源码编辑或 push。新资产写入独立目录（本地和计算服务器路径同见 manifest），该目录自身不是 Git checkout；新代码用 SHA256 标识。

STEPQuant 官方： https://github.com/Dreamer-Toby/STEPQuant 。指定 commit、实际 clone HEAD 和本轮 `ls-remote HEAD/main` 均为 `f2deac58131219e6aecff274f8eabe88e08d2e3c`，detached HEAD，clone clean。本轮未自动采用未来版本。

| 原始路径 | 字节 | SHA256 |
|---|---:|---|
| `/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/shared/rotation/cayley_rotation.py` | 7549 | `62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067` |
| `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py` | 34744 | `77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69` |
| `/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py` | 88546 | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` |
| `/data/zypan/GDN-quantization/results/propagation/gdn_rw_int8_quantizer_v1_final_classification.json` | 4709 | `54f9a7f42237ef96a869a2ffeaf096a9bbc218158398a83f864ecb189d229a52` |
| `/data/zypan/GDN-quantization/results/propagation/gdn_rw_int8_quantizer_v1_protocol_audit.json` | 1642 | `5787a272ce00617a8c88338efa3a2609ce34b04ebf365f11faa67c49c0e9ab29` |
| `/data/zypan/GDN-quantization/results/propagation/gdn_magnitude_conditioned_functional_v1_final_classification.json` | 22908 | `47eef0a2ebb74f8835cc1edeafed9ea993552cf9f2200767b74284130bc44a81` |
| `/data/zypan/GDN-quantization/results/propagation/gdn_magnitude_conditioned_functional_v1_protocol_audit.json` | 2382 | `2c9bdd49401fe3fff7a96a773d59f037316fba3adf15ac69e5b8c982c88a15f0` |
| `/data/zypan/GDN-quantization/experiments/propagation/run_gdn_read_write_aware_int8_quantizer_feasibility.py` | 25885 | `5f5d082ccbfaf62da0b22aaa04a77a9be7cb2389911b2d7e717b6f3a8a1c7ed9` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/numerical_recovery.py` | 35065 | `b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c` |
| `/Users/pytenter/Documents/临时项目1/qwen_hadamard_dual_axis_int8_feasibility_v1/references_STEPQuant/stepquant/quantization.py` | 7245 | `0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69` |
| `/Users/pytenter/Documents/临时项目1/qwen_hadamard_dual_axis_int8_feasibility_v1/references_STEPQuant/stepquant/core.py` | 1741 | `3947d82c2f90d248a68e82773927f2b7b04d4044f678f17dfd7a4bfb931883b0` |
| `/Users/pytenter/Documents/临时项目1/qwen_hadamard_dual_axis_int8_feasibility_v1/references_STEPQuant/stepquant/calibration.py` | 7003 | `f34a78c1b260d99621df4d54ba6e32d92fd05740f8354b467a000382b1f675eb` |
| `/Users/pytenter/Documents/临时项目1/qwen_hadamard_dual_axis_int8_feasibility_v1/references_STEPQuant/stepquant/packing.py` | 945 | `4d42479d031561ed9e755eb1aab86b78dd03446be38932cd518174f80033eff4` |

## C128、Hadamard 与 runtime

`cayley_rotation.py:134–149` 的 dataclass、EPS、canonical_symmetric_int8_qdq 和 qwen_c128_qdq 通过 AST 原样编译；baseline 实际调用该冻结函数。没有导入训练 runner。source=value.float()；沿 -2（Key）absmax，先 clamp_min(1e-12)，再除127；torch.round ties-to-even，clamp[-127,127]；codes*scale 重构。原函数临时 codes 是 FP32 整数值，scale FP32；新物理编码把 codes 转 signed int8。

`run_qwen_c5_c6_formal.py:175–218`：128阶归一化 Sylvester H；Q/K 若需归一化先 value*rsqrt(sum(value²)+1e-6)，然后 FP32 `value @ H`；Value 不旋转。Hadamard 坐标下 state 为 H.T @ S_native。本轮仅合成 Hadamard case 从 native 合成 state 转换一次，其余生成的 state 直接定义在 H 坐标；无 C6R、Delta 或 rotation 训练。

正式 runner 还包含另一个 FrozenRotationPatch，但本轮不执行该类。只提取原 get_state、quantize_c128、Hadamard 及静态常量。`quantize_c128:269–286` 要求 state 后三维[32,128,128]，每次更新原位 copy_(dequant.to(state.dtype))；原正式函数也检查 dequant finite。G1 在 CPU fake cache 上执行这段**未经改写**的函数，sys.settrace 只在返回时读取 codes/scale/dequant locals，并检查24层写回结果。

`modeling_qwen3_5.py:260–265,345–361,535`：torch fallback GDN 把 q/k/v/beta/g 转 FP32，创建 FP32 recurrent state，返回后更新 cache。原 runtime 物理保留的是完整浮点 QDQ state；它没有通过此路径持久保存 int8 codes。模型权重 BF16 与 recurrent state FP32 需区分；本轮未重新证明实际 GPU dispatch、device map 或 fused kernel 路径。正式 AIME loop 在初始 prefill 后采样第一个 token，然后在每次单 token decode forward 返回、下一次采样前调用 quantize_c128；初始 prefill 后无该调用。未来新 runtime 必须明确冻结 prefill/anchor边界，不能把旧 loop 的静态审计当作全 token 集成资格。

## STEPQuant 映射和差异

官方 `stepquant/quantization.py` 的 QuantizationPlan、_fp16_scale、fit_state 通过 AST 原样编译，未导入 StateCodec、模型或 calibration。所有 bits=8、architecture=gdn、impact=ones，因此无 pivots、无 INT2、无 temporal allocation、无动态 importance。源码链接： https://raw.githubusercontent.com/Dreamer-Toby/STEPQuant/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/quantization.py 。

采用原下界：row abs-mean 1e-20、LS den 1e-30、scale 转 FP16前 clamp[2^-24,65504]；零分母 retain 旧 c。除法严格先 x/r 再 /c，拟合一次，存储最终 scale 后重编码；decode 顺序 (r*c)*z，与官方 fit_state 完全一致。官方 StateCodec.decode 使用 z*r*c，运算结合不同，本轮不宣称与该 decode 逐位等价。官方 fit_state 返回临时 int16 levels；本 codec 最终 int8，比较时转换 int16 验证整数一致。官方 pack 存有偏移的 uint8，而本 codec 直接保存 signed int8，字节格式不同但整数值相同。

官方 core.py 的 output 在 writeback 前由 updated state 读出，row_impact 为 transported read 的平方，impact_factors gamma=.25；calibration.py 从 FP teacher trace 采样并进行 mixed-bit 分配，默认 GDN pivots=32。本轮未运行上述 calibration/allocation，也不实现 weighted/Dual-W。C 是我们自定义的单轴 LS 消融，并非 STEPQuant 独立方法。

## 历史资产

两个旧 quantizer/functional 研究只作身份和设计背景参考；未把旧结果当新 codec 数据。8-doc 研究、旧 AIME 固定结果和时空分析未改动。读取了26个 V3 小索引，发现 native FP full-cache 数值资格资产，但不是已有 Hadamard FP 轨迹，未加载或复制。read/write 旧 runner 的 torch.load 指向 G 指标对角；MSE/KL/scale 摘要不构成完整 state。C6R 8-doc state 排除。`REAL_STATE_DATA_AVAILABLE=NO` 的范围限于本轮可核验且符合坐标约束的历史 state；不是声称所有服务器都没有 FP tensor。详细排除证据见 `analysis/attempt_002/real_state_availability_audit.json`。
