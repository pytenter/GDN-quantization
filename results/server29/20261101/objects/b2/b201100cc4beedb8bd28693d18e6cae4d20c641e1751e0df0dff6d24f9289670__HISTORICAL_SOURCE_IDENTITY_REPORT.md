# 历史 13.742 TPS 源码与执行身份审计

本报告只读核验历史小文件与源码，没有导入 Torch、构造模型、查询 GPU 或执行 forward。当前 `ALLOW_GPU_RESTORE=NO`。结论是**旧源码和短窗口协议可以恢复；候选尚未取得 GPU 数值或性能资格，历史二进制环境精确重建仍为 UNKNOWN**。

## 13.742 TPS 的真实含义

`13.742036038730332 tokens/s` 是历史调和任务 `OLD_FP_R0/R1/R2` 三次普通执行的中位数，三次分别为 `13.640066181342853 / 13.742036038730332 / 13.74268286877`。该实验在固定非 AIME Doc1 的前 256 个 token 上，先执行 128 次 forward，再同步计时后 128 次；batch=1、prefill=1 token、保留 cache，同一 GPU UUID `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`。它是 `MATCHED_NON_AIME_PERFORMANCE_NOT_NUMERICAL_EQUIVALENCE`，不是原始 AIME 自由生成吞吐。旧 H+C128 三次中位数为 `9.064609924852187`。

本轮重新核验了历史清单中的 **7 个执行文件、194 个外部小依赖以及 23 个执行证据文件**，全部 SHA/bytes 匹配。三次旧 FP/三次旧 H 的 source/plan/fixture 绑定、256 个实际三轴 RoPE 位置、八个 attention KV 长度的历史元数据也逐条通过。本轮没有将这些元数据冒充 per-token logits/state 的历史字节参考。

历史 source manifest SHA：`b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3`；输入 JSON SHA：`18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d`。完整输入是 32768 个 ID，实际只消费前 256 个；本轮证据另列该前缀的 canonical JSON SHA。

## 可恢复的实际源码

| 对象 | 历史原始 SHA256 | 证据与边界 |
|---|---|---|
| `matched_worker.py` | `08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c` | 历史执行清单和三次 RESULT 绑定；本地及当前远端 raw bytes 匹配 |
| `matched_controller.py` | `c16ae53e7fb2c4ea6c1c02d798d79f09310d62cb6ce5bf41b4b2035a354b46da` | 历史调度与环境继承协议 |
| 旧 `run_qwen_aime26_formal.py` | `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd` | worker 54–56 行导入实际 `aime26-sglang-rotation-v1` 路径并断言 raw SHA；canonical 备份也由外部依赖清单固定 |
| `aime26_common.py` | `34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7` | canonical 路径在旧清单内。实际 OLD 同目录路径本轮当前 SHA 相同，但该实际路径不在旧清单内；不能补成历史 import 绝对路径的事前 pin |
| Qwen3.5 architecture | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` | worker 68 行断言实际模块 raw SHA；保留全文 |
| `cache_utils.py` | `b63732985b520a62aa4b06fc2af8ca6e506912679e0f0479b706534d041fb1bd` | 历史外部依赖固定 |
| `hub_kernels.py` | `4d8a568275fc46da1e710bc3d9b01a4fc84ee80b862f3ecc8e28b42683885a94` | 必须保留装饰器分派语义，不可为提速静默强制 fallback |
| 模型 `config.json` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` | 32 层，`(L,L,L,A)*8`，24 GDN/8 attention，vocab 248320 |

完整 201 项可恢复小文件列表、原路径、当前本地副本、bytes/SHA 和关键函数 AST 摘要见 `LOCAL_HISTORICAL_SMALL_ASSET_VERIFICATION.json`。本轮 13 个小文件远端只读 hash 中，12 个有历史绑定且匹配；另一个为上述实际 common 的 current-only 补充。首次使用远端 `python3` 命令不存在的失败回执单独保留；改为既定绝对 Python 后仅使用标准库完成，未改远端环境。

## forward、状态与 prefill 协议

| 部分 | 冻结代码位置 | 可确定内容 |
|---|---|---|
| 模型加载 | old runner 170–187 | 本地离线 AutoTokenizer/AutoModelForCausalLM，BF16，单卡 `device_map={"":0}`，`eval()`；不加载新 checkpoint |
| 固定输入 forward | matched worker 70–72、99–127 | 每次输入一个已冻结 token；`model(input_ids=...,past_key_values=cache,use_cache=True)`，没有原 AIME prefill 的显式 attention_mask |
| 首 token 与 recurrent | architecture 444–535 | 没有 previous state 时使用 chunk helper；有 cache 且 seq_len=1 时使用 recurrent helper，更新 recurrent cache |
| chunk 数学 | architecture 249–327，特别是 291 | 恢复的 Torch fallback 把 q/k/v/beta/g 转 FP32，chunk 内为原生 `g.cumsum(dim=-1)`；旧 runner 没有新 compensated cumsum 替换 |
| recurrent 数学 | architecture 331–380 | q/k/v/beta/g 转 FP32；依次 decay、`kv_mem`、delta、外积 state update、readout，最后转回输入 query dtype；该函数体没有 cumsum |
| state carrier | worker 88–98；旧 RESULT state_metadata | 历史实际 helper 返回 state 为 FP32 `[1,32,128,128]`；输入 q 为 BF16 `[1,1,32,128]` |
| cache | cache_utils 1045–1086 | 首次建立 conv/recurrent buffer，后续 recurrent `copy_` 到持久 buffer；完整 attention KV 按实际 token 增长。`mark_static_address` 本身不是 CUDA Graph 已执行的证据 |
| attention/readout | architecture 668–701、711–718 | Q/K norm、RoPE、KV update，依配置选择 attention interface；matched ENV 实际为 SDPA；原 MLP gate/up/down 不变 |
| 原始 finite gate | matched worker 113 | 保留旧 logits `isfinite().all().item()`；旧路径没有 NEW Observe 的全层额外 state finite collector |
| 计时内审计 | matched worker 73–98、121–126 | 保留原普通测量中的 Python hooks、shape/dtype、cache 长度与计数检查，t255 进度日志也在窗口内；不得把移除这些审计后的新速度当同协议复现 |
| 同步与位置记录 | matched worker 100–105、127、133–134 | 计时边界同步；RoPE 小 tensor 的 `.cpu().tolist()` 在窗口结束之后；普通 TPS 没有 profiler |

历史 helper 计数为：256 次模型 forward、8192 decoder 层调用；t0 共 24 次 chunk，后 255 token 共 6120 次 recurrent，计时区间 recurrent 3072 次。计数确认 helper 入口与 phase；不单独证明该装饰器最终采用哪套物理实现。

`hub_kernels.py` 792–835 行按 HF kernels（若请求）、原可选包、Torch fallback 分派；812 行会尝试导入 FLA，824 行 `functools.wraps` 可能令 unwrap 所见 Torch 源码与实际 implementation 不同。历史完整动态分派信息及 DSO maps 未保留，故不能仅凭 90d raw SHA、unwrap 文件名或 helper 调用计数把历史物理 kernel/library 身份写成完全已知。可恢复原源码分派规则；不得偷偷禁用可选分派以制造 “exact”。

## 旧 AIME 与后续量化兼容性边界

原 AIME `generate_one` 270–309 行执行完整 prompt 的一次 prefill，使用全 1 attention_mask，第一次采样来自 prefill logits；之后一 token cached decode，非 FP 方法才做 C128 QDQ。原 AIME 的总计时包含采样、finite gate 和量化，不能与 13.742 的固定输入后 128-token decode 窗口混称。

旧 H+C128 在匹配测试的 t0 不做 QDQ，256 forward 共 255 次 QDQ；NEW 版本则 256 次。旧 C128 是原 `amax(dim=-2)`、floor `1e-12`、round/clamp `[-127,127]`、dequant `copy_`；旧 Hadamard Q/K 旋转与 readout保持原序，不可将新缓存优化或 Dual-M/Group32 codec 换入并称同一配置。将来 H 或 Group32 symmetric/asymmetric 需单独授权、同 runtime FP 重新建基线，当前审计不授予这些 GPU 工作。

原始长期 AIME 跑完、state dtype FP32，以及本次短窗口性能结果都不证明任意 32K/81K 数值稳定。旧 chunk 的原生 cumsum 路径必须作为数值风险保留，尤其未来若换成真实多 token prompt prefill；本次 1-token prefill 的资格范围不能覆盖多 token prompt。现有吞吐测试没有保存完整历史 per-token logits/state/update 字节参考；未来同候选至少两次独立 fresh process 的一致性也只能证明所冻结短前缀的 repeatability（当前候选拟定32-token prefix），不能追认历史逐字节一致或长期正确。

## 环境与恢复模式建议

真实 matched OLD ENV 已记录：Python3.10.18、Torch2.5.1+cu121/CUDA build12.1、Transformers5.16.0.dev0、intra/inter-op=24/24、deterministic=False、matmulTF32=False、cuDNNTF32=True、cuDNNdeterministic=False、BF16 模型、SDPA。`CUBLAS_WORKSPACE_CONFIG/OMP_NUM_THREADS/MKL_NUM_THREADS/CUDA_DEVICE_ORDER` 的记录为 null，表示当次 unset；其他没记录的环境项（如 LD_LIBRARY_PATH）是 NOT_RETAINED，不能扩张解释为 unset。旧 controller 只继承环境，给 OLD 分支设置离线/CVD 等，不主动清除以上变量；候选若显式重建已记录状态，应将这种前置环境构建写入新协议。

旧 source manifest 不含历史 DSO 二进制 SHA，也没有完整旧 maps；相同版本字符串不等于相同历史物理库。这不妨碍恢复已保存的源码，但阻止 “EXACT historical binary environment 已恢复” 的结论。另需区分：早期原 AIME 全 shard SHA 不完备，而后来生成 13.742 的调和批次已有共同 checkpoint 的 4 shard CPU raw SHA 身份文件。模型当前 fresh hash 由独立环境审计负责；本子任务没有重复读取模型大文件。

建议隔离候选采用 **HISTORICAL_PORT，CPU_ONLY / GPU_NOT_QUALIFIED**：原 runner/math raw copy 完整保留，记录将 matched worker 的 OLD 指向隔离副本、对应相对 common 导入、受控环境与当前依赖绑定的全部路径适配；逐函数/AST证明数学差异为 0。路径重绑定本身不授权改变 math。模式 A 仅可说明 source/protocol-faithful 可恢复部分，不能在二进制环境缺口存在时写成已完成 EXACT replay。A 或 B 最终均须未来独立人类授权、冻结 admission、至少两次独立 fresh process 数值检查（当前候选拟定32-token prefix，C3未指定长度）与后续分离的 3 次 128+128 普通 benchmark；本报告不把这些未来资格标 PASS。

结论状态：`CPU_SOURCE_IDENTITY_AUDIT_COMPLETE_GPU_NOT_AUTHORIZED`。本审计保持所有历史文件不变，未作新的 root-cause、数值性能实验或环境安装。
