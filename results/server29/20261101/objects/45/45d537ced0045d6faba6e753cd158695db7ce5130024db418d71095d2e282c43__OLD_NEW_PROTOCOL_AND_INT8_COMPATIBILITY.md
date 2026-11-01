# OLD/NEW 协议差异与未来 INT8 兼容性独立审计

状态：`CPU_ONLY_PROTOCOL_AUDIT_COMPLETE_GPU_NOT_AUTHORIZED`。本审计只读取本地已保存源码和历史小型回执，使用 Python 标准库解析 AST 与 SHA256；未导入被审计模块、torch 或模型，未 SSH、查询 GPU、执行 forward，也未修改历史资产或 codec。当前 `ALLOW_GPU_RESTORE=NO`。

完整来源、raw SHA、行号、12项事实、7项函数 AST 比对及未来资格建议在同目录 `PROTOCOL_COMPATIBILITY_AUDIT.json`。这些是 CPU 来源和协议证据，不是恢复版 GPU 数值/性能 PASS。

## 一、真正需要保留的三种执行协议

| 协议 | 首次 forward 与 cache | H+C128 QDQ 时序 | 合法解释 |
|---|---|---|---|
| 原 OLD AIME runner | 整个 prompt 一次 prefill，带 all-ones attention mask；从其 logits 采首 token | prefill 不 QDQ；每次后续 cached decode 完成后 QDQ | 原 AIME 生成协议，包含 sampler；不是 13.742 TPS 的固定短窗测量 |
| 历史 matched OLD | t0 单 token prefill，不显式传入 attention mask；t1..255 cached decode；保留128-token warmup状态 | 仅 t>0 QDQ，全程255次 | 13.742036 FP / 9.064610 H+C128 中位数的实际协议 |
| matched NEW / 新科学 worker | 同一短窗使用单 token 首 prefill；新科学 worker 也逐 token 推进 | 包括 t0，每个 forward 后 QDQ；短窗全程256次 | 7.776711 FP / 6.650598 H+C128 的历史短窗，以及新数学协议各自的证据 |

历史 matched `RESULT.json` 明确写出 `MATCHED_NON_AIME_PERFORMANCE_NOT_NUMERICAL_EQUIVALENCE`，不是数值等价资格。四组各保留256个 forward、24次 chunk、255×24次 recurrent 的实际计数。两组 H 的 QDQ 次数为255/256。虽然正式计时都是位置128..255，进入该窗口之前的 QDQ cache 谱系已经不同，不能通过忽略 t0 消除差异。

新科学 worker 保存的当前 logits 来源于本次 forward；其外层 QDQ 发生在该 forward 之后。因此 QDQ 主要影响下一个 token 对缓存的消费，不会倒改已经算出的当前 logits。Future-KL 的 token/预测位置必须按这个因果时序冻结。

## 二、同函数不等于同协议

OLD runner `9ff56d00...` 与新版 formal runner `77b1e1f6...` 的以下7个 AST 完全相同：`hadamard`、`normalize_qk`、`QwenKeyHadamardPatch`、`get_state`、`quantize_c128`、`load_model_and_tokenizer`、`setup_seed`。AST SHA逐项保留在 JSON。

但 NEW 的 core 会 memoize 固定 Hadamard，并替换 chunk scan；调用编排、prefill/QDQ 时序、数值检查及运行环境也不同。函数 AST 相同不证明完整输出逐字节相同，不证明 OLD/NEW 的 KL 数值可以直接配对。

两者架构源码均为 `90d92912...`。OLD 可恢复的 chunk fallback 源码在 `g.cumsum(dim=-1)` 使用 native scan；NEW `core.load()` 通过 `chunk_binding(...,'compensated')` 仅将这一处换为 FP32 Neumaier compensated scan。该替换没有改 recurrent helper 函数体。chunk/recurrent 带 hub/FLA decorators；已保存 Python fallback 源码不足以证明历史实际装载的物理实现或动态库集合。

## 三、状态、旋转和 QDQ 的具体合同

GDN helper 将 q/k/v/beta/g 转为 FP32；返回 FP32 recurrent state，读出转回原 query dtype。历史 matched 回执实际观察到 chunk 与 recurrent state 均为 `[1,32,128,128] / torch.float32`；模型参数 BF16。每层持久 recurrent state 为2 MiB，24层合计48 MiB，不含 attention/conv cache。

固定 H128 为归一化 Sylvester Hadamard。Patch 在 GDN 核心之前按原条件归一化 q/k，再分别做 FP32 右乘 H；v 不旋转。Patch覆盖 chunk 和 recurrent。OLD 每次变换构造 H，NEW memoize 常量矩阵。数学定义相同，生命周期不同。

旧 C128 的实际公式为：

```
source = state.detach().float()
scale = amax(abs(source), dim=-2, keepdim=True).clamp_min(1e-12) / 127
codes = round(source / scale).clamp(-127, 127)
dequant = codes * scale
state.copy_(dequant.to(state.dtype))
```

这里减二维是 K 维128个元素；scale形状 `[B,32,1,128]`，对每个 value column 一条 FP32 scale。实际旧函数的 codes 为浮点临时量，没有转成物理 INT8；持久状态是 FP32 QDQ 重建载体。不能把其速度或48 MiB载体表述为 packed INT8 runtime。

架构从 `cache.layers[layer].recurrent_states[0]` 读取下一次 initial_state；cache 更新函数使用 `copy_` 保留缓冲地址，C128再原位覆盖这个缓冲。可见源码没有保存另一条未量化 persistent state；`source`、`dequant` 等临时变量本身不是 persistent shadow。这个结论限于可见源码，不能替代实际别名/对象图和真实 next-consumption 资格。

NEW Observe 有每层 state finite 检查和上一轮 QDQ pointer 检查，开启或指定位置时还核对字节 SHA。OLD matched 主要保留 state shape/dtype、logits finite 和位置/调用次数；没有保存同等级逐 token state 消费字节证据。新恢复资格应独立补验，计时窗则保留历史检查口径，不加入逐 token 导出/hash，也不删除原安全检查。

## 四、环境证据的时间范围

以下是历史 matched OLD/NEW 这轮的实际 `ENVIRONMENT.json`，不是对更早原 AIME 运行时刻的追溯填空：

| 项目 | matched OLD | matched NEW |
|---|---|---|
| Torch / CUDA build / Transformers | 2.5.1+cu121 / 12.1 / 5.16.0.dev0 | 相同版本记录 |
| CPU intraop / interop | 24 / 24 | 1 / 1 |
| global deterministic | False | True |
| matmul TF32 / cuDNN TF32 | False / True | False / False |
| cuDNN deterministic | False | False |
| attention backend | sdpa | sdpa |
| workspace / OMP / MKL / device order | 全部 null，即该进程环境变量未设置 | :4096:8 / 1 / 1 / PCI_BUS_ID |

环境变量未设置不等于底层库没有默认行为。历史完整 DSO maps、hub/optional-package 实际选择和旧二进制环境没有被上述回执补全。当前同路径库 SHA不能倒推为历史已装载库 SHA。

## 五、未来 H+C128 / C32 Sym / C32 Asym 的科学边界

| 方案 | 本轮 CPU 能确认什么 | 仍然缺少什么 |
|---|---|---|
| H+C128 Sym | 原旋转/量化函数可恢复；有明确原位 QDQ 写回合同 | 恢复版真实 GPU 数值、scale/codes合法性、消费身份、fresh复现、速度均 NOT_RUN |
| H+C32 Sym | 可在相同外层写回/消费接口设计独立 codec | 精确group轴/布局、scale shape/dtype/epsilon、round/clamp/零组定义和实现尚未由本审计冻结/验证 |
| H+C32 Asym | 同一个 cache 接口原则上可承载重建状态 | 除前列项目，还缺 signedness、码域、zero-point公式/存储/舍入与常量组处理；不能当作已有合法codec |

如果采用恢复版 OLD，未来应重新运行同一恢复协议的 FP Native、H+C128 Sym、H+C32 Sym、H+C32 Asym。所有 H 臂共享同一固定 H 定义；所有臂共享模型/输入、GDN后端/更新语义、prefill/attention-mask、cache生命周期、QDQ插入规则、dtype、实际运行库、评分和统计合同。FP不施加量化本身是预先定义的方法区别，不允许混入另一种 runtime 或 teacher。

必须重建匹配的 FP teacher 与量化轨迹。历史 NEW_STRICT 下的 Future-KL、H/Dual-M、C6R/C7 数值只能作带协议标签的历史参考，不能直接填入 OLD 的 Group32 对照。若要做 AIME，还必须重新冻结 prompt、sampler、seed、EOS 和评分器；若做 Future-KL，还必须匹配位置、完整词表 FP32 KL、reducer、算术分辨率和统计单位。短窗通过不能替代32K/81K资格。

FP≥13 TPS不会推导出INT8≥13 TPS。历史9.064610才是 OLD H+C128 的相关短窗参考，也不是本轮恢复版 H 实测。

## 六、仅供未来授权的最小预登记建议

本节不是执行授权，不生成可用 Fast 或正式恢复入口。

建议固定单张可独占3090和原文档token SHA/seed20261015：先两次独立 fresh FP 数值进程，各完整256 forward（t0单token prefill、255 cached），再在全部数值资格通过后做3次独立 fresh `128 warmup +128 measure`。性能单位只做边界 CUDA 同步，无Profiler/逐tokenexport/hash。必须保留原 OLD 检查、原生位置/缓存和共同计数口径；若编排有适配，记录成明确的移植差异。

每个数值进程预留最多600秒，3个短窗各120秒，共1560秒dispatch上限；建议保护停止1650秒、清理预留150秒、硬上限1800秒=0.5 GPUh。这些是尚待根任务统一冻结的建议上限，不能被视为已验证耗时。任一单位失败立即停止，不重试、不扩额。NEW 当前对照只有另行冻结并仍能容纳于剩余保护预算才做，否则保留7.776711为历史参考。

数学资格和重复性分开：记录全部24层实际输入/更新/readout/缓存、完整248320 logits、finite、cursor、更新次数和真实下一token消费；用独立且执行前冻结的数学参考/误差合同判断正确性，再比较两个fresh过程每token完整对象的字节/元数据与最早差异。原 V3 误差合同文件 SHA `83e64320564f05d3b3d7eb79656d2c3198f396489d9cbb32d39dba449a8a6ec1` 可作为明确、前瞻采用的参考，但其 PASS原本属于新匹配数学路径，不能继承给native OLD。本审计未将这些阈值自动转成恢复版资格。

若找到真正保存的历史 OLD 数值参考，先核验身份再比；若找不到，只能证明本轮自身行为，历史逐字节一致为 UNKNOWN。若不满足冻结的fresh稳定性要求，应报告 `RUNTIME_REPRODUCIBILITY_FAIL`，不因快而放宽门槛。1-token prefill 的256短程即使通过，仍未覆盖多token chunk prefill、32K/81K、量化或随机生成任务。

## 七、本轮结论

CPU来源支持恢复 OLD 执行协议与 C128写回接口的可行性；当前没有恢复版 GPU数学、fresh或TPS实测。H+C128/C32 Sym/Asym均未取得恢复版 INT8资格。新 compensated-route资格明确保留 `old_historical_parity=FAIL_UNCHANGED`，不能将该旧失败改写，也不能继承其PASS给OLD。

本轮合法状态仍为 CPU审计完成、GPU未授权；历史二进制精确复现、实际安全性/重复性、当前速度和长程科学可用性保留未决。未启动任何H、Dual-M、Group32、AIME或Future-KL。
