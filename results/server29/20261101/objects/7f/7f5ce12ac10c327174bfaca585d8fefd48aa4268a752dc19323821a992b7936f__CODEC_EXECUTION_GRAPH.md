# 原 Dual-M 与官方 STEPQuant：CPU 调用链和性能边界审计

本报告只读源码，没有执行 GPU、没有选定或实现候选。**实际 kernel 数、前三耗时阶段、LS 占比和加速比均待 Phase 0 profiler，不能由 AST 操作数推算。**

官方参考使用已冻结 commit `f2deac58131219e6aecff274f8eabe88e08d2e3c`。七个指定文件已逐 SHA/bytes 匹配历史官方 manifest，并重新从该 commit 的 raw URL 取文本，七份字节均一致。GitHub 当前首页仅供上下文；没有更新运行时。[官方仓库](https://github.com/Dreamer-Toby/STEPQuant)、[固定 commit 的 pool.py](https://raw.githubusercontent.com/Dreamer-Toby/STEPQuant/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/kernels/pool.py)。源与行号的机器可读证据见 `REFERENCE_IMPLEMENTATION_AUDIT.json`。

## 真正的旧执行流程

```mermaid
flowchart TD
    A[单token Model Forward] --> B[各GDN层: 原normalize_qk + Q和K各一次FP32 H128 matmul]
    B --> C[原冻结GDN update和当前readout]
    C --> D[全部24层forward结束，native FP32 recurrent state已更新]
    D --> E[torch.cat 24层state为24×32×128×128]
    E --> F[原encode AST: row FP16 scale]
    F --> G[initial column FP16 scale]
    G --> H[初次INT8 levels]
    H --> I[一次column LS: v=r*z，sum v*x / sum v²]
    I --> J[拟合column转FP16，最终re-encode]
    J --> K[(row.float * col.float) * codes.float]
    K --> L[有限性与原FP16 scale风险门禁]
    L --> M[24次state.copy_，写入原FP32 cache carrier]
    M --> N[下一token原GDN消费该QDQ cache]
```

这条路径未调用官方 Triton `PackedStatePool`，GDN update/readout 和 Dual-M 没有融合。普通路径只做一次 encode 和一次 decode；只在非科学 canary 的资格时刻才额外执行官方 fit/逐层 reference parity，不应混入基线测速。

旧32K在每个输入token后QDQ，而最近AIME是同一批量prefill后QDQ、随后逐decode-token QDQ。两者数学codec同源，但不能直接将两套吞吐当同协议速度比。历史32K `core.load()`还绑定 compensated chunk；该合同与最近AIME original90d/detFalse不同，新优化必须遵循本任务恢复的冻结参考，不能借优化更换GDN数学。

## 同步、发射和观察开销

原 `trace_encode` 正常finite路径一次批量codec可静态确认10个Tensor→Python标量检查站点：source有限性2，safe_half NaN3，levels NaN2，validate code/columns/rows3。外层dual另有decoded finite1、bound branch1、raw-hit总计8次int(sum)，合计**20个站点/批量token**，再加每token CUDA event synchronize。此处是控制路径站点数，不是已测量的实际同步API数或耗时。

metrics位置另有7个float标量统计；原scale-risk分支另有double完整state/误差/energy/mask计算及标量检查。observerON会进行CPU tensor SHA，ON/OFF移除必须先逐codes/scales/state/logits和下一token消费验证。observerOFF仍有module prehook、kernel wrapper、调用计数、有限性和消费身份检查。原32K外层还在forward后全局synchronize，并在QDQ后event.synchronize；新测量不应为了计时再插入逐token全局同步。

H transform每模型token有48个FP32 matmul Python调用（Q/K各一次×24层），H矩阵已memoize。**不能把48次matmul调用直接称作48个CUDA kernel。** 原H+C128的24层循环还有finite/min/max的72个标量读取站点；Dual-M虽多LS，却通过24层batch减少逐层发射/同步，不能事先假定LS是额外墙钟的主因。

## 内存生命周期

单层 `[1,32,128,128]` FP32为2,097,152B；24层native carrier和额外cat各50,331,648B。`source.detach().float()`对已FP32状态不必再复制。scaled、float levels、v、decode有效scale、codes.float与decoded结果会产生多轮完整state大小中间张量，但生命周期不同，不能直接相加作峰值。trace_encode的r/c0/c1三个rawscale FP32 clone合计1,179,648B。条件risk分支每个完整double state为100,663,296B，需实测实际共存峰值。

INT8 codes24层12,582,912B，FP16 row/column scales各196,608B，编码payload合计12,976,128B；persistent carrier仍FP32 48MiB。不能把编码payload称为物理INT8常驻推理。

## 官方设计哪些可参考

| 路径 | 源码可确定的显式Triton launch | 边界 |
|---|---:|---|
| dense encode | rows + column，共2 | column合并初始列scale、初码、LS、最终码和packedstore |
| dense decode | 1 | codec_kernel或byte_decode |
| 同步packed GDN step（batch1 all8） | update + column，共2 | update/readout已换官方GDN，不是本任务codec-only替换 |
| 异步packed GDN step（batch1 all8） | readout + prepare_rows + fit，共3 | 另有stream/event依赖；保持packed state，非原FP32 carrier |
| 旧PyTorch Dual-M | 无官方Triton；实际CUDA数未知 | 必须CUPTI/profiler确认ATen到kernel与memcpy映射 |

`fit.py`把初始列scale/初码/一次LS/最终码/store放在同一tile，`fused.py`提供dense row+column拆分；这些融合和coalescing边界可参考。`rows.py`和`readout.py`的affine coefficients/partial sums则依赖官方packed recurrence，不能整体移植。`pool.py`按16-value tile布置packed GDN，byte布局为key连续signedINT8；它不是现有signedINT8 code tensor的同一物理布局。

官方async用writer.wait_stream(current)、next reader.wait_event(completion)、stable scratch和record_stream；CUDA graph另要求捕获张量地址/生命周期、joined或paired依赖。它们允许与后续无依赖层工作重叠，但不能让下一token读到未完成写回。当前CPU bool/item安全检查也不能静默删除以强行capture。整套packed runtime或完整SGLang移植均不在范围。

**直接复制官方kernel不能自动标EXACT**：其归约树、fast_dividef、reciprocal乘法、FMA及normalized weighted LS重排可能使FP16scale或INT8code变位；原decode乘法结合顺序也必须保留。官方fused额外依赖Triton≥3.6/Gluon与cuda-bindings≥13，旧环境为Torch2.5.1/Triton3.1，本次没有升级或导入该GPU栈。

Phase 0应分别记录非instrumented固定token wall、CUDA event和短profiler trace。先区分模型/GDN主干、Hadamard、codec数学、CPU发射/标量同步与诊断开销，再决定唯一候选方向；在profile前不声称瓶颈、加速比或全模型加速上限。
