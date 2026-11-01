本轮CPU静态兼容审计完成；**INT8 GPU资格、C32 Sym与C32 Asym均 NOT_RUN / NOT_QUALIFIED**。当前FP实际结果由主报告及真实回执单独决定，本审计没有读取或假设FP通过。即使FP短程通过，也只触发有条件的兼容性结论，不授权新GPU单位。

核验当前407源清单SHA `48ec3e527b28cc1a1860a72559d0e9fafad96866c9d981f6b0c80d4bf5ebe19c` 中以下实际源码的SHA/bytes；没有导入Torch/模型，没有查询Driver/NVML，也没有执行Codec。七个历史C128共享函数/类AST逐项相同；具体收据与行号见 `analysis/int8_compatibility/INT8_COMPATIBILITY_SOURCE_AUDIT.json`。

1. **H+C128 hook与写回。** 旧runner76–113行在GDN core前对q/k按原规范归一化并右乘FP32 H128，返回core转换至value原dtype；151–167行读取实际cache state并以state.copy_原位写回QDQ。hadamard、normalize_qk、QwenKeyHadamardPatch、get_state、quantize_c128、load_model_and_tokenizer、setup_seed七项与formal C128 runner的AST相同。函数同源不等于完整调用协议同一；本轮恢复FP没有启用H或QDQ。

2. **首token / prefill量化时点。** 原AIME旧runner289–304行先执行整段prompt prefill，未调用quantize；首decode因此读取未QDQ的前缀state，之后每decode末QDQ。matched_worker115–117行仅OLD H在t>0时量化，256输入对应255次，NEW H从t0量化对应256次。此次恢复FP是1-token首chunk及后255次recurrent的固定输入回放；不可将其32-token数值资格当作真实多token prefill、INT8或81K数值资格。

3. **C128与C32的分组轴。** C128的state约定[B,H,K,V]=[1,32,128,128]；amax(dim=-2,keepdim=True)沿128个K元素求每V列最大值，scale为[B,H,1,V]，不是沿V的分组。当前恢复worker和matched FP单位未选择C32实现。若后续意图沿K划为4×32，reshape/group边界、scale形状及尾组合同必须另行冻结；不能将STEPQuant的V组32或attention KV缓存分组视为同一算法。

4. **Sym / Asym的scale与zero-point。** 已有C128是对称零点0：scale=abs(state).amax(K).clamp_min(1e-12)/127；torch.round后clamp[-127,127]，重建codes*scale。codes在该GPU路径中仍为浮点临时量；本审计不将其称为packed INT8。当前未选定C32 Sym/Asym源，不替用户决定affine码区间、min/max、zero-point dtype与clamp、舍入、常量组或scale floor；这些缺口须在新独立合同中明确后实际资格。

5. **下一次GDN是否消费写回状态。** 旧runner151–162行的get_state/state.copy_定位实际cache。原native modeling505–535行从cache.layers[layer].recurrent_states[0]取initial_state，传给recurrent helper，再更新cache；cache_utils1072–1086、1406–1425行将新状态copy_至同层实际存储。静态路径支持下一次消费写回状态；本次FP raw没有替INT8证明post-QDQ code/scale/dequant/pointer及next-consumption逐token身份。

6. **是否有persistent高精度shadow。** 选定旧C128函数没有额外保留未量化的recurrent shadow：source/codes/dequant为局部临时对象，持久cache自身为FP32 QDQ重建载体。它依旧占FP32 state存储，不是物理int8 cache。另存的CPU EncodedState源码会保留owned int8 codes与scales，但它是不同CPU codec，未接入此次GPU path；不可把其storage审计移植为当前GPU压缩证明。这里是静态限定结论，不是全运行期别名/活跃性穷举。

7. **同协议匹配FP基线。** 未来若比较FP Native、H+C128 Sym、H+C32 Sym、H+C32 Asym，应重新在同一个最终批准恢复协议下匹配source/运行库/GPU、输入与采样、prefill/GDN更新、QDQ插入时点、cache生命周期、dtype、打分位置和reducer。当前FP仅128warmup+128measure；原NEW_STRICT或整prompt AIME谱系不能直接代入。FP约13TPS即使实测达到，也不能推断任何INT8配置同速，更不授权AIME、32K Future-KL、Group32 GPU或训练。

源码收据（路径相对于本run/payload；完整SHA也保留在JSON）：

- `candidate/source_tree/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py`：26510 bytes；SHA256 `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`。
- `candidate/source_tree/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/formal_c128_runner.py`：34744 bytes；SHA256 `77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69`。
- `candidate/benchmark_replay/source/matched_worker.py`：12661 bytes；SHA256 `08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c`。
- `candidate/source_tree/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py`：88546 bytes；SHA256 `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`。
- `candidate/source_tree/data/zypan/transformers-qwen35/src/transformers/cache_utils.py`：103241 bytes；SHA256 `b63732985b520a62aa4b06fc2af8ca6e506912679e0f0479b706534d041fb1bd`。
- `candidate/source_tree/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/src/dual_axis_codec.py`：6234 bytes；SHA256 `cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6`。
- `worker/restoration_worker.py`：36644 bytes；SHA256 `1b709359fe4652a611814bc07648b8a42181259603917f29058e486ea86f9613`。
- `qualification/GPU_PLAN.json`：3993 bytes；SHA256 `ca84495337ca96c5c519ff043097b8911f33f57348cf01788f21a9d6a1905547`。
- `SOURCE_FREEZE.json`：115804 bytes；SHA256 `48ec3e527b28cc1a1860a72559d0e9fafad96866c9d981f6b0c80d4bf5ebe19c`。

此次审计只写新run的补充报告，原407、旧530/51封存、worker与CPU reducer均未修改。没有创建C32 Codec或新launcher；没有中间准确率/吞吐据此改计划。
