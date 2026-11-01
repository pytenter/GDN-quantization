当前仅CPU静态兼容性审计；INT8 GPU资格 NOT_RUN / NOT_QUALIFIED。

旧runner的Hadamard、normalize_qk、Patch、get_state、quantize_c128、load_model、setup_seed七项函数AST与已有新formal C128 runner相同，但调用协议不同。旋转在GDN core前作用于q/k，旧实现按需构建FP32 H128。C128沿K维amax，scale形状[B,32,1,128]，scale=amax(abs(state)).clamp_min(1e-12)/127；round/clamp codes为FP32临时量，cache持久状态是FP32 QDQ重建载体，不能称物理packed INT8。

历史OLD matched H在t0首prefill不QDQ，其后共255次；NEW从t0即QDQ共256次。原AIME又是整段prompt单轮prefill，不量化prefill，首decode消费未QDQ前缀，之后每decode末QDQ。即使后128计时窗相同，warmup状态谱系已不同，旧新KL不能作为严格匹配组。

旧quantize函数将重建值写回实际cache；静态未见额外未量化persistent shadow。历史没有充分的逐tokenpointer/bytes消费回执，不能把静态代码解释成已通过当前动态消费门禁。

H+C128具备可复用源码基础；C32 Sym/Asym具备结构上的接入位置，但各自轴、group边界、scale/zero-point、饱和/舍入、prefill注入与存储合同仍需独立冻结并实际资格。本轮不新写这些Codec、不修改原Codec迁就速度，不运行它们。

未来FP Native、H+C128 Sym、H+C32 Sym、H+C32 Asym必须全部在同一个最终批准的恢复协议下重建基线：模型/输入、旋转、prefill/GDN更新、QDQ注入、cache生命周期、sampling、Future-KL评分器、dtype和设备身份一致。不得直接代入历史NEW_STRICT、旧AIME或不同OLD谱系的结果。FP约13TPS不能推断H或其他INT8相同速度；历史OLD H+C128约9.065TPS也仅为历史参考。

本轮可提供源码和协议差异，不能批准正式Group32、AIME或长Future-KL。即使下一阶段FP短程通过，INT8未跑时仍只能标 FP_ONLY_QUALIFIED_INT8_NOT_RUN，且适用范围限于实际已测的FP短程协议。
