# QWEN_GDN_SCALE_SURROGATE_INPUT_RECOVERY_AND_VALIDATION_V2

## 结论

PRIMARY_STATUS = ENGINEERING_OR_NUMERICAL_STOP；FINAL_ACTION = FAIL_AND_STOP。
第一项新运行门禁 CANONICAL_DETERMINISTIC_CAPTURE_ENTRY = FAIL。
V2已取消4 GPUh硬上限，没有85%显存门槛；本次停止与预算、磁盘不足均无关。

实际错误：`cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.`。
完整原始 traceback 保存在 execution/FIRST_FAILURE.json；没有重试、关闭 deterministic、安装替代 scan 或修改科学配置。

## 1. V1缺少哪些输入？

四文档正式32768-token drive与8192学生锚点张量已按历史release释放。剩余SHA记录不是张量。clean FP端点、旧连续G及其原始记录、Doc1工程256/2048 drive仍存在。自然Parent的逐token paired scales、out_proj matched teacher输出和精确first-onset不在可复用资产中。

## 2. 找到可复用历史anchor/drive了吗？

复用V1的96项身份PASS和Direction269项审计；只新增检查本次使用的源码、真实Doc1工程首shard和历史cumsum闭环封存文件。Doc1工程2048可供资格测试，不是8192-anchor或8192后短窗。没有找到新的正式8192学生锚点/drive备份。

## 3. 如何处理历史确定性差异？

历史capture deterministic=False、replay=True明确保留。新进程 deterministic=True、warn_only=False、TF32=False、CUBLAS_WORKSPACE_CONFIG=:4096:8，Torch2.5.1+cu121/CUDA12.1未切换。对真实Doc1 token1/layer0输入调用原生产Torch chunk绑定；原CUDA cumsum入口失败。
历史fixed-order替代曾通过operator/chunk，single-token historical32有145/145对象字节一致，但multi-token dev256只有12/145对象PASS；logits max_abs=0.375、relative_L2=0.009980417299255454，全模型资格FAIL，fresh-process determinism未执行。不能把局部PASS转成整体PASS或直接用于原C5/C6多token teacher collector。

## 4. 历史输入重建是否合格？

否。原生产deterministic入口已失败，尚未进入H/C6R重建、8192累计J或状态身份比较。单个端点J接近不能证明整个旧trajectory等价。没有将新短程信号与旧G正式配对。

## 5. 新匹配面板是否建立？

NOT_STARTED。PATH B也需要科学资格合格的deterministic capture和真实teacher forward。不能借新面板名义绕过同一未合格源码/数值门禁。

## 6. Track B是否完成？

NOT_RUN。原helper collect_teacher_targets使用chunk-to-position teacher forward，原训练8-capture、1024-token语义不得静默改为逐token FP teacher。没有用WikiText targets冒充RepoBench配对，也没有用q/k/v驱动近似B1。

## 7. Track A是否完成？

NOT_RUN。8192-anchor之后的H32/H64窗口索引已明确记录：0-based [8192,8224)、[8192,8256)，对应1-based8193..8224、8193..8256。没有人工reset、学生轨迹或短窗测量。

## 8. 真实S1/S2分数？

H32/H64 S1/S2/D1/D2/B0/B1/B2全部NOT_MEASURED。定义和epsilon=1e-12记录在FROZEN_SURROGATE_SPEC.json，但执行资格未建立；没有空壳TRACK_A/B CSV，没有编造排名或相关性。

## 9. 自然旋转能区分吗？

新surrogate未测，不能回答。旧G已有Doc1/2 H<C6R<C7；Doc3 C6R<C7<H；Doc4 H<C7<C6R。这仅是历史Discovery背景，不是新预测成功。

## 10. S1能预测合法长期G吗？

NOT_EVALUATED。旧G定义保持1024..T各token与24层等权FP64累计，G=J32768/J8192；没有新matched G。短窗与旧G numerator重叠，不能宣称严格未来预测。

## 11. S1相对原C6/B2有增量吗？

NOT_EVALUATED；不能报有增量，也不能报NO_ELIGIBLE_INCREMENTAL_DISCORDANCE。B0仍是per-head matrix relative-MSE且分母clamp_min；B1必须是真实out_proj/teacher global relative-MSE；B2=B1+0.1B0。没有更改这些定义。

## 12. pre-onset资格？

NO。没有自然精确first-onset和新合法future-only边界观测。

## 13. 实际资源消耗？

新增正式science GPUh=0。执行一次GPU0工程入口检查，进程wall-clock（含导入）4.776418秒；保守GPU-process wall-hours上界0.001326783，不把它伪装成CUDA kernel activity精确积分。
峰值 allocated=5297152 bytes，reserved=6291456 bytes；没有加载9B模型、正式token吞吐或大型临时drive，新drive=0 bytes。封存前小型产物379161 bytes，结束/data可用49641897984 bytes。最终交付字节数见RESULT_AND_DELIVERY_HASHES.json。

## 14. 是否值得进入独立full-model persistent KL？

本轮没有新科学依据支持进入，且用户未授权该pilot，因此NOT_RUN。
下一步所需的是单独解决并资格认证原multi-token teacher/capture的deterministic numerical closure（源码、parity和fresh-process determinism均须合格），或用户另行明确批准一种新的确定性前向语义与独立matched面板资格协议。不能在本轮静默替换teacher forward、放宽旧caps、修改quantizer或重新解释旧FAIL。恢复后才继续本V2尚未完成的H/C6R最小重建与Track B/A Discovery。

## 复现和身份

Gate source commit：f6003ff16a8c081a44ccca70796c77f16e51f86a。
仅在独立新输出目录中执行本脚本prepare → git freeze → probe；原目录write-once，不能重复probe。
环境：CUDA_VISIBLE_DEVICES=0，CUBLAS_WORKSPACE_CONFIG=:4096:8；Python /data/ydai/miniconda3/envs/bitdecode/bin/python。
GPU0 UUID：GPU-a78176ed-f97f-c54f-6201-5a01b570f422。
旧source/checkpoint/rotation/文档及旧COMPLETE/PASS/FAIL均未改写。科学单位0，训练0，AIME0，自由生成0。
