# 本次共享推理路径冻结说明

本次使用原 `90d929...` 模型源码实际 Torch native chunk/recurrent fallback，不替换 `g.cumsum`，不使用历史32K工程的 compensated scan。加载器同时检查实际装饰器闭包的 `is_new_implementation=False`、`implementation` 与原函数一致及原源 SHA；不能只凭 `inspect.unwrap` 的文件名证明实际后端。CPU 导入检查已经验证该闭包，完整模型实际 GPU 调用仍需资格。

历史五文档 H+Dual-M 使用了其已资格 compensated chunk 绑定；本次只复用其原 H key 坐标变换、Dual-M codec、安全检查和 writeback 定义，不把两次完整模型历史称作相同。两种新方法共同使用原 AIME native GDN 数学。保留历史 native/H 数值不等价失败。

共同 flags 事前固定为 `deterministic_algorithms=False`、`cudnn.benchmark=False`、`cudnn.deterministic=False`、CUDA matmul/cudnn TF32 均 False、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。保留原 native 数学的原因包括：当前 Torch2.5.1 官方函数文档将 CUDA floating `cumsum` 列在 strict-deterministic 模式的不可用操作中。本次不以更换 scan 绕过这个接口。128-token fresh-process 逐字节重复是实测门禁；False 不代表所有 CUDA 运算普遍确定，失配即真实 STOP。

正式随机采样器、chat template、EOS 与 top-k ties 来自 `9ff56d...` 原始 AST，相关节点与 `77b1e1...` 逐 AST 相同。独立生成器、mask、cache 和钩子每请求新建，返回后移除。首个 batched prompt prefill 后立即作本方法 QDQ，当前 logits 不回算；每个 decode 后再次 QDQ，下一原 kernel 必须消费写回对象和 storage。正式每 token只检查对象、data_ptr、可用 version 和计数；canary 另外核对每次字节 SHA。

H+Dual-M 的原参考 codes、FP16 scales 与重构按旧冻结检查；STEP 官方 codes、FP16 scales、pivots 必须精确，`StateCodec.decode` 与 `fit_state` 重构使用官方 `tests/test_core.py:59` 既定 `atol=rtol=1e-6`，并保留最大绝对误差、不等数量和 byte-equal 原值。这个容差在任何 GPU 结果前固定，fresh-process 主路径仍要求逐字节相等。

资源资格保留真实81920旧 HF/manual 长生成源证据，新增活模型下的全长 BF16 KV、cat overlap、16-query-head workspace 与真实单 token Torch SDPA 调用；事前要求 CUDA driver 至少1GiB free。该测试是 synthetic 长KV工程压力资格，不能称为两个新方法81920-token GDN科学数值等价资格。

两方法实际持久 recurrent cache 均为48MiB FP32 tensor，承载各自重构 QDQ 值，不保留未量化 FP32 shadow。STEP packed codes/FP16 pivots/scales 仅在每次编码时暂存，逐层记录真实 bytes，不能宣称持久 physical packed 存储节省。

源码审查阶段的 `4fc336...` 旧草案及其 CPU 回执保存在 `provenance/pre_review_runtime_sources`，该草案没有执行 GPU。最终源与 CPU 回执另行绑定，不能混称旧草案 GPU 已合格。
