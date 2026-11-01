# C6R 完成训练来源审计

TRAINING_PROVENANCE_GATE = PASS。此结论只说明真正完成的 A800 C6R 训练身份、数学核心、配置和所选冻结导出已恢复；不表示新的可运输训练入口已经实现或本轮训练获授权。

权威任务为 `QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1`。其冻结代码 commit 为 `ad1b9fc19bd5fd5cc9014b7473f31edce22976a3`。该 commit 对象在 `.55:/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1` 实际存在；`git show` 的 `scripts/a800_matched.py` blob SHA 与运行文件、trusted environment freeze、原 A800 preregistration 三方一致，均为 `0f85e40100070c051a01d05125341c1863b5c9808247803a725b1b218f54412d`。最早检查的 mechanism worktree 无该对象，不应将该局部缺失推广为所有仓库均无原 commit。

原训练实际完整配置从以下已封存 A800 文件恢复：

`/data/zypan/worktrees/qwen-gdn-c6r-c7-4x3090-deployment-v1/experiments/QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1/source_evidence/protocol/preregistration.json`

其 SHA 为 `009279fb84dd6d45f73bdfc27629a6d28de8d1a6dbfab64b65c83e9c4de5f3c8`。不是用早期 125-update 线索代替实际训练证据：原 A800 stop evidence 中真实 C6R summary 为 PASS，train_log 有完整 update1..125；125 次实际顺序逐项等于该 preregistration 的125次 seed0有放回抽样顺序。

## 已完成的真实训练合同

- 64 TRAIN、16 VALIDATION；原 corpus 另有16 HELDOUT，共96篇 WikiText-2 raw，全部已属于历史方法开发范围。corpus SHA `37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0`。
- 每 update 使用一个文档的前1024 tokens，batch1；固定8个 capture，位置为 `sorted(Random(int(raw_text_sha[:16],16) XOR 20260921).sample(range(128,1024),8))`。125 optimizer updates，1000 capture exposures，128000 student tokens。
- Hadamard初始化：每层theta全0，Delta=I，实际 q/k 依次乘 H128、Delta。24个 GDN 层各8128个FP32 theta，共195072参数，head-shared。模型9B权重冻结BF16。
- Cayley：上三角theta/下三角负theta组成skew，`Delta=(I-A)(I+A)^-1`，使用线性求解、不显式inverse。每步正交误差门限1e-4，并检查finite/proper。
- Adam lr0.001、weight_decay0；精确原源码未覆盖Torch2.5.1默认betas=(0.9,0.999)、eps=1e-8。global-norm clip1；灾难梯度阈值1e6。
- H64：每64 tokens若有capture loss则backward，所有64-token边界detach recurrent/conv/KV图；数值INT8-QDQ state连续，不reset clean FP。完整1024-token文档后才一次optimizer.step，跨16个图段累积梯度。无activation checkpointing。

## 数学核心原字节

`/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py`

SHA `48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e`。

`/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py`

SHA `62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067`。

`/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py`

SHA `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`。

均逐文件fresh SHA等于原 A800 source manifest。原base runner另为 `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`。

Functional loss为每个capture的24个GDN `linear_attn.out_proj`完整输出的relative sum-squared MSE，24层等权。真实模块执行GDN readout→RMSNorm（先norm再gate）→out_proj，因此本loss包含norm、gate和out_proj的真实功能坐标；不是Final-Hidden objective，也不是只对裸query readout取MSE。

State辅助项使用当前token更新前的student initial recurrent state，依次经 `einsum(Delta,state)`、`einsum(H,state)`恢复原坐标，与clean teacher previous state对比；先每B/head矩阵sumSq error/sumSq reference，再mean，24层等权。epsilon1e-12。总目标 `Functional+0.1*State`，每文档8 captures等权。

Teacher为原规范clean FP轨迹，按chunk-to-capture采集24 previous states、24 full out_proj输出；统一teacher另含一个normalized lm_head-input hidden供C7/secondary使用。A800真实新建80文档/640captures/31360 tensor manifest rows，并有24 captures/family在线与缓存exact parity。此审计未重新加载这些历史大teacher tensor。

Student每个文档从cache=None/token0建立自身历史，q/k采用H@Delta，更新state执行原C128 QDQ再回写，下一token消费该state。C128按K轴[-2]分组、scale[B,H,1,V]，FP32 absmax/127、clamp_min1e-12、ties-even round、clamp[-127,127]。原STE为 `value+(exact_dequant-value).detach()`，state梯度identity、scale/round/clip反向路径detach。该表达式存在FP32消减差异的潜在逐位问题，必须由本轮CPU parity独立衡量；不能只凭其docstring声称hard forward逐位一致。

## 选择与冻结C6R身份

20个eligible validation updates为 `[7,13,19,25,32,38,44,50,57,63,69,75,82,88,94,100,107,113,119,125]`。update0只是reference。选择16doc×8captures平均纯Functional validation最小（不含0.1 State辅助项），完全相等时更早update胜出，AIME未用于选择。

实际选中update94/exposure752，Functional validation `0.048934229369479`。原theta checkpoint SHA `620a35101b95cadd16f50cd1b869aeef54da9db0b3c76076fe82a374a6ee17cf`；冻结FP32 Delta SHA `a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6`，deployment manifest SHA `f365dbab44e31b127abc5f9cc37916e432c4f3c88cc7706c82886812686d1103`。当前所选export可读并fresh SHA一致；历史export-parity记录逐层加载原theta后与frozen Delta exact相等。原theta文件当前可读位置未定位，不冒充已重新读取；新两臂Hadamard初始化不需要以该theta恢复。

新配对A/D计划的统一final125选择规则与历史C6R minimum-validation-selected94不同，必须单独预注册。冻结旧C6R仍保持原selected94身份，不得重选或更改。

## 资源证据与原控制层缺口

历史A800 C6R gradient_smoke64 peak allocated `31035885568` bytes / reserved `31054626816` bytes，均约28.9GiB，已超过24GB3090。三次完整1024/H64 readiness peak allocated分别 `32532857856,32429154816,32493552640`，reserved分别 `33441185792,33338425344,33380368384` bytes。不得为了本轮资格故意在3090运行原完整9B图造成OOM；代表性局部状态资格不能写成完整模型训练PASS。

125个train_document实际runtime合计 `12365.375844005495` 秒（约3.435h）；含20轮Shared16验证的train()为 `33532.51159002632` 秒（约9.315h），完整phase `33716.35395746678` 秒。正式最大peak allocated `32629188096` / reserved `33527169024` bytes。不保证未来两臂8付费小时可完成，需未来A800短admission实测和完整成本模型。

原wrapper通过历史绝对A800路径导入的 `formal_pipeline.py`、`fulldoc_common.py`、`c7_pipeline.py` 未在本轮已检查的当前55源树中找到原文件。缺失控制辅助包括F原子JSON/torch保存、file SHA、fresh_bank及init校验、append_jsonl、memory_snapshot、cpu_state_dict、内存与灾难梯度阈值；M的seed/clean/HiddenCapture/C7 hidden secondary包装。科学核心Loss、C128、Cayley、recurrence模型原字节和完整实际训练配置已恢复，因此来源门禁可PASS；但不能称旧absolute-path launcher可原封不动启动。新的单一可运输entry必须实际包含或替代这些控制依赖，逐项源码差异审计并CPU自测，不能以stub/dry-run外壳代替。

原candidate checkpoint格式仅condition/bank/update/exposure/validation/preregSHA，没有optimizer和RNG。新任务合法边界恢复必须明确保存/恢复optimizer、rotation、RNG及步数/数据游标，不得称旧格式已具完整resume合同。

本审计仅CPU读取/源码与文件hash，无GPU执行、训练、历史文件修改、A800连接或付费资源操作。完整原样小文件快照、实际hash及逐项检查见 `FROZEN_C6R_PROVENANCE.json` 和 `provenance/snapshots/`。本轮新Dual-M数学、梯度、数据与资源门禁尚应由其独立真实证据判定；来源PASS不等于完整readiness。
