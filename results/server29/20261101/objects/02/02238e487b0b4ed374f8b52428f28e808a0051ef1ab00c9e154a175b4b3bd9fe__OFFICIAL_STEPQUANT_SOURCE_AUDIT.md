# 完整 STEPQuant@6 源码与 Qwen3.5-9B 校准审计

官方数学实现可复用，当前 9B 完整 Plan 尚未生成，`STEPQUANT_PLAN_QUALIFICATION = NOT_QUALIFIED`。这表示必须先完成已授权的非 AIME 校准；不是把缺少旧 Plan 当作数学失败，也不能用已有 all8、w=1、pivots=0 的 Dual-M 代替。

固定官方 commit 为 `f2deac58131219e6aecff274f8eabe88e08d2e3c`，独立只读源码在 `official_source/STEPQuant_f2deac58`。125 个 tracked 文件共 1,359,337 bytes 已逐文件绑定 SHA，并在 55 fresh 核验。GitHub 当前 main 为 `7e0f2318da7de435cf0deca1deb02aa07ee306c0`；通过官方 API 完整 tree 与本地固定 commit 的逐 blob 对照，差异只有 README 和两个媒体文件，算法、校准、配置及测试的 blob 全部相同。原始 API 和逐文件差异证明保存在 `official_source/UPSTREAM_TREE_AND_COMPARE_RAW.json` 与 `OFFICIAL_COMMIT_IDENTITY.json`。新网络 clone 曾因 HTTP2 framing 错误失败，随后从已存在的完整官方 Git 缓存建立独立副本，不修改旧缓存，也没有执行下载的 shell/CLI 脚本。

完整 @6 的定义来自 [core.py](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/core.py)、[calibration.py](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/calibration.py)、[allocation.py](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/allocation.py) 和 [quantization.py](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/quantization.py)：

- row impact 为 squared transported read，使用 post-convolution、原生归一化及 `1/sqrt(K)` 缩放后的 query、key、log-decay、beta；`gamma=0.25`，impact factor 在每个 head 的 key 轴上作几何均值归一化。
- lifetime 为 horizon2048 的有限几何和，基于每 head 的平均 log-decay；不是临时自造的寿命阈值。
- GDN 的预算单位是整个 head，@6 候选为 4/6/8 位。先做 exact global DP，按已分配残余风险稳定排序，保护默认 32 个完整 head 为 FP16，再对剩余 head 做 residual DP。9B 有 24×32=768 heads，integer budget 为 `6×768−8×32=4352`。名义预算将 pivot 按 INT8 计费，实际 FP16 pivot 与 scales 必须另计。
- row factor、value-column factor 均按实际保存的 FP16 数值拟合，floor 为 `2^-24`，上限65504；进行一次 key-row-weighted column LS，再 final nearest-level re-encode。权重为实际 impact 的平方，不能替换为1。4/6/8 位分别使用对应 signed levels 与原 rounding/clamp；pivot 存 FP16。
- 官方 StateCodec 使用真实 bit pack 和 FP16 pivot/scales。新正式运行时按共同约定使用 FP32 tensor 承载真实 QDQ 值，PackedState 只作为瞬时 codec 载荷/成本核算；不得称作 persistent physical packed state。

官方 [paper calibration 配置](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/configs/calibration/paper.json) 与 [calibration_segments](https://github.com/Dreamer-Toby/STEPQuant/blob/f2deac58131219e6aecff274f8eabe88e08d2e3c/stepquant/cli.py) 固定完整 WikiText-2-raw train，revision `b08601e04326c79dfdd32d625aee71d232d685c3`，全部36718行以双换行连接、无 special tokens；用 `torch.Generator(seed=0)` 在完整 token stream 上有放回抽32个连续2048窗口。sample interval8、reservoir64、horizon2048，segment 之间 cache 清空，校准期间没有量化反馈。旧 C6R 的96篇选定 raw 文档不是这个完整 source，未被冒用。

数据在 55 和 Mac 原下载地址分别遇到网络不可达/超时。已从同 revision 镜像取 6,357,543 bytes，并与 [Hugging Face 官方文件页](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet) 公布 SHA256 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` 精确比对，随后 55 独立 fresh SHA 和 parquet 读取通过。新 CPU 准备只将官方函数的数据 loader 绑定到该完整、已校验 parquet，没有改官方函数体；与独立 seed0 窗口重构32/32 exact。实际完整训练 stream 为2,505,827 tokens，32窗口共65536 tokens，合并 token SHA 为 `bc1ecdf59dbfe2279056d4ac59b0a8b7b5977ad1df8591b2e18fe89290a3b255`。具体窗口、每窗 token SHA、tokenizer 身份见 `CALIBRATION_INPUTS.json` 与 JSONL。全文 CR 数为0，AIME 未参与。

9B 结构为24 GDN layers、每层32 value heads、key/value dim128，原 state 为 `[1,32,128,128]` FP32。官方 HF adapter 针对另一版 Transformers：它在 GDN 实例上读取 kernel 属性，并把 cache 当 tensor；本项目冻结 `90d929…` 源则调用 qmod 全局 kernel，cache 为 `recurrent_states[0]`。因此不直接运行官方 ModelAdapter 或其 greedy generate，不切换 SGLang，也不重写 GDN kernel。

已获授权的最小 I/O 适配是 `NativeShadowStatistics`：原 native kernel 是实际模型输出/cache 的唯一来源；旁路用未改官方 RecurrentKernel/LayerStatistics 收集 FP32 shadow 统计。先严格执行 native l2norm 的原输入 dtype 数值，再向官方 kernel 传 `use_qk_l2norm_in_kernel=False`，由其只做一次 query 缩放，避免重复 norm/scale。shadow 不回写 logits、真实 initial state 或 cache。它的中间状态是官方 FP32 shadow 参考，不能声称与 native chunk 每8token中间状态逐字节相同。

CPU 证据已成立：12个 FP32/BF16、9/64/128 token、recurrent/chunk 合同检查中，shadow 开关不改变原 native 返回值与 initial state，shadow omega/log-decay/reservoir 与原官方 observer exact。原 recurrent/reference 断言沿用官方 `atol=1e-6, rtol=1e-5`；codec/fit 沿用 `1e-6/1e-6`，FP16 pivot exact。chunk/BF16 与 shadow 自身存在描述性差异（最大输出 `3.0517578125e-5`，final-state `1.1920928955078125e-6`），该差异没有被伪称为 byte parity。另因环境没有 pytest，保留首次 runner 缺失记录后，不安装环境，直接调用去除 pytest 参数装饰器的原 test_core 函数体，原断言与容差不变，32个官方参数组合均 PASS。

下一步仅由根协调器启动同一 `shared_runtime_worker.py --mode calibration`、同一冻结模型加载器，在一张已绑定 UUID 的3090上完成32真实 segments，并在 CPU 使用未改官方 `calibrate` 拟合完整 Plan。每 segment 写进度；真实统计约3GiB保留55，Plan及24层映射/bit/pivot/impact/成本摘要在真正生成后保存。未完成前没有9B Plan SHA、precision map、pivot mask或准确率数值。Plan 合格后仍须完成四卡 canary、资源与81920长度资格，再进入正式120条生成；本审计没有执行 GPU 或正式生成。
