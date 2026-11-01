# 源码证据

| 实际路径 | HEAD | branch | dirty条目数 |
|---|---|---|---:|
| `/data/zypan/GDN-quantization` | `8f631858767c369829133d88c769f84019677c1e` | `research-sync-2026-09-02` | 188 |
| `/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2` | `a979a9f678f4ba0d6b9510a56760d36728929b8c` | `exp/qwen-gdn-c7-persistent-final-hidden-v1` | 157 |
| `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen` | `fa7918d82c144a30ba3459114abb53e70df73d98` | `exp/hadamard-init-dense-orthogonal-oracle-v1-qwen` | 1339 |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1` | `622df3543c85f72fe700046110a0da35ca214af7` | `qwen-gdn-functional-causal-replication-v1` | 157 |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3` | `c07f5efb9bf7cefae4906d927d78e3814ea15371` | `exp/qwen-gdn-deterministic-scale-discovery-v3` | 203 |

完整dirty porcelain、origin、Python/package metadata见SOURCE_IDENTITY_AUDIT.json。所有历史工作树原样保留，仅独立analyses目录写入。实际V3当前HEAD与历史contract记载的旧commit不同，因此按实际源码SHA和当前HEAD分别记录，不假定HEAD代表冻结数值函数身份。当前依赖内容与原正式manifest匹配。

| 当前读取文件 | SHA256 |
|---|---|
| `/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py` | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` |
| `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py` | `77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/numerical_recovery.py` | `b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/matched_history.py` | `e026487e2e2a35756d99d7bd9ffe9d0c0ac567aa156f94728e5a2f39d4468ace` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/scale_science.py` | `8e096bba13c354803f7ce40eb42b26c62b4112ce5d56814fd1d241192582a44a` |
| `/data/zypan/worktrees/qwen-gdn-fd-scale-feedback-direction-v1/experiments/QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1/scripts/common.py` | `d00fb2c23d2172bb6782100b4c4dd3538d57f2b8cfa2c0ae382487ed09441eed` |
| `/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1/scripts/formal_worker.py` | `c4e66288680bc0d2398a711097a16c824928ddca9057e5449cc13ca100314aca` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/replication_worker.py` | `ad7106a864729326e830c44ac982d15b0b9320ccfe9b7819bdf35e146827b3b6` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/NEW_NUMERICAL_CONTRACT.json` | `0031725d882b758ab59d292124b8604638f21a93a7e26a0bd60a74cf3716089a` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/QUALIFICATION_POLICY.json` | `83e64320564f05d3b3d7eb79656d2c3198f396489d9cbb32d39dba449a8a6ec1` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/MULTITOKEN_FULL_MODEL_PARITY.json` | `83d187fd66d150e02d1283f2c051cbea1093af47f941583e19bfcba1b0559be5` |
| `/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/FRESH_PROCESS_DETERMINISM.json` | `09ddb99a10f0cd6bca9aeb544ee6b72f07b953eb6485c35b72a25413a3d04f9a` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/SOURCE_AND_RUNTIME_IDENTITY.json` | `e3f5c43322d705899c18e9d172a7cb5476193b5192b09abab96b95b60b7870e6` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/V3_NUMERICAL_CLOSURE_REUSE.json` | `e8c4a136283764e7fc599c698b3ca9d3b065f37180a160929dc65c2f381adf8a` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/NUMERICAL_QUALIFICATION.json` | `70348fadab432f0dd02f51265ac36965728a6ed5529e06af498668bed154efb8` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/EXCLUSION_REGISTRY.json` | `4f902954e7916cd74ed63b4391bc6b9ce051c7e60405a1b067f6d35f5c1ce33c` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/DATASET_SELECTION_AUDIT.json` | `eb1874d083fcc8b024dbf870d960a0fd1d962df80d256f070a33eee736966f59` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/FROZEN_8DOC_SELECTION.json` | `eeb00f6197cea3b497a0269ebe5e6a2513efd9ee7a1b664ed13c4da6e3aad3d7` |
| `/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/DOCUMENT_INPUT_ARTIFACTS.json` | `a964ed7336a0871d8aac26e68eabf0d175e4c96160a5dfa367252e71951b2ca4` |
| `/data/zypan/modelscope_models/Qwen3.5-9B/tokenizer_config.json` | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `/data/zypan/modelscope_models/Qwen3.5-9B/config.json` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` |
| `/data/zypan/modelscope_models/Qwen3.5-9B/model.safetensors.index.json` | `26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5` |

CPU V1 seal清单SHA：d659b34a8d8f7dc6196192bcad9d2f95a0aad3d9b1237f58f2e501777c58b015；交付回执SHA：562c63ace8179fb9f519e09ec4953b138621f63a1148892d00b1716eb6689754。两端70文件、原失败完整合成输入panel的远端SHA均重新核验。继承V1源码不修改，新src/cpu_qualification.py只编排原tests/reference。

STEPQuant固定commit f2deac58131219e6aecff274f8eabe88e08d2e3c 的quantization.py SHA：0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69；通过V1 seal恢复源码并重新运行all8/ones/pivots0的9案例exact测试。没有取其他版本或从公式重写Dual-M。

Hadamard CPU FP32原始矩阵SHA：6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68。H.T@H的CPU maxabs偏差为5.960464477539063e-08，仅为矩阵身份/舍入记录，不是full-model功能等价证明。

Torch2.5.1+cu121、Transformers installed5.16.0.dev0、Tokenizers0.23.1、Triton3.1.0；FLA distribution metadata未发现，live GPU dispatch未运行。模型小配置和tokenizer文件内容SHA核验，权重4shard仅记录历史SHA+当前stat，本轮没有模型加载或新权重content rehash；未来GPU前需重新验证。详见runtime边界审计。

Ling排除ID来自本轮人类直接回复：pg19-test-41603/2544/7987/26183，冻结任务LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1，日期2026-10-14。原manifest路径未提供且未假称核验。人类明确允许用此集合完成排除；详见history/HUMAN_LING_EXCLUSION_ADDENDUM.json。
