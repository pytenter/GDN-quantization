# Frozen source manifest — Server A / Qwen

- Task: `GDN_KDA_AIME26_REFERENCE_PATH_CLOSURE_V1`
- Timestamp (UTC): `2026-09-17T05:29:38.104744+00:00`
- Experiment Git HEAD: `8f631858767c369829133d88c769f84019677c1e`
- Branch: `experiment/aime26-sglang-rotation-v1`
- Isolated tree: `/data/zypan/worktrees/aime26-sglang-rotation-v1`
- Isolation mechanism: `git clone --shared (Git 1.8.3.1 lacks git worktree)`
- All copied/tracked SHA256 values equal originals: **YES**

> Infrastructure note: Server A provides Git 1.8.3.1, which predates `git worktree`. A shared-object local clone is used as the closest strictly isolated equivalent; the original working tree is untouched.

| Role | Original | Original SHA256 | Snapshot/tracked path | Copied SHA256 | Equal |
|---|---|---|---|---|---|
| external canonical recurrent-state quantizer | `/data/zypan/experiments/qwen35_gdn_quant/state_fake_quant.py` | `8fcb9e481d8811b88745069c45c685c66fd14cd41a0259eb9e58ab1ca44ed999` | `reference_snapshot/qwen/state_fake_quant.py` | `8fcb9e481d8811b88745069c45c685c66fd14cd41a0259eb9e58ab1ca44ed999` | YES |
| external Qwen model runtime implementation | `/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py` | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` | `reference_snapshot/qwen/modeling_qwen3_5.py` | `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a` | YES |
| external Qwen model configuration implementation | `/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/configuration_qwen3_5.py` | `3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf` | `reference_snapshot/qwen/configuration_qwen3_5.py` | `3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf` | YES |
| external Qwen modular source | `/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modular_qwen3_5.py` | `ca692bd333089c2d7c74194c24015a0a2f32ab06ac8391905f1524d30ce11809` | `reference_snapshot/qwen/modular_qwen3_5.py` | `ca692bd333089c2d7c74194c24015a0a2f32ab06ac8391905f1524d30ce11809` | YES |
| external Qwen model configuration | `/data/zypan/modelscope_models/Qwen3.5-9B/config.json` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` | `reference_snapshot/qwen/model_config.json` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` | YES |
| tracked canonical historical runner | `/data/zypan/GDN-quantization/experiments/orientation/run_end2end_bit_axis_screening.py` | `106c0c21aa080c34b03d1a094887f29da659ca9788b7b2586f38f6a095c0ab87` | `experiments/orientation/run_end2end_bit_axis_screening.py` | `106c0c21aa080c34b03d1a094887f29da659ca9788b7b2586f38f6a095c0ab87` | YES |
