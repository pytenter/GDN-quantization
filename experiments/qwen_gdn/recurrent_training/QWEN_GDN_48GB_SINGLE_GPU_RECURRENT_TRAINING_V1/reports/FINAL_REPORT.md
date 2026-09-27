# QWEN_GDN_48GB_SINGLE_GPU_RECURRENT_TRAINING_V1 — final report

**Outcome:** the 48GB single-vGPU fixed-sample H32 recurrent C5 memory diagnostic passed, including five sustained updates without observed memory creep. The strict cross-host rotation-matrix-hash portability gate failed despite exact checkpoint/theta/source hashes. Consequently formal recurrent training is **not ready** and was **not started**.

| Item | Result |
| --- | --- |
| vGPU hostname / GPU | `autodl-container-a88f4a8a56-14ae8f67` / `NVIDIA vGPU-48GB` |
| Visible CUDA devices / VRAM | 1 / 47.3819 GiB |
| Driver / Torch / CUDA runtime | 570.124.04 / 2.5.1+cu121 / 12.1 |
| Transformers / Triton / FLA | 5.16.0.dev0 / 3.1.0 / absent on both hosts |
| Data disk / free after model | `/root/autodl-tmp` / approximately 31.1 GiB |
| Model copy | one copy, 19,329,394,438 bytes |
| Source and model-index/config SHA parity | PASS |
| H1 allocated / reserved | 17.318 / 17.570 GiB, PASS |
| H4 allocated / reserved | 17.785 / 18.029 GiB, PASS |
| H8 allocated / reserved | 18.422 / 18.668 GiB, PASS |
| H16 allocated / reserved | 19.731 / 19.992 GiB, PASS |
| H32 allocated / reserved | 22.634 / 22.660 GiB, PASS |
| H32 one update / five sustained updates | PASS / PASS |
| Observed memory leak / largest stable diagnostic horizon | NO in eight updates / H32 |
| 48GB_H32_TRAINING_FEASIBLE | YES for the preregistered 32-token fixed-sample diagnostic only |
| Rotation checkpoint / SHA-256 | `/root/autodl-tmp/qwen_48gb_rotation_h32_8update_v1.pt` / `554b59891cc3c5b37a48203fd553e3992e6b7759f5867080ea377e165377f228` |
| Copied to original server | YES, `/tmp/qwen_48gb_rotation_h32_8update_v1.pt` |
| Canonical checkpoint portability | FAIL: 24/24 CPU rotation-matrix exact hashes differ; theta/source/checkpoint hashes exact |
| FORMAL_RECURRENT_TRAINING_READY | NO |
| Formal C5/C6 started / AIME on vGPU | NO / NO |
| Original canonical inference modified / other tasks interrupted | NO / NO |

The matrix mismatch is numerically small (max absolute 4.768e-7, relative L2 1.093e-7), but the task required exact hashes. No tolerance was added after inspection. Because that gate failed, the original-server fixed checkpoint forward/QDQ/writeback portability subchecks remain `NOT_RUN`. This is a stop-and-report result, not authorization to start formal C5/C6. The 32-token feasibility result must not be extrapolated without qualification to a complete 1024-token document.

Evidence: `analysis/final_verdict.json`, `analysis/forward_parity.json`, `analysis/H*.json`, `analysis/H32_stability.json`, `analysis/checkpoint_portability.json`, `analysis/rotation_matrix_numeric_diagnostic.json`, and `hashes/artifact_sha256.txt`. The full environment and package deltas are in `configs/`. No passwords, model weights, raw recurrent traces, or rotation binary are committed.
