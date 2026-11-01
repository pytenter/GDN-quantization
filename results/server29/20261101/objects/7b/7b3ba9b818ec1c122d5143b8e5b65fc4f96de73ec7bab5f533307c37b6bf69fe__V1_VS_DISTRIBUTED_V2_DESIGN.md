# V1 topology versus distributed V2 design

V1's preserved conclusion is `FORWARD_EXACT_GATE=PASS`, `TRAJECTORY_ENVELOPE_GATE=FAIL`, `FIXED_NON_AIME_ENDPOINT_PROBE=FAIL`, `NEXT_STAGE_AUTHORIZED=NO`. V2 asks a different question: whether a separately defined distributed runtime can be mathematically valid, reproducible within its own topology, gradient-correct, memory-feasible, and checkpoint-portable. It does not re-score V1 or require PP2 trajectories to fit the old single-GPU numerical envelope.

| Property | Old V1 dual-GPU implementation | Proposed V2 PP2 |
| --- | --- | --- |
| Processes | One Python process | Two ranks, one process per GPU |
| Partition | HF `device_map`, contiguous complete decoder blocks | Explicit contiguous complete-block pipeline stages |
| Cross-stage transfer | Implicit cross-device autograd in one process | Explicit NCCL send/recv for hidden activations and their gradients |
| Tensor/GDN partition | No TP; each GDN block wholly on one GPU | No TP initially; each GDN block wholly on one rank |
| FSDP/ZeRO/offload/checkpointing | None | None in initial PP2 stage |
| Scheduling | Implicit sequential model call | One microbatch: rank0 forward, send, rank1 forward/backward, send gradient, rank0 backward; no overlap |
| Validity gate | Single-versus-dual trajectory envelope (failed) | Same-topology repeatability, numerical gradient sanity, checkpoint portability, stability |

The existing V1 topology inventory records candidate complete-block splits at blocks 13, 14, 16, 17 and 18. Split 16 balances 12 GDN blocks and approximately 8.954 GB of static weights on each side; it is only a **candidate**, not a final PP2 choice. V2 must measure rank-local static bytes and activation peaks before freezing its split. The original server's GPU0/1 pair is PHB-connected, has no active NVLink or direct PyTorch P2P, but a short bounded NCCL test completed; raw data are in `analysis/p2p_bandwidth.json`.

The old runtime's recurrent training code is in the read-only V1 experiment and uses a patch around canonical Qwen3.5 Torch recurrent/chunk functions plus differentiable post-QDQ recurrent writeback. PP2 must retain the same function/state/QDQ timing per complete block, while adapting capture and losses to rank-local layer sets. Loading the entire 9B model on each 24GB GPU before partition is not acceptable; stage-local checkpoint loading or equivalent bounded construction must be audited. Nothing in this design report authorizes formal C5/C6 training or AIME generation.
