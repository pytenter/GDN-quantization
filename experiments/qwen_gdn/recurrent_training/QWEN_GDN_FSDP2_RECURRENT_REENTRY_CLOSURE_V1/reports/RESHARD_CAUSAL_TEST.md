# Reshard single-factor intervention

In the same reduced canonical N4 model, both `reshard_after_forward=True` and `False` passed backward. The three-block nested stack peaked at 2,117,376,000 allocated bytes with reshard=True and 2,907,156,480 with reshard=False on rank 0: retaining full parameters increased peak allocation by 789,780,480 bytes (about 37%). This shows a memory cost without a correctness rescue in the tested reduced graph.

The preregistered positive signal (N1 True pass, N4 True parent-style fail, N4 False pass) did not occur. `RESHARD_REENTRY_CAUSAL_SIGNAL=NO_IN_TESTED_REDUCED_CONDITIONS`; parent H4 causality remains unresolved. Root-only/grouped wrapping and full 9B H4/no-reshard were not run because the reduced signal was absent, no legal fix existed, and full-model parameter retention would risk the 24GB memory budget. This is not a proof that PyTorch 2.5.1 is bug-free or that parent H4 can train.
