# Ling L6/L7 new-member handoff

The handoff separates immutable code/provenance from binary artifacts:

- GitHub branch `handoff/ling-l6-l7-reproduction-v1` contains the trainer, exact runtime snapshot, configs, manifests, verification tools, and reproduction instructions.
- Hugging Face dataset `pytenter/ling-kda-artifacts` contains the canonical L6/L7 checkpoints, final rotations, and training metadata.
- The official model is `inclusionAI/Ling-3.0-tiny` at exact revision `e3a47d5b986e7141b6efd62597d598ebb392060d`.

## Route A: direct evaluation

Download and verify the four binaries with the supplied scripts, load either final rotation without re-saving it, and run the established sanity/evaluation path.

## Route B: retraining

Obtain the 64 train and 16 validation frozen recurrent traces through an approved distribution route, verify them against the committed manifests, then run `train_l6_l7_from_frozen_traces.sh`. Training is fixed to 512-token trajectories, H128 truncated gradients, Adam at 0.003, no weight decay, global gradient clipping at 1.0, 100 optimizer steps, seed 0, independent Hadamard initialization, frozen model weights, and real INT8 recurrent writeback.

The trace public-upload license gate is intentionally unresolved; the repository does not claim that the traces are publicly redistributable.
