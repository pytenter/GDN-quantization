# Data provenance and public-upload decision

The frozen recurrent traces were derived from the Salesforce WikiText-2 raw calibration corpus. The recorded raw-corpus SHA256 is:

`37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0`

Official dataset page: <https://huggingface.co/datasets/Salesforce/wikitext>

The official metadata currently carries `cc-by-sa-3.0` and `gfdl` tags, while dataset-card text states CC BY-SA 4.0. The available evidence therefore does not conclusively settle the applicable version or the redistribution classification of token IDs and model-derived recurrent tensors.

Consequently:

`TRACE_PUBLIC_LICENSE_GATE = BLOCKED_PENDING_LICENSE_REVIEW`

`TRACE_HF_UPLOAD_GATE = BLOCKED_BY_LICENSE_ONLY`

The four canonical checkpoints/rotations, training metadata, and manifests do not contain the corpus text or recurrent trace payloads and may be uploaded in Phase 1. The 80 trace files must not be uploaded to the public repository until an explicit license determination is recorded.
