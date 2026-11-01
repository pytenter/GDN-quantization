CPU-only delivery tooling for QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1

Current classification: CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED / HISTORICAL_PORT.
This tool grants no GPU authorization and imports no model or GPU runtime.

The default command is a read-only local preview:

    python3 -B delivery_tools/seal_cpu_delivery.py --root "$PWD"

The parent may prepare the exact small-file copy once after final reviews:

    python3 -B delivery_tools/seal_cpu_delivery.py --root "$PWD" --prepare

The parent may then seal and transfer once:

    python3 -B delivery_tools/seal_cpu_delivery.py --root "$PWD" --run

Preparation never seals or transfers. Run requires the prepared copy to match
all current root files, current CPU-only NO authorization, zero GPU ledger,
report-review SHA bindings, worker/controller reviews, and the frozen source
SHA/bytes. Final ROOT status and transport fields are checked separately from
the earlier report review, which explicitly deferred those finalization fields.

The source roster excludes delivery itself, tar.gz files, caches, .DS_Store,
and delivery operation state/receipts/bundles. Existing CPU operation logs remain
included. Tensor/model/native binary files, symlinks, large files and unsafe
paths are rejected. Counts are computed from the actual roster.

The one-shot operation state is outside delivery at
`delivery_tools/receipts/DELIVERY_TRANSFER_STATE.jsonl`. On any failure inspect
that state and all existing destination paths; do not rerun or overwrite.
The exact payload is published to 55 remote_root/final_delivery and
29 archive_root/CPU_DELIVERY, using the existing SSH masters and atomic
no-clobber publication. Each host independently verifies every file SHA/byte
count and actual roster. No raw scientific tensor is transferred to the Mac.

The manifest indexes payload files. SHA256SUMS also indexes the manifest.
FINAL_SEAL_RECEIPT anchors them, FINAL_REPORT, all audit JSON and SOURCE_FREEZE.
LOCAL_DELIVERY_VERIFICATION is an external receipt excluded from its own hash
closure. A separate external replication receipt verifies its identical bytes
on all three hosts, avoiding a circular hash claim.

`transfer_primitives.py` is the byte-exact historical utility with SHA
0214931830433d9e9d7fab27f418d21b838e3464d1a4307b7375c5a535d67948.
Only its stable/roster/check_payload, transport and remote verifier primitives
are reused. Its historical task-specific STOP main is never called.

`test_seal_cpu_delivery.py` tests temporary synthetic roots and a fake transport;
it never uses SSH, initializes a GPU, seals this actual root or creates real
authorization. Its receipt explicitly describes that limitation.
