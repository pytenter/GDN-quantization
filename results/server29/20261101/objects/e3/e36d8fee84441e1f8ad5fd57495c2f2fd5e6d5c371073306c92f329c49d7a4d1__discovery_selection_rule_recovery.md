# Discovery rule recovery: FAIL


============================================================
11. FRESH DOCUMENT POOL
============================================================

Use fresh non-AIME long-form text.

Preferred source:

  PG19 TEST / existing independent long-text pool

Do NOT use AIME or reasoning endpoint data.

Mechanically exclude every document used for any Qwen:

  - training
  - validation
  - C5/C6/C5R/C6R/C7 calibration
  - Shared16
  - teacher-forced mechanism discovery
  - historical M5 analysis
  - Stage-B diagnostics
  - any prior PG19 mechanism experiment
  - any candidate selection
  - any prior propagation experiment if one exists

Also exclude Ling-used documents if a shared PG19 pool/manifests make
this easy to audit.

This cross-model exclusion is conservative, not mandatory for model
independence, but use it if document identity is available.

Before any measurement:

select exactly 8 fresh docs by a deterministic pre-existing ordering.

Freeze:

  DISCOVERY_DOCS = first 4 eligible docs
  CONFIRMATION_DOCS = next 4 eligible docs

For each doc freeze at least:

  32769 Qwen token IDs

or whatever exact count is required for a 32768-step replay plus
next-token bookkeeping.

Emit:

  protocol/frozen_fresh_documents.json

Gates:

  FRESH_DOCUMENT_SELECTION_GATE = PASS
  DOCUMENT_OVERLAP_AUDIT_GATE = PASS

============================================================
12. CONFIRMATION MUST BE PRE-FROZEN
============================================================

All 8 documents must be selected and hashed before discovery begins.

However:

  CONFIRMATION_SCIENTIFIC_RESULTS_READ_BEFORE_SELECTION = NO

It is allowed to prepare/capture immutable FP transition traces for all
8 documents for engineering efficiency before discovery aggregation,
provided:

- no rotation-conditioned confirmation metric is computed/read;
- confirmation docs cannot influence metric selection;
- traces are immutable and hash-frozen.

============================================================
13. ROTATION PARAMETERIZATION AUDIT
============================================================

Use only frozen trusted A800-era rotations / H.


The original rule names a deterministic pre-existing ordering but does not define or pin it. Parent protocol/source history does not supply a candidate-pool manifest, dataset revision, comparator or applicable tie policy. The historical calibration corpus order is not an authorized fresh TEST ordering and used calibration documents are explicitly excluded. No Discovery document selected, normalized or tokenized. No Confirmation content or token statistics read. This failure is rule ambiguity, NOT merely absence of already-materialized inputs.

Section8 requires FAIL_AND_STOP; document-ID/hash/dataset order must not be substituted without a separately prospective authorization.
