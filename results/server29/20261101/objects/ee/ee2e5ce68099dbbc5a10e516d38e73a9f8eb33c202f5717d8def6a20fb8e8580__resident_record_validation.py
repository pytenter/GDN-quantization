"""Strict, gold-blind validation for one resident C1/S3 generation.

Only stdlib imports: safe before Torch and in independent scoring processes.
These checks validate records; they never authorize a GPU or score an answer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CAP = 81920
VOCAB = 248320
LAYERS = tuple(i for i in range(32) if i % 4 != 3)
SPARSE_PREFIX = 32
SPARSE_PERIOD = 256
CORE_SHA = "4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95"
SAMPLER_SHA = "9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd"
SCORER_SHA = "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b"
UUIDS = {1: "GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3",
         2: "GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31",
         3: "GPU-30b07aba-a728-7649-4899-e09d33f8bc75"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def objsha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sparse_position(index):
    return index > 0 and (index <= SPARSE_PREFIX or index % SPARSE_PERIOD == 0)


def sparse_positions(total):
    return [index for index in range(total) if sparse_position(index)]


def verify_consumption_contract(root):
    root = Path(root)
    contract = json.loads((root / "RESIDENT_FORMAL_CONSUMPTION_CONTRACT.json").read_text())
    require(contract["input_index_zero_based"] is True
            and contract["sparse_prefix_inclusive"] == SPARSE_PREFIX
            and contract["sparse_period"] == SPARSE_PERIOD
            and contract["max_new_tokens"] == CAP
            and contract["resident_core_sha256"] == CORE_SHA
            and contract["formal_manifest_sha256"] == sha(root / "FORMAL_JOB_MANIFEST.json")
            and contract["worker_sha256"] == sha(root / "scripts/resident_formal_worker.py")
            and contract["validator_sha256"] == sha(root / "src/resident_record_validation.py"),
            "Pre-frozen formal consumption/source/manifest contract differs")
    return contract


def validate_evidence(entry):
    expected_fields = {
        "codes": ("torch.int8", [1, 32, 4, 32, 128], 524288),
        "scales": ("torch.float32", [1, 32, 4, 128], 65536),
        "negative_zero_flags": ("torch.bool", [1, 32, 4, 32, 128], 524288)}
    fields = entry["encoded_fields"]
    require(len(fields) == 3 and {field["name"] for field in fields} == set(expected_fields),
            "Encoded evidence fields missing/duplicate")
    for field in fields:
        dtype, shape, size = expected_fields[field["name"]]
        require(field["dtype"] == dtype and field["shape"] == shape and field["bytes"] == size
                and isinstance(field["sha256"], str) and len(field["sha256"]) == 64,
                "Encoded actual-field dtype/shape/bytes/SHA evidence differs")
    require(entry["bytes"] == 2097152 and entry["actual_values_and_bytes_equal"] is True
            and entry["actual_writeback_sha256"] == entry["expected_decoded_sha256"],
            "Actual resident State value/byte evidence differs")


def validate(wrapper, job, root, *, bind_files=True):
    root = Path(root)
    require(wrapper.get("status") == "PASS" and wrapper.get("unit") == job["id"],
            "Formal worker record/unit is not PASS")
    copy = dict(wrapper)
    saved = copy.pop("payload_sha256", None)
    require(saved == objsha(copy), "Worker payload SHA differs")
    row = wrapper["result"]
    require(row.get("status") == "COMPLETE" and row.get("nonfinite") is False
            and row.get("runtime_error") is None, "Incomplete or technical-failure generation")
    gpu = job["gpu"]
    require(gpu in UUIDS and job["gpu_uuid"] == UUIDS[gpu], "Frozen physical UUID differs")
    require(row["condition"] == job["condition"] in ("C1", "S3")
            and row["problem_id"] == job["problem_id"] and row["seed"] == job["seed"] == 1
            and row["gpu"] == gpu and row["gpu_uuid"] == UUIDS[gpu], "Sample identity differs")
    require(row["input_ids_sha256"] == job["input_ids_sha256"]
            and row["input_tokens"] == job["input_tokens"] and 0 < row["input_tokens"] <= 373,
            "Frozen prompt token payload differs")
    tokens = row["generated_token_ids"]
    require(isinstance(tokens, list) and 0 < len(tokens) <= CAP
            and all(type(token) is int and 0 <= token < VOCAB for token in tokens),
            "Generated token sequence/range invalid")
    n = len(tokens)
    require(n == row["generated_token_count"] and objsha(tokens) == row["generated_token_ids_sha256"],
            "Generated token payload/hash differs")
    stops = row["stop_token_ids"]
    require(isinstance(stops, list) and bool(stops)
            and len(stops) == len(set(stops))
            and all(type(token) is int and 0 <= token < VOCAB for token in stops), "EOS roster invalid")
    eos = tokens[-1] in stops
    require(not any(token in stops for token in tokens[:-1])
            and row["finish_reason"] == ("eos" if eos else "length")
            and (eos or n == CAP) and row["max_new_tokens"] == CAP,
            "EOS/length termination is not the original protocol")
    require(isinstance(row["response"], str), "Response must be exact tokenizer text")
    total = row["input_tokens"] + n - 1
    audit = row["quantizer_audit"]
    require(row["actual_model_forward_calls"] == total
            and audit["processed_tokens"] == total
            and audit["qdq_calls"] == 24 * total
            and row["prompt_layer_QDQ"] == 24 * row["input_tokens"]
            and row["decode_layer_QDQ"] == 24 * (n - 1), "Prompt/Decode QDQ p+n-1 identity differs")
    require(set(row["writebacks_by_layer"]) == {str(layer) for layer in LAYERS}
            and all(value == total for value in row["writebacks_by_layer"].values()),
            "Actual 24-layer writeback counts differ")
    require(audit["qk_batched_matmul_calls"] == 24 * total
            and audit["unique_qk_vectors_rotated"] == 32 * 24 * total
            and audit["original_gdn_helper_calls"] == 24 * total
            and audit["full_token_finite_checks"] == total
            and audit["chunk_prefill_calls"] == 24
            and audit["recurrent_decode_calls"] == 24 * (total - 1), "Resident operation coverage differs")
    require(audit["state_rotation_matmul_calls"] == 0 and audit["P_inside_QDQ"] is False
            and audit["high_precision_continuation_shadow"] is False
            and audit["state_dtype"] == "FP32" and audit["packed_INT8_persistent_storage"] is False,
            "Resident coordinate/QDQ/State-storage contract differs")
    positions = sparse_positions(total)
    require(row["sparse_next_consumption_positions"] == positions
            and row["sparse_next_consumption_checks"] == 24 * len(positions),
            "Frozen sparse actual next-consumption roster differs")
    require(row["sparse_evidence_sha256"] == objsha(row["sparse_next_consumption_evidence"]),
            "Sparse evidence payload SHA differs")
    coverage = [(entry["input_index"], entry["layer"]) for entry in row["sparse_next_consumption_evidence"]]
    require(len(coverage) == len(set(coverage))
            and set(coverage) == {(index, layer) for index in positions for layer in LAYERS},
            "Sparse evidence lacks exact per-layer coverage")
    for entry in row["sparse_next_consumption_evidence"]:
        validate_evidence(entry)
    final = row["final_writeback_evidence"]
    require(len(final) == 24 and {entry["layer"] for entry in final} == set(LAYERS)
            and all(entry["input_index"] == total - 1 for entry in final),
            "Final cache does not contain the last resident QDQ writeback")
    for entry in final:
        validate_evidence(entry)
    require(row["final_writeback_is_not_an_extra_next_consumption"] is True,
            "Final writeback cannot be called an additional next consumption")
    require(row["last_output_token_fed_back"] is False and row["sampler_calls"] == n,
            "Last output was fed back or sampler call count differs")
    sampling = row["sampling_audit"]
    require(sampling["top_k_parameter"] == 20
            and 1 <= sampling["min_survivors"] <= sampling["max_survivors"] <= VOCAB
            and sampling["original_warper_unchanged"] is True,
            "Original tie-aware HF sampler audit failed")
    expected_sampling = dict(temperature=1., top_p=.95, top_k=20, min_p=0.,
                             presence_penalty=1.5, repetition_penalty=1.,
                             generated_only_presence=True, seed=1, thinking=True)
    require(row["sampling_parameters"] == expected_sampling, "Formal sampler parameters changed")
    require(row["resident_core_sha256"] == CORE_SHA and row["sampler_source_sha256"] == SAMPLER_SHA
            and row["scorer_source_sha256"] == SCORER_SHA, "Original core/protocol SHA differs")
    require(isinstance(row["wall_seconds"], (int, float)) and math.isfinite(row["wall_seconds"])
            and row["wall_seconds"] > 0 and row["TPS"] > 0,
            "Monotonic resource timing invalid")
    if bind_files:
        verify_consumption_contract(root)
        require(wrapper["source_freeze_sha256"] == sha(root / "SOURCE_FREEZE.json")
                and wrapper["permit_sha256"] == sha(root / "GPU_PERMIT.json")
                and wrapper["formal_permit_sha256"] == sha(root / "FORMAL_GPU_PERMIT.json")
                and wrapper["formal_authorization_sha256"] == sha(root / "AUTHORIZATION_FORMAL.json"),
                "Formal authorization/source/permit binding differs")
        require(row["resident_core_sha256"] == sha(root / "src/resident_rotation.py")
                and row["sampler_source_sha256"] == sha(root / "protocol/sampler.py")
                and row["scorer_source_sha256"] == sha(root / "protocol/aime26_scorer_v4.py")
                and row["dataset_freeze_sha256"] == sha(root / "AIME26_DATASET_FREEZE.json"),
                "Actual source/input payload binding differs")
        require(row["consumption_contract_sha256"] == sha(root / "RESIDENT_FORMAL_CONSUMPTION_CONTRACT.json"),
                "Formal actual sparse-consumption contract SHA differs")
    return row
