"""CPU-only prospective input identity audit. Does not select pairs or run science."""
import base64
import concurrent.futures
import hashlib
import json
import os
import struct
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

D = Path(os.environ["LING_CROSS_RESULTS"])
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
MODEL = Path("/data/zypan/models/Ling-3.0-tiny")
TASK = "LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.name == "PROSPECTIVE_INPUT_ORDER_POLICY_FREEZE.json" and p.exists():
        existing = json.loads(p.read_text())
        assert {k:v for k,v in existing.items() if k != "epoch"} == {k:v for k,v in value.items() if k != "epoch"}
        return
    with p.open("x", encoding="utf8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def token_sha(values):
    h = hashlib.sha256()
    for token in values:
        h.update(struct.pack("<I", token))
    return h.hexdigest()


def fetch(item):
    name = item["name"]
    doc = "pg19-test-" + Path(name).stem
    url = ("https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/"
           + urllib.parse.quote(name, safe="") + "?alt=media&generation=" + item["generation"])
    payload = urllib.request.urlopen(url, timeout=60).read()
    assert len(payload) == int(item["size"])
    assert base64.b64encode(hashlib.md5(payload).digest()).decode() == item["md5Hash"]
    path = D / "prospective_inputs" / "raw_text" / (doc + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    text = payload.decode("utf8")
    return dict(document=doc, object_name=name, generation=item["generation"],
                size_bytes=len(payload), object_md5_base64=item["md5Hash"],
                source_url=url, raw_text_path=str(path), raw_sha256=digest(path),
                normalized_text_sha256=hashlib.sha256(
                    " ".join(unicodedata.normalize("NFC", text).split()).encode()).hexdigest())


def main():
    started = time.time()
    listing = json.loads((D / "PG19_TEST_GCS_LISTING.json").read_text())
    assert len(listing["items"]) == 100 and "nextPageToken" not in listing
    history = json.loads((D / "historical_path_identity_inventory.json").read_text())
    excluded = sorted(history["path_identity_evidence"])
    assert len(excluded) == 20 and "pg19-test-860" in excluded and "pg19-test-5956" in excluded
    publish(D / "PROSPECTIVE_INPUT_ORDER_POLICY_FREEZE.json", dict(
        task=TASK, seed=0, maximum_calibration_screen_documents=40, target_documents=4,
        order='ascending SHA256(UTF8(task + "|seed=0|" + document_id + "|" + raw_sha256)); document_id lexicographic tie-break',
        pool_listing_sha256=digest(D / "PG19_TEST_GCS_LISTING.json"),
        historical_identity_inventory_sha256=digest(D / "historical_path_identity_inventory.json"),
        new_document_calibration_or_long_science_executed=False,
        raw_text_normalization_for_dedup_only="NFC then split/join whitespace; original bytes supplied to tokenizer",
        input_tokenization="frozen local tokenizer.encode(original UTF8 text, add_special_tokens=False)",
        scientific_input_tokens="first 32769 tokens; execute first 32768; no padding/repetition/concatenation",
        gpu_hours=0, epoch=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        inputs = list(executor.map(fetch, listing["items"]))
    print(json.dumps(dict(event="full_test_raw_identity_complete", books=len(inputs))), flush=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    for item in inputs:
        text = Path(item["raw_text_path"]).read_text(encoding="utf8")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert all(type(x) is int and 0 <= x < 157184 for x in tokens)
        prefix = tokens[:32769]
        path = D / "prospective_inputs" / "tokens" / (item["document"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as f:
            json.dump(prefix, f, separators=(",", ":")); f.write("\n")
        item.update(full_token_count=len(tokens), input_token_count=len(prefix),
                    token_file_path=str(path), token_file_sha256=digest(path),
                    canonical_prefix_token_sha256=token_sha(prefix),
                    canonical_full_token_sha256=token_sha(tokens))
        oldraw = Path("/data/zypan/diagnostic_inputs/phase3_mixed_hardware_v1_20261003/raw_text") / (item["document"] + ".txt")
        oldtok = oldraw.parent.parent / "tokens" / (item["document"] + ".json")
        if oldraw.exists() and oldtok.exists():
            oldtokens = json.loads(oldtok.read_text())
            assert digest(oldraw) == item["raw_sha256"]
            assert tokens[:len(oldtokens)] == oldtokens
            item["cached_raw_and_token_prefix_exact"] = True
        item["order_sha256"] = hashlib.sha256(f'{TASK}|seed=0|{item["document"]}|{item["raw_sha256"]}'.encode()).hexdigest()
        print(json.dumps(dict(event="input_tokenization", document=item["document"], full_tokens=len(tokens))), flush=True)
    ordered = sorted(inputs, key=lambda x: (x["order_sha256"], x["document"]))
    historical_inputs = [x for x in inputs if x["document"] in excluded]
    seen_raw = {x["raw_sha256"] for x in historical_inputs}
    seen_norm = {x["normalized_text_sha256"] for x in historical_inputs}
    seen_prefix = {x["canonical_prefix_token_sha256"] for x in historical_inputs}
    seen_full_tokens = {x["canonical_full_token_sha256"] for x in historical_inputs}
    for item in ordered:
        reason = ("HISTORICAL_MEASUREMENT_ID" if item["document"] in excluded else
                  "RAW_CONTENT_DUPLICATE" if item["raw_sha256"] in seen_raw else
                  "NORMALIZED_CONTENT_DUPLICATE" if item["normalized_text_sha256"] in seen_norm else
                  "PREFIX_TOKEN_DUPLICATE" if item["canonical_prefix_token_sha256"] in seen_prefix else
                  "FULL_TOKEN_DUPLICATE" if item["canonical_full_token_sha256"] in seen_full_tokens else
                  "INSUFFICIENT_TOKENS" if item["input_token_count"] != 32769 else None)
        item.update(input_identity_eligible=reason is None, exclusion_reason=reason)
        if reason is None:
            seen_raw.add(item["raw_sha256"]); seen_norm.add(item["normalized_text_sha256"])
            seen_prefix.add(item["canonical_prefix_token_sha256"]); seen_full_tokens.add(item["canonical_full_token_sha256"])
    publish(D / "PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json", dict(
        status="COMPLETE_CPU_INPUT_IDENTITIES_ONLY_NO_CALIBRATION_OR_LONG_OUTCOMES",
        pool="complete official PG19 test split", pool_books=100,
        official_source="https://github.com/google-deepmind/pg19",
        listing_sha256=digest(D / "PG19_TEST_GCS_LISTING.json"),
        order_policy_sha256=digest(D / "PROSPECTIVE_INPUT_ORDER_POLICY_FREEZE.json"),
        historical_excluded_ids=excluded, historical_ids_missing_from_pool=sorted(set(excluded)-{x["document"] for x in inputs}),
        input_identity_eligible_count=sum(x["input_identity_eligible"] for x in ordered),
        model_tokenizer_identity={n:digest(MODEL/n) for n in ["config.json","tokenizer.json","tokenizer_config.json"]},
        inputs=ordered, selected_documents=[], rotation_pairs_selected=False,
        GPU_scientific_units=0, gpu_hours=0, elapsed_seconds=time.time()-started))


if __name__ == "__main__":
    main()
