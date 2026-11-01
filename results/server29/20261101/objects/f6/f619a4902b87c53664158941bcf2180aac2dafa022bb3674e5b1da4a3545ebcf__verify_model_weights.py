"""CPU-only full-weight identity check, with stat receipts for launch freshness."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import sys
import time
sys.dont_write_bytecode = True
from run_matched_l7 import ROOT, publish

manifest = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/MODEL_WEIGHT_SHA256.txt")
model = Path("/data/zypan/models/Ling-3.0-tiny")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()
assert sha(manifest) == "d8a7cf059fd4b4f2fd7f7d0b1118417be02f7b39a5c428461c027f5d71faff6d"
rows = [line.split("  ",1) for line in manifest.read_text().splitlines() if line.strip()]
assert len(rows) == 32
start = time.time()
def check(row):
    expected, filename = row
    path = model/filename
    assert path.parent == model
    before = path.stat()
    actual = sha(path)
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
    assert actual == expected
    return {"path":str(path),"sha256":actual,"bytes":after.st_size,
            "mtime_ns":after.st_mtime_ns,"passed":True}
with ThreadPoolExecutor(max_workers=2) as pool:
    checked = list(pool.map(check,rows))
publish(ROOT/"FRESH_MODEL_WEIGHT_AUDIT.json", {"status":"PASS","files_rehashed":32,
    "manifest_sha256":sha(manifest),"files":checked,"gpu_hours":0,
    "elapsed_seconds_cpu_io":time.time()-start,"completed_epoch":time.time()})
print("FRESH_MODEL_WEIGHT_REHASH_PASS",len(checked))
