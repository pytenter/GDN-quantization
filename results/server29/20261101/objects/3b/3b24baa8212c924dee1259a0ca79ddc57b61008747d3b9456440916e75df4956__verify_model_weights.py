"""CPU file hashing only; compare all weights to the retained frozen identity."""
import concurrent.futures
import time
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json

manifest = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/MODEL_WEIGHT_SHA256.txt")
model = Path("/data/zypan/models/Ling-3.0-tiny")
expected = [line.split("  ", 1) for line in manifest.read_text().splitlines() if line.strip()]
start = time.time()
def check(row):
    digest, filename = row
    path = model / filename
    assert path.parent == model
    actual = sha256(path)
    return {"path": str(path), "expected_sha256": digest, "actual_sha256": actual,
            "bytes": path.stat().st_size, "passed": actual==digest}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(check, expected))
passed = all(x["passed"] for x in rows)
atomic_json(ROOT / "FRESH_MODEL_WEIGHT_AUDIT.json", {"status": "PASS" if passed else "FAIL",
       "files_rehashed": len(rows), "manifest_source": str(manifest), "manifest_sha256": sha256(manifest),
       "rows": rows, "elapsed_seconds_cpu_io": time.time()-start, "gpu_hours": 0})
assert passed
with (ROOT / "INHERITED_MODEL_WEIGHT_SHA256.txt").open("xb") as f:
    f.write(manifest.read_bytes())
print("FRESH_MODEL_WEIGHTS_BYTE_IDENTITY_PASS", len(rows))
