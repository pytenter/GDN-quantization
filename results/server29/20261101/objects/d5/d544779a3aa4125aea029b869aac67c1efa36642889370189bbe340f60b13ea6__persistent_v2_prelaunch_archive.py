"""Preserve pre-Commit-A operational launch failure; zero worker/model attempt."""
import time
from pathlib import Path
import persistent_v2_common as A
A.require(A.cmd(['git','rev-parse','HEAD'])==A.PARENT,'pre-Commit-A scope only')
A.require(not (A.ROOT/'provenance/v2_premeasurement_commit.json').exists(),'experiment already activated')
A.require(not list((A.ROOT/'analysis').glob('*_worker_result.json')),'scientific evidence exists; forbidden to move')
A.require(not list((A.ROOT/'analysis').glob('*_worker.error.json')),'worker started; cannot recover as prelaunch')
dst=A.ROOT/'logs/precommit_launch_incident_001';dst.mkdir(exist_ok=False)
files=[p for p in (A.ROOT/'analysis').iterdir() if p.is_file() and p.name!='production_runner_scientific_path_identity.json']
files += [A.ROOT/'provenance/protected_artifact_final_revalidation.json',A.ROOT/'logs/offline.log',A.ROOT/'logs/offline.exitcode']
records=[]
for p in files:
 A.require(p.is_file() and A.ROOT in p.resolve().parents,'invalid incident target')
 rel=p.relative_to(A.ROOT);new=dst/str(rel).replace('/','__');A.require(not new.exists(),'incident target exists')
 records.append(dict(original=str(rel),archived=str(new.relative_to(A.ROOT)),sha256=A.sha(p)))
 p.rename(new)
A.save('provenance/precommit_launch_incident_001.json',dict(status='OPERATIONAL_PRELAUNCH_NOT_SCIENTIFIC_MEASUREMENT',time=time.time(),cause='git commit failed: missing repository author identity; shell dispatched controller without successful Commit A; controller clean/precommit gate blocked all spawning; premature CPU finalize created NOT_RUN placeholders and failed missing commit record',model_loads=0,workers_spawned=0,GPU_measurements=0,scientific_retries=0,original_configs_and_sources_unchanged=True,remediation='reuse existing trusted parent commit author via per-command git identity; first valid matrix only after Commit A success/clean status',files=records))
with (A.ROOT/'manifests/v2_precommit_incident_sha256.txt').open('x') as f:
 for r in records:f.write(r['sha256']+'  '+str(A.ROOT/r['archived'])+'\n')
A.verify_manifest()
print('PRELAUNCH_EVIDENCE_PRESERVED_ZERO_MODEL_ATTEMPTS')
