from pathlib import Path
import hashlib,json,tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
P=Q/'phase_b_inputs';P.mkdir()
rows=[]
for name in ['historical_rebase_qualification.py','historical_rebase_supervisor.py','historical_rebase_prepare_inputs.py']:
    a=Path('/data/zypan')/name;b=P/name
    with b.open('xb') as f:f.write(a.read_bytes())
    h=hashlib.sha256(b.read_bytes()).hexdigest();assert h==hashlib.sha256(a.read_bytes()).hexdigest()
    rows.append(dict(path=str(b.relative_to(R)),sha256=h))
with (P/'INPUT_IDENTITY.json').open('x') as f:json.dump(dict(files=rows,scientific_source_edits=False,runtime_edits=False,formal_measurements=0),f,indent=2)
with tarfile.open(Path(str(R)+'-authority')/'scientific-repro-historical-phase-b-inputs.tar.gz','w:gz') as t:
    for p in sorted(Q.rglob('*')):
        if p.is_file():t.add(p,arcname=str(p.relative_to(R)))
print('PHASE_B_INPUTS_PREPARED',flush=True)
