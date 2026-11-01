"""CPU-only final source freeze, no GPU measurement."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path.cwd();D=R/'experiments/LING_FUSED_RMSNORM_GATED_DETERMINISM_CLOSURE_V1'
Q=R/'experiments/LING_V3_FIXED_CONFIG_REUSE_QUALIFICATION_V1'
V=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    with (D/p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
for name in ['closure_common.py','forward.py','whole.py','isolated_backward.py','run_closure.py','prepare.py']:
    (D/'scripts').mkdir(exist_ok=True);shutil.copyfile(Path('/tmp/rmsnorm_closure_staging')/name,D/'scripts'/name)
shutil.copyfile('/tmp/rmsnorm_closure_staging/bootstrap.py',D/'scripts/bootstrap.py')
# Append-only pre-GPU observer plumbing recovery; preserve the first CPU-generated main.
old=D/'runtime/mechanical_v3_whole_main.py';tree=ast.parse(old.read_text());fn=tree.body[0]
if not any('observe_graph' in ast.unparse(n) for n in fn.body):
    for i,n in enumerate(fn.body):
        if ast.unparse(n)=='values[0].backward()':fn.body.insert(i,ast.parse('observe_graph(values[0])').body[0]);break
    else:raise AssertionError('backward boundary absent')
observed=D/'runtime/mechanical_v3_whole_main_observed.py';observed.write_text(ast.unparse(tree)+'\n')
emit(Path('provenance/premeasurement_observer_plumbing.json'),dict(original=str(old),original_sha256=sha(old),observed=str(observed),observed_sha256=sha(observed),change='add no-return-value autograd hash observer before unchanged backward; no numerical expression modified',GPU_measurements_before=0))
rows={}
for p in (D/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec');rows[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
for p in (D/'runtime/historical').rglob('*'):
    if p.is_file():rows[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
for inv in [V/'runtime/phase_a_source_freeze.json']:
    for r in json.loads(inv.read_text())['files']:
        p=Path(r['path']);assert sha(p)==r['sha256'];rows[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
for r in json.loads((Q/'runtime/fla_source_identity.json').read_text())['files']:
    p=Path(r['V3_path']);assert sha(p)==r['V3_sha256'];rows[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
for p in [old,observed,D/'runtime/historical_fixed_forward_manifest.json',D/'protocol/prospective_call37_input_manifest.json',D/'protocol/measurement_plan.json',V/'scripts/common.py',V/'scripts/contemporary_smoke.py',V/'protocol/contemporary_determinism_plan.json',V/'configs/l7r.json']:
    rows[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
emit(Path('runtime/source_freeze.json'),dict(gate='PASS',files=sorted(rows.values(),key=lambda r:r['path']),math_semantics_unchanged=True,first_GPU_measurement=0))
print('SOURCE_FREEZE_PASS',len(rows),flush=True)
