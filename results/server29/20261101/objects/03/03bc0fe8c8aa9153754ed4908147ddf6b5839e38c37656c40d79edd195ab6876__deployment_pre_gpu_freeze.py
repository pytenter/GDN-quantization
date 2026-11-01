#!/usr/bin/env python3
"""Freeze an observational cleanup adjustment prospectively, before first GPU work."""
import ast,difflib,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads((ROOT/'configs/frozen_smoke_protocol.json').read_text())
 old=ROOT/'scripts/deployment_closure.py';new=ROOT/'scripts/deployment_closure_gpu.py'
 assert sha(old)==cfg['source_sha256']
 assert json.loads((ROOT/'analysis/prepare.completion.json').read_text())['status']=='PASS'
 assert not list((ROOT/'analysis').glob('gpu*'))
 a={x.name:ast.dump(x) for x in ast.parse(old.read_text()).body if isinstance(x,ast.FunctionDef)}
 b={x.name:ast.dump(x) for x in ast.parse(new.read_text()).body if isinstance(x,ast.FunctionDef)}
 assert set(a)==set(b) and all(a[k]==b[k] for k in a if k!='smoke')
 # Only observed GPU lifetime cleanup, no CPUnumeric override; imported formal source unchanged.
 diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new)))
 assert 'del p' in diff and "torch.set_num_threads(4)" in diff
 out=dict(status='PASS',old_sha256=sha(old),new_sha256=sha(new),GPU_work_started=False,CPU_prepare_AST_exact=True,canonical_numerical_code_changed=False,reason='Prospective code review found the loop-local p retaining the final model Parameter after del model; delete the reference. Remove unused observer CPU-thread override to preserve historical settings. Append source freeze check. No GPU smoke had started; no semantic retry.',diff=diff)
 with (ROOT/'analysis/pre_gpu_runtime_and_cleanup_source_amendment.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
 print('PRE_GPU_SOURCE_FREEZE=PASS')
if __name__=='__main__':main()
