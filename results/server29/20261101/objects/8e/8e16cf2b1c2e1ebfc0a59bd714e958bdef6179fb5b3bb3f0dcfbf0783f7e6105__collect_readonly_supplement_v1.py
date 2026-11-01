"""Static dependency, receipt, and process supplement; stdlib only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cmd(a):
    x = subprocess.run(a, capture_output=True, text=True, timeout=30)
    return dict(command=a, returncode=x.returncode, stdout=x.stdout, stderr=x.stderr, epoch=time.time())

def main():
    assert OUT != ROOT and OUT.name == 'LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.dont_write_bytecode
    base = json.loads((OUT/'READONLY_SERVER_EVIDENCE.json').read_text())
    files = {}; sources = {}; hashes = {}
    freeze = base['data']['PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json']
    for p in freeze['dependencies']:
        if Path(p).suffix == '.py' and p not in base['static_sources']:
            sources[p] = Path(p).read_text(); hashes[p] = sha(p)
            assert hashes[p] == freeze['dependencies'][p]
    for p in sorted((ROOT/'runs').glob('*.json')):
        if p.name.endswith(('.STARTED.json', '.LAUNCH_AUTHORIZATION.json')) or p.name.count('.') == 1:
            files[str(p)] = json.loads(p.read_text()); hashes[str(p)] = sha(p)
    for name in ('PERSISTENT_GUARD_EXCEPTION_1791702534453816856.json',
                 'PERSISTENT_SOURCE_CPU_AUDIT_V3.json', 'PERSISTENT_SOURCE_CPU_ENGINEERING_FAILURE_V1.json',
                 'PERSISTENT_PRE_GPU_BUILD_ENGINEERING_APPENDIX_V1.json'):
        p = ROOT/name; files[str(p)] = json.loads(p.read_text()); hashes[str(p)] = sha(p)
    processes = {str(pid): dict(proc_path_exists=Path('/proc/'+str(pid)).exists()) for pid in (341582,341594)}
    sessions = {s:cmd(['tmux','has-session','-t',s]) for s in
        ('ling_l7_b_persistent_qualification_gpu6','ling_l7_b_persistent_qualification_gpu3')}
    repos = {p:dict(expected_commit=c, head=cmd(['git','-C',p,'rev-parse','HEAD']),
                   status=cmd(['git','-C',p,'status','--porcelain=v1']),
                   expected_exists=cmd(['git','-C',p,'cat-file','-e',c+'^{commit}']))
             for p,c in base['data']['SOURCE_IDENTITIES.json']['source_repositories'].items()}
    current = {p:sha(p) for p in base['original_inventory_before']}
    assert current == base['original_inventory_before']
    result = dict(epoch=time.time(), actual_server_evidence=True, sources=sources, hashes=hashes,
        original_files_still_identical=True, original_file_count=len(current), files=files,
        persistent_processes=processes, persistent_tmux=sessions, source_repositories=repos,
        new_persistent_canonical_paths=[str(p) for p in (ROOT/'runs_b_persistent').glob('*.json')
            if p.name.count('.') == 1],
        complete_cost_gate_exists=(ROOT/'PERSISTENT_COMPLETE_COST_GATE_V1.json').exists(),
        no_gpu_imports=not any(n.split('.')[0] in {'torch','fla','triton','transformers'} for n in sys.modules),
        collector_sha256=sha(__file__), base_evidence_sha256=sha(OUT/'READONLY_SERVER_EVIDENCE.json'))
    p = OUT/'READONLY_SERVER_SUPPLEMENT.json'
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(json.dumps(dict(path=str(p),sha256=sha(p),no_gpu_imports=result['no_gpu_imports'],
                         original_files_still_identical=True)))

if __name__ == '__main__': main()
