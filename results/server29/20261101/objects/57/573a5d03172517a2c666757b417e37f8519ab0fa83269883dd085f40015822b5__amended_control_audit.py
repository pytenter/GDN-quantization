#!/usr/bin/env python3
"""Prospective operational source freeze; CPU-only, no scientific imports."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--host',choices=('A800','3090'),required=True);args=p.parse_args()
    a=json.loads((ROOT/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json').read_text())
    if args.host=='A800':
        for rel in ('protocol/preregistration.json','provenance/trusted_environment_freeze.json','scripts/a800_matched.py','scripts/a800_posttraining.py','scripts/a800_run_all_training_offline.sh'):
            expected=a['artifact_hashes'].get(rel)
            if rel=='protocol/preregistration.json':expected=a['original_preregistration_sha256']
            if rel=='provenance/trusted_environment_freeze.json':expected=a['trusted_environment_sha256']
            if expected is None or sha(ROOT/rel)!=expected:raise RuntimeError('original scientific/protocol source changed '+rel)
        files=['amended_training_entry.py','amended_a800_controller.py','amended_a800_posttraining.py','amended_c5r_reproduction_gate.py','amended_a800_inventory.py','amended_control_audit.py']
        frozen={f'scripts/{name}':sha(ROOT/'scripts'/name) for name in files}
        for ext in ('json','md'):
            rel=f'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.{ext}';frozen[rel]=sha(ROOT/rel)
        save(ROOT/'protocol/amended_control_freeze.json',{'status':'PASS','sha256':frozen,'original_scientific_sources_unchanged':True,'GPU_computation_changed':False,'semantic_retry':False})
    else:
        subprocess.run(['/data/ydai/miniconda3/envs/bitdecode/bin/python',str(ROOT/'scripts/test_amended_control.py')],cwd=ROOT/'scripts',check=True)
    save(ROOT/f'protocol/amended_{args.host}_control_audit.json',{'status':'PASS','host':args.host,'CPU_only':True,'script_sha256':sha(Path(__file__)),'prospective_only':True})
if __name__=='__main__':main()
