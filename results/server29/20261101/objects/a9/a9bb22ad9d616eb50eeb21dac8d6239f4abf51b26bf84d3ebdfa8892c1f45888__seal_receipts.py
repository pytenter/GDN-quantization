"""Forensic receipt inventory only, NOT a qualified execution-capsule freeze."""
import json
from pathlib import Path
from capsule_v2 import sha, write

C=Path(__file__).resolve().parent


def main():
    protected=json.loads((C/'post_correction_preservation.json').read_text())
    assert protected['gate']=='PASS'
    raw=json.loads((C/'V2_A/verify/runtime_escape_report.json').read_text())
    elf=[]; nonnative_flagged=[]
    for row in raw['artifacts']:
        p=Path(row['path'])
        if not p.is_file():continue
        with p.open('rb') as f:header=f.read(4)
        if header==b'\x7fELF':elf.append(row['path'])
        elif row['phase'] in ('observer_implementation','python_compile','python_exec','python_import'):
            nonnative_flagged.append(row['path'])
    write(C/'COUNTER_AND_PRESERVATION_CLARIFICATION.json',dict(
        protected_identity_gate='PASS',original_two_A_attempts_byte_identity='PASS',
        original_v2_cleanup_ledger_remaining=1,
        raw_supervisor_native_counter=raw['NATIVE_EXTERNAL_ARTIFACTS'],
        raw_native_counter_definition='Inherited counter treats executable=True (including Python-origin/observer source) as native; not an ELF-only count',
        observed_ELF_regular_file_count=len(elf),observed_ELF_paths=elf,
        nonnative_code_origin_flagged_paths=nonnative_flagged,
        raw_runtime_escape_counter=raw['RUNTIME_ESCAPE_COUNT'],
        raw_runtime_escape_definition='Inherited counter increments on unknown ingress classification; no confirmed executable escape is established',
        scientific_failure_established=False,qualification_gate='FAIL_AND_STOP',
        execution_capsule_frozen=False,external_dependency_manifest_frozen=False,
        source_preservation_evidence_sha256=sha(C/'post_correction_preservation.json')))
    files=[dict(path=str(p.relative_to(C)),size=p.stat().st_size,sha256=sha(p))
        for p in sorted(C.rglob('*')) if p.is_file()]
    write(C/'receipt_sha256_inventory.json',dict(files=files,
        purpose='FORENSIC_RECEIPT_TRANSPORT_INVENTORY_NOT_EXECUTION_CAPSULE',
        capsule_frozen=False,manifest_self_excluded=True,FINAL_ACTION='FAIL_AND_STOP'))
    print('FORENSIC_RECEIPT_INVENTORY_CREATED',len(files),flush=True)


if __name__=='__main__':main()
