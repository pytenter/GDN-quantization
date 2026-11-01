"""CPU-only prospective code/policy extension; no input reselection."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,ROOT,read,save,sha
from isolated_replay_engineering import GATES,require

for phase in ['factorization','schema','fp','invariance','decomposition','isolation']:require(phase)
original=read(AMEND/'storage_lifecycle_report.json')
# Preserve original one-margin accounting report. Additional required2048
# engineering input is explicitly budgeted, not hidden or discarded.
additional=2048*24*24768+2*24*32*128*128*4+64*1024**2
base=original['BASE_PEAK_BYTES']+additional
margin=max(5*1024**3,(base+4)//5);required=base+margin
st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
save(AMEND/'engineering2048_storage_supplement.json',dict(
    scope='Previously required2048-token deterministic engineering fixture, not a new scientific document',
    previous_report_sha256=sha(AMEND/'storage_lifecycle_report.json'),
    original_BASE_PEAK_BYTES=original['BASE_PEAK_BYTES'],additional_engineering_bytes=additional,
    components=dict(drive=2048*24*24768,reference_states=2*24*32*128*128*4,metadata_and_serialization_allowance=64*1024**2),
    BASE_PEAK_BYTES=base,SAFETY_MARGIN_BYTES=margin,PEAK_REQUIRED_STORAGE_BYTES=required,
    AVAILABLE_DATA_BYTES=available,HEADROOM_BYTES=available-required,
    STORAGE_CAPACITY_GATE='PASS' if required<=available else 'FAIL',margin_counted_once=True,
    scientific_protocol_changed=False,documents_reselected=False))
assert required<=available,'Supplemental real engineering storage capacity failed; STOP'
documents=read(ROOT/'protocol/frozen_discovery_documents.json')['documents']
items=[]
for index,row in enumerate(documents,1):
    items.append(dict(doc_id=row['canonical_doc_id'],text_sha256=row['text_sha256'],
        token_file=str(ROOT/row['token_path']),token_file_sha256=row['token_sha256'],token_count=32768,
        output_directory=str(AMEND/'scientific'/('doc_%02d'%index)),required_free_bytes=required))
fixture=dict(items[0]);fixture.update(token_count=2048,output_directory=str(AMEND/'engineering/capture_2048'),
    required_free_bytes=additional+margin)
save(AMEND/'protocol/extended_capture_policy.json',dict(
    production_policy=read(AMEND/'protocol/capture_engineering_preregistration.json'),
    engineering2048=fixture,documents=items,physical_gpu=0,minimum_runtime_free_bytes=margin,
    identity_rule='Existing frozen Doc1 prefix for2048 qualification; same existing four32768 prefixes for science',
    capture_computation_changed=False,compression=False,dtype_conversion=False,drop_tensors=False,
    first_successful_valid_capture=True,engineering2048_retries=0,
    scientific_FP_reference_checkpoints=[2048,8192,32768],
    scientific_per_token_FP_hash_checked_not_persisted=True,
    scientific_sparse_FP_hash_persisted=True,scientific_source_code_freeze_before_first_capture=True))
print('LONG_QUALIFICATION_POLICY_PREPARED',flush=True)
