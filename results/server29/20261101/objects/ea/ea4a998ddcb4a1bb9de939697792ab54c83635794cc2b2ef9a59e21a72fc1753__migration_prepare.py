"""Prospective hardware-only amendment; preserve original protocol objects."""
from inherit import *
import json,time

def load(p):return json.loads(Path(p).read_text())
def emit(p,value):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as s:json.dump(value,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')

assert not REMOTE
assert not (D/'run/STARTED.json').exists() and not list((D/'raw/confirmation').glob('*.json'))
original=load(D/'protocol/inherited_confirmation_docs.json');docs=[]
for i,doc in enumerate(original['streams']):
    g=GPUS[i%2];docs.append(dict(doc,physical_gpu=g,gpu_uuid=UUIDS[g]))
hw=c.hardware();assert hw['host']=='nlpg-SYS-4029GP-TRT'
for g in GPUS:assert hw['gpus'][g]['used_mib']<2000
emit('protocol/hardware_assignment_amendment_v1.json',dict(
    amendment='LING_LONG_HORIZON_PROPAGATION_CONFIRM_HARDWARE_AMENDMENT_V1',scope='HARDWARE_AND_OPERATIONAL_TRANSPORT_ONLY',
    user_authorization='你把我原本要进行的任务移到八卡3090上面来进行吧，使用上面的两张空闲的卡',
    freeze_before_holdout_measurements=True,holdout_measurements=0,original_protocol_objects_preserved=True,
    original_assignment_manifest=c.row(D/'protocol/inherited_confirmation_docs.json'),
    original_hardware='2xRTX4090',effective_host='nlpg-SYS-4029GP-TRT',effective_hardware='2xRTX3090',physical_gpus=GPUS,gpu_uuids=UUIDS,
    effective_streams=docs,selection='GPU3 andGPU5 idle before any holdout computation; GPU4 occupied and untouched; GPU0 excluded',
    assignment='DOC0/DOC2 -> physicalGPU3; DOC1/DOC3 -> physicalGPU5; fixed once prospectively',
    unchanged=['four document identities and token IDs','16 rotation matrices/order','K=8 and exact probe signs',
        'FP model exact revision','all52 package versions and Python','qualified fixed RMSNorm forward','KDA transition','persistent INT8R128 QDQ writeback',
        'all metric definitions','h32768 only primary','all advancement thresholds','no training/AIME/freegen','final STOP'],
    fp_policy='Capture all four new FP traces exclusively on assigned3090; never mix or substitute4090 FP numeric traces. Retain cross-hardware bitwise outcomes separately.',
    development_qualification='128-token historical training source only; exact replay parity and sameGPU repeated bitwise required on both selected3090 GPUs',
    original_parent_verdict='FAIL_PRESERVED',external_processes_terminated=False,time_unix=time.time()))
emit('provenance/3090_runtime_transport_identity.json',load(D/'transport/3090_transport_identity_PASS.json'))
emit('provenance/hardware_identity_amendment_v1.json',dict(gate='PASS',actual_hardware=hw,
    effective_gpu_uuids=UUIDS,holdout_measurements=0,external_processes_terminated=False))
emit('provenance/3090_execution_boundary.json',dict(
    authoritative_git_and_controller=str(R),measurement_git_and_root=str(DEST),
    controller_4090_gpu_usage='NONE',measurements='ONLY3090 GPU3/GPU5',parent_originals='READ_ONLY_AND_RETAINED_ON4090',
    transport_identity='exact SHA256 for every imported input/source/model/library; original JSON bytes retained, paths remapped in-memory only',
    commit_transport='same premeasurement commit/tree transported as shallow engineering checkout; no GitHub push',
    result_sync='byte-identical64 canonical raw JSON plus4 FP trace manifests copied back for original-parent verification and final frozen analysis',
    scientific_functions_edited=False,hardware_amendment_preserves_all_scientific_rules=True))
print('PROSPECTIVE3090_HARDWARE_AMENDMENT_PREPARED; HOLDOUT_MEASUREMENTS=0',flush=True)
