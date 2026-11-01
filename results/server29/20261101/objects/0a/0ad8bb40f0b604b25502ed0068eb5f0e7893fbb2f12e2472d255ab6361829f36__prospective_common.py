"""Frozen input routing and identity gates; no new scientific formula."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(os.environ['LING_PROSPECTIVE_RESULTS']).resolve()
WT=Path(__file__).resolve().parent
OLD=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def publish(path,value):
    from recovery_state import atomic_json
    atomic_json(path,value)

def verify():
    assert not (OUT/'STOP.json').exists(),'FIRST_FAILURE_STOP'
    audit=read(OUT/'PREREGISTRATION_RECONCILIATION.json');assert audit['status']=='PASS'
    freeze=read(OUT/'SOURCE_CODE_FREEZE.json')
    assert sha(OUT/'CALIBRATION_CONFIG.json')==freeze['config_sha256'],'CONFIG_CHANGED'
    assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
    assert subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()==freeze['source_commit']
    for name,h in freeze['files'].items():assert sha(WT/name)==h,('SOURCE_CHANGED',name)
    for name,h in audit['frozen_protocol_files'].items():assert sha(OLD/name)==h,('ARCHIVED_PROTOCOL_CHANGED',name)
    for name,h in audit['scorer_source_SHA'].items():assert sha(name)==h,('SCORER_SOURCE_CHANGED',name)
    for row in audit['source_assets']:
        s=Path(row['path']).stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),row['path']
    for name,h in audit['historical_STOP_preserved'].items():assert sha(name)==h,('HISTORICAL_STOP_CHANGED',name)
    assert shutil.disk_usage(OUT).free>=64*1024**3,'STORAGE_ADMISSION_FAILED'
    cfg=read(OUT/'CALIBRATION_CONFIG.json')
    routing=read(OUT/'EXECUTION_ROUTING.json')
    assert routing['execution']==str(OUT) and routing['source_worktree']==str(WT)
    uuid=os.environ.get('LING_TARGET_GPU_UUID')
    assert uuid in routing['authorized_devices'],'UNAUTHORIZED_DEVICE'
    cfg['selected_gpu_uuid']=uuid
    return cfg

def entry_for(document):
    audit=read(OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
    entry=next(x for x in audit['inputs'] if x['document']==document)
    assert entry['input_identity_eligible'] and entry['exclusion_reason'] is None
    return entry

def load_inputs(torch,cfg,document,conditions):
    entry=entry_for(document);path=Path(entry['token_file_path'])
    assert sha(path)==entry['token_file_sha256'] and sha(entry['raw_text_path'])==entry['raw_sha256']
    tokens=read(path)
    assert isinstance(tokens,list) and len(tokens)==32769 and all(type(x) is int and x>=0 for x in tokens)
    identities=read(OLD/'DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json')['calibration']['rotation_identities']
    rotations,ids={},{}
    for condition in sorted(set(conditions)):
        if condition=='FP':continue
        item=identities[condition];assert sha(item['path'])==item['sha256'],'ROTATION_CHANGED'
        artifact=torch.load(item['path'],map_location='cpu',weights_only=False)
        assert artifact['layer_ids']==cfg['layers']
        mapping=artifact['rotations']
        matrices=[mapping[layer] if layer in mapping else mapping[str(layer)] for layer in cfg['layers']]
        tensor=torch.stack([x.float() for x in matrices],dim=0)
        assert tuple(tensor.shape)==(18,128,128) and torch.isfinite(tensor).all()
        rotations[condition],ids[condition]=tensor,item
    return entry,tokens,path,rotations,ids
