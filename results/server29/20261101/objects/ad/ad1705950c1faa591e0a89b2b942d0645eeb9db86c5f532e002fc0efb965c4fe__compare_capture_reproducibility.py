"""G4 exact comparison; never selects a better capture or retries failures."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,ROOT,POLICY,read,save,sha,tensor_sha

def main():
    import torch
    p=read(POLICY);a=AMEND/'engineering/capture_01';b=AMEND/'engineering/capture_02'
    rows=[];first_mismatch=None
    for path in [a,b]:
        assert not (path/'capture.error.json').exists(),'failed capture; do not compare as valid'
        assert read(path/'capture_seal.json')['DOCUMENT_DRIVE_SEAL_GATE']=='PASS'
        assert sha(path/'drive_manifest.json')==read(path/'capture_seal.json')['drive_manifest_sha256']
    ma=read(a/'drive_manifest.json');mb=read(b/'drive_manifest.json')
    for key in ['token_count','layer_ids','fixture_token_ids_file_sha256','capture_policy_sha256','capture_script_sha256','source_sha256','dtypes','initial_state','original_production_call_counts']:
        assert ma[key]==mb[key],'capture identity differs: '+key
    assert ma['token_count']==256 and len(ma['shards'])==len(mb['shards'])==4
    for sa,sb in zip(ma['shards'],mb['shards']):
        assert sa['first_token']==sb['first_token'] and sa['last_token']==sb['last_token']
        assert sha(a/sa['relative_path'])==sa['file_sha256'] and sha(b/sb['relative_path'])==sb['file_sha256']
        ta=torch.load(a/sa['relative_path'],map_location='cpu',weights_only=True)
        tb=torch.load(b/sb['relative_path'],map_location='cpu',weights_only=True)
        for layer in ma['layer_ids']:
            for name in p['tensors']:
                x=ta['layers'][str(layer)][name];y=tb['layers'][str(layer)][name]
                exact=tuple(x.shape)==tuple(y.shape) and x.dtype==y.dtype and tensor_sha(x)==tensor_sha(y)
                row=dict(first_token=sa['first_token'],last_token=sa['last_token'],layer=layer,name=name,
                    shape=list(x.shape),dtype=str(x.dtype),tensor_a_sha256=tensor_sha(x),tensor_b_sha256=tensor_sha(y),byte_exact=exact)
                if not exact:
                    row['max_abs_difference']=float((x.float()-y.float()).abs().max())
                    if first_mismatch is None:first_mismatch=row
                rows.append(row)
    ha=read(a/'fp_state_hashes.json');hb=read(b/'fp_state_hashes.json')
    state_mismatches=[]
    assert len(ha)==len(hb)==256
    for ra,rb in zip(ha,hb):
        assert ra['token']==rb['token']
        for layer in ma['layer_ids']:
            key=str(layer)
            if ra['per_layer_sha256'][key]!=rb['per_layer_sha256'][key]:
                state_mismatches.append(dict(token=ra['token'],layer=layer,a_sha256=ra['per_layer_sha256'][key],b_sha256=rb['per_layer_sha256'][key]))
    checkpoint_rows=[]
    for ca,cb in zip(ma['fp_reference_checkpoints'],mb['fp_reference_checkpoints']):
        assert ca['token']==cb['token']
        xa=torch.load(a/ca['relative_path'],map_location='cpu',weights_only=True)
        xb=torch.load(b/cb['relative_path'],map_location='cpu',weights_only=True)
        for layer in ma['layer_ids']:
            x=xa[str(layer)];y=xb[str(layer)]
            exact=tensor_sha(x)==tensor_sha(y)
            checkpoint_rows.append(dict(token=ca['token'],layer=layer,byte_exact=exact,
                max_abs_difference=float((x-y).abs().max()),a_sha256=tensor_sha(x),b_sha256=tensor_sha(y)))
    exact=all(r['byte_exact'] for r in rows) and not state_mismatches and all(r['byte_exact'] for r in checkpoint_rows)
    retained_all_docs=4*19478347776+read(AMEND/'storage_lifecycle_report.json')['SAFETY_MARGIN_BYTES']
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    result=dict(CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE='PASS' if exact else 'FAIL',BYTE_EXACT='YES' if exact else 'NO',
        DRIVE_RELEASE_AFTER_SEAL_ALLOWED='YES' if exact else 'NO',fresh_processes=2,
        canonical_fixture_capture='capture_01; first successful valid capture, never selected from downstream quality',
        drive_tensor_comparisons=len(rows),state_hash_comparisons=256*24,checkpoint_comparisons=len(checkpoint_rows),
        drive_tensor_rows=rows,state_hash_mismatches=state_mismatches,checkpoint_rows=checkpoint_rows,
        first_mismatch=first_mismatch,capture_a_manifest_sha256=sha(a/'drive_manifest.json'),capture_b_manifest_sha256=sha(b/'drive_manifest.json'),
        container_sha_exact_required=False,all_original_dtypes=True,no_threshold_change=True,
        four_full_raw_drive_retention_required_bytes=retained_all_docs,available_bytes_now=available,
        FINAL_ACTION='CONTINUE_ENGINEERING' if exact else ('FAIL_AND_STOP_STORAGE_RETENTION_REQUIRED' if retained_all_docs>available else 'FAIL_AND_STOP'),
        scientific_discovery_units=0,scientific_metrics_computed=False)
    save(AMEND/'engineering/capture_runtime_variability.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)
    return 0 if exact else 2

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        path=AMEND/'engineering/capture_comparison.error.json'
        if not path.exists():save(path,dict(status='FAIL_AND_STOP',exception=type(e).__name__,message=str(e),no_retry=True))
        raise
