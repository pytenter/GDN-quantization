"""Outcome-independent documents/bank and historical-trace feasibility only."""
from common import *
import argparse, copy, math, re

def documents():
    # Reuse the exact independent hash order and overlap audit, not a new filter.
    parent=PARENTS[2]
    old=load(parent/'configs/phase3_heldout_manifest.json')
    audit=load(parent/'provenance/phase3_input_overlap_audit.json')
    assert old['gate']==audit['gate']=='PASS'
    duplicates={x['document_id'] for x in audit['duplicate_exclusions']}
    independent=sorted([x for x in audit['candidates'] if x['token_count']>=32769 and not any(x['overlap_32word_counts'].values()) and x['document_id'] not in duplicates],key=lambda x:x['stable_identity_sha256'])
    assert len(independent)==77
    # Only selected-input metadata is scanned. Candidate listings are explicitly
    # NOT treated as used documents; no metrics or outcomes participate.
    metadata=list((parent/'configs').glob('*.json'))
    metadata += [P/'experiments/LING_LAYERWISE_PROPAGATION_FRESH_CONFIRM_V1/configs/fresh16_document_selection.json']
    for name in ['LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V1','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY','LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC']:
        root=P/'experiments'/name
        metadata+=list((root/'protocol').glob('*manifest*.json'))
        metadata+=list((root/'configs').glob('*selection*.json'))
    metadata+=list((P/'experiments/LING_L6_L7_LAYERWISE_PROPAGATION_DIAGNOSIS_V1/configs').glob('*.json'))
    for root in PARENTS[:2]:
        metadata+=list((root/'inputs').glob('*.json'))
        metadata+=list(root.glob('*registration*.json'))
    reasons={}
    for q in sorted(set(metadata)):
        for doc in set(re.findall(r'pg19-test-\d+',q.read_text())):reasons.setdefault(doc,[]).append(str(q))
    for category,dd in audit['excluded_documents'].items():
        for doc in dd:reasons.setdefault(doc,[]).append('historical overlap '+category)
    for doc in duplicates:reasons.setdefault(doc,[]).append('frozen duplicate/shared-passage exclusion')
    chosen=[copy.deepcopy(d) for d in independent if d['document_id'] not in reasons][:8]
    assert len(chosen)==8 and len({x['document_id'] for x in chosen})==8
    cache=Path(old['streams'][0]['external_token_path']).parent
    assert sha(cache/'official_test_listing.json')==old['source_listing_sha256']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for rank,d in enumerate(chosen):
        src=cache/(d['document_id'].removeprefix('pg19-test-')+'.txt')
        assert sha(src)==d['source_file_sha256']
        assert hashlib.sha256(('deepmind-gutenberg/test/'+src.name+'\0'+d['source_file_sha256']).encode()).hexdigest()==d['stable_identity_sha256']
        ids=[int(x) for x in tok(src.read_text(),add_special_tokens=False)['input_ids']]
        assert len(ids)==d['token_count'] and len(ids)>=32769
        ids=ids[:32769];f=D/'inputs/tokens'/(d['document_id']+'.json')
        emit(f,ids);f.chmod(0o444)
        d.update(selection_rank=rank,physical_gpu=rank%2,gpu_uuid=UUIDS[rank%2],phase='DISCOVERY' if rank<4 else 'CONFIRMATION',token_file=row(f),token_ids_sha256=hashlib.sha256(b''.join(x.to_bytes(4,'little') for x in ids)).hexdigest(),token_count_with_target=32769)
    emit('protocol/frozen_documents.json',dict(gate='PASS',rule='first8 after all prior selected metadata exclusions in pre-existing independent hash order',streams=chosen,source_audit=row(parent/'provenance/phase3_input_overlap_audit.json'),source_manifest=row(parent/'configs/phase3_heldout_manifest.json'),pretraining_overlap='UNKNOWN',discovery_count=4,confirmation_count=4,no_confirmation_measurement_before_selection=True))
    emit('provenance/overlap_audit.json',dict(gate='PASS',protected_ids=reasons,scanned_metadata=[row(q) for q in sorted(set(metadata))],independent_order=[x['document_id'] for x in independent],selected_ids=[x['document_id'] for x in chosen],selection_outcome_independent=True,overlap_scope='inherited exact normalized32-word overlaps + stable document IDs, not semantic/pretraining decontamination'))
    print('FRESH_DOCUMENT_SELECTION_GATE=PASS',[(d['document_id'],d['phase']) for d in chosen],flush=True)

def rotations():
    import torch
    torch.set_num_threads(4)
    src=OLD/'experiments/LING_RECURRENT_DENSE_L6_L7_V1/rotations'
    expected={'H':'6500807d6a9ed45b48d54110917e68c206ee88e7263221e163f67ca5d9be8cbf','L6':'ca6ef0d398d0b030dd5c295f62b8c0f43e11d41b92b319bd750ffa064977d71a','L7':'57eaebb47ff031c2b171cbb94563f81481d4f4632098af8339a27d01347060a2'}
    bank={};provenance={}
    for name in ['H','L6','L7']:
        f=src/(name+'_final_rotation.pt');assert sha(f)==expected[name]
        x=torch.load(f,map_location='cpu',weights_only=False);bank[name]={int(k):v.float().contiguous() for k,v in x['rotations'].items()};provenance[name]=dict(source=row(f),rule='unchanged canonical historical rotation')
    for name in ['L7R','L8']:
        m=load(V4/'artifacts'/(name.lower()+'_rotation_manifest.json'));assert m['gate']=='PASS' and sha(m['path'])==m['sha256']
        x=torch.load(m['path'],map_location='cpu',weights_only=False);assert x['condition']==name
        bank[name]={int(k):v.float().contiguous() for k,v in x['rotations'].items()};provenance[name]=dict(source=row(m['path']),manifest=row(V4/'artifacts'/(name.lower()+'_rotation_manifest.json')),rule='frozen V4 selected rotation; no reconstruction')
    for name,source in [('DA_fromH','H'),('DA_fromL6','L6')]:
        f=PARENTS[1]/'rotations'/('DATAADAPT_init_'+source+'_final_rotation.pt')
        x=torch.load(f,map_location='cpu',weights_only=False);assert x['task']=='LING_L6_L7_DATAADAPTIVE_ROTATION_V1' and list(x['layer_ids'])==LAYERS
        assert x['recovery']=='stored = R^T @ native'
        bank[name]={int(k):v.float().contiguous() for k,v in x['rotations'].items()};provenance[name]=dict(source=row(f),parent_identity_inventory=row(D/'provenance/parent_inventory_before.json'),rule='original retained DA artifact; transposed state layout equivalence, no rebuilding',historical_original_release_hash='NOT_RECORDED; exact current parent bytes inventoried before any new measurement')
    bank['IDENTITY']={l:torch.eye(128) for l in LAYERS};provenance['IDENTITY']=dict(rule='R=I valid in exactly the same Value-axis transform/QDQ/recovery path; no bypass')
    polar_meta={}
    for target in ['L6','L7','L7R','L8']:
        for alpha in [.25,.5]:
            name=f'H_to_{target}_a{int(alpha*100):02d}';bank[name]={};singular=[]
            for l in LAYERS:
                a=(1-alpha)*bank['H'][l].double()+alpha*bank[target][l].double()
                u,s,vh=torch.linalg.svd(a,full_matrices=False)
                assert s.min().item()>1e-6,('POLAR_SINGULAR_AMBIGUITY',name,l,float(s.min()))
                r=u@vh;u2,s2,v2=torch.linalg.svd(a,full_matrices=False)
                assert torch.equal(r,u2@v2),('POLAR_NONDETERMINISTIC',name,l)
                assert torch.sign(torch.linalg.det(r))==torch.sign(torch.linalg.det(a))
                bank[name][l]=r.float().contiguous()
                singular.append(dict(layer_id=l,min_singular=float(s.min()),max_singular=float(s.max()),det_A=float(torch.linalg.det(a)),det_R=float(torch.linalg.det(r))))
            provenance[name]=dict(rule='unique full-rank polar(A)=U@Vh in CPU float64, cast float32; no determinant/sign correction',from_condition='H',target=target,alpha=alpha,min_singular_stop_threshold=1e-6,rows=singular)
    assert len(bank)==16
    rows=[]
    for name,rr in bank.items():
        assert sorted(rr)==LAYERS
        ortho=[]
        for l,r in rr.items():
            assert r.shape==(128,128) and torch.isfinite(r).all()
            err=float((r.double().T@r.double()-torch.eye(128,dtype=torch.float64)).abs().max())
            assert err<=1e-4,(name,l,err)
            ortho.append(dict(layer_id=l,sha256=tensor_sha(r),orthogonality_max_abs=err))
        f=D/'rotations'/(name+'.pt');f.parent.mkdir(parents=True,exist_ok=True)
        assert not f.exists();torch.save(dict(condition=name,layer_ids=LAYERS,rotations=rr),f);f.chmod(0o444)
        rows.append(dict(condition_id=name,artifact=row(f),layers=ortho,provenance=provenance[name]))
    uniqueness=[]
    for i,aa in enumerate(bank):
        for bb in list(bank)[i+1:]:
            dist=max(float((bank[aa][l]-bank[bb][l]).abs().max()) for l in LAYERS)
            assert dist>1e-6,('ROTATION_DUPLICATE_OR_AMBIGUITY',aa,bb,dist)
            uniqueness.append(dict(a=aa,b=bb,max_abs_difference=dist))
    emit('protocol/frozen_rotation_bank.json',dict(gate='PASS',count=16,conditions=rows,execution_order=list(bank),layer_ids=LAYERS,identity_legal=True,rotation_basis='native key-first state @ R; native value @ R',orthogonality_gate='PASS',orthogonality_max_abs_threshold=1e-4,uniqueness_gate='PASS',uniqueness_threshold=1e-6,unique_pairs=uniqueness))
    print('ROTATION_BANK_FREEZE_GATE=PASS; COUNT=16',flush=True)

def main():
    assert not (D/'run/STOP.json').exists()
    assert load(D/'analysis/transition_parity_smoke.json')['gate']=='PASS'
    assert load(D/'provenance/historical_metric_definition_audit.json')['gate']=='PASS'
    documents();rotations()

if __name__=='__main__':
    try:main()
    except BaseException as e:stop('FRESH_DOC_AND_BANK_PREPARATION',e);raise
