"""Publish bounded CPU evidence siblings; preserve all claims on failure."""
import base64,hashlib,json,time,traceback
from pathlib import Path
import review_once as R
O=R.O
HERE=R.HERE
ROOT=R.ROOT

def row(path,name):
    data=Path(path).read_bytes()
    return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def unchanged_files(files):
    for path,x in files:
        O.require(Path(path).stat().st_size==x['bytes'] and O.digest(path)==x['sha256'],'Publication input changed: '+str(path))

def publish(t,index,files,core):
    O.require(0<len(files)<=24 and sum(x['bytes'] for p,x in files)<=33554432,'Independent planned part bound exceeded')
    unchanged_files(files)
    inventory=[x for p,x in files]
    O.write_new(HERE/('PART%d_LOCAL_INVENTORY.json'%index),O.canonical(dict(files=inventory))+b'\n')
    digest=hashlib.sha256(O.canonical(inventory)).hexdigest()
    final=R.M.ARCHIVE+'.independent_terminal_supplement_'+digest
    O.write_new(HERE/('PART%d_CLAIM.json'%index),O.canonical(dict(status='INDEPENDENT_BOUNDED_CPU_PART_CLAIMED_ONCE',files=len(files),bytes=sum(x['bytes'] for p,x in files),archive_root=final,core_manifest_sha256=core,unix=time.time()))+b'\n')
    payload=[dict(x,base64=base64.b64encode(Path(p).read_bytes()).decode()) for p,x in files]
    proof=O.call(t,O.REMOTE_SUPPLEMENT,dict(final=final,core_manifest_sha256=core,files=payload),role='29',timeout=300)
    O.require(proof['status']=='PASS_NEW_SIBLING_SUPPLEMENT_ALL_SHA_BYTES_ATOMIC_SEAL' and proof['files']==len(files),'Part publication receipt differs')
    O.write_new(HERE/('PART%d_VERIFICATION.json'%index),O.canonical(proof)+b'\n')
    unchanged_files(files)
    return proof

def second_review(t,proofs,core):
    checks=R.call(t,"print(json.dumps(dict(status='PASS_PUBLISHED_PARTS_SECOND_FRESH_SHA',parts=[regular_manifest(x['archive_root'],x['manifest_sha256']) for x in P['parts']])))",dict(parts=proofs),'29')
    snapshot=O.call(t,'print(json.dumps(snapshot(P)))',dict(root=R.M.REMOTE,archive=R.M.ARCHIVE,source_sha256=O.digest(ROOT/'source_manifest.json'),permit_sha256=O.digest(ROOT/'GPU_PERMIT.json')),timeout=60)
    O.require(snapshot['status']=='READY_FOR_INDEPENDENT_AUDIT' and snapshot['formal_completed']==60 and not snapshot['active'] and not snapshot['NVML']['own_apps'] and not snapshot['scoped_processes'] and all(not x['owned_alive'] for x in snapshot['identities']),'Final exact owning identity/NVML release changed')
    O.verify_rows(ROOT,json.loads((ROOT/'source_manifest.json').read_bytes())['files'])
    mac=R.inventories()[0]
    a55=json.loads((HERE/'AUDIT55.json').read_bytes());a29=json.loads((HERE/'AUDIT29.json').read_bytes());local=json.loads((HERE/'LOCAL_EXTRA_EVIDENCE_VERIFICATION.json').read_bytes())
    O.require(a55['legal_formal_union']==60 and a55['current_completed']==6 and a55['reused']==54 and a55['full_accuracy_reported'] and a55['score_review']['new_score_passes']==0 and len(a55['roots'])==len(a29['roots'])==len(mac)==8,'Extra independent review coverage differs')
    O.require(sum(x['verification']['files'] for x in a55['Triton'])==322 and a55['model17']['files']==17 and a55['actual_mapped_artifacts']>0 and a55['actual_unique_regular_mapped_files']>0,'Extra whole model/Triton/actual mapped coverage differs')
    for x in local['files']:
        O.require(Path(x['path']).stat().st_size==x['bytes'] and O.digest(x['path'])==x['sha256'],'Reviewed local extra evidence changed')
    primary=json.loads((ROOT/'independent_delivery/RESULT.json').read_bytes())
    O.require(primary['formal_completed']==60 and primary['final_status']=='FORMAL_COMPLETE_60_60_AND_STOP' and primary['core_manifest_sha256']==core,'Primary final status/anchor differs')
    return dict(status='PASS_SECOND_INDEPENDENT_COMPLETION_IDENTITIES_MAC_SOURCE_AND_PUBLISHED_PARTS',source_manifest_sha256=O.digest(ROOT/'source_manifest.json'),permit_sha256=O.digest(ROOT/'GPU_PERMIT.json'),core_manifest_sha256=core,formal_completed=60,first_extra_review=dict(review55_sha256=O.digest(HERE/'AUDIT55.json'),review29_sha256=O.digest(HERE/'AUDIT29.json'),local_extra_sha256=O.digest(HERE/'LOCAL_EXTRA_EVIDENCE_VERIFICATION.json'),all_current_and_prior_roots=8,model17=17,Triton322=322,actual_mapped_artifacts=a55['actual_mapped_artifacts'],actual_regular_mapped_files=a55['actual_unique_regular_mapped_files']),current_exact_identity_snapshot=snapshot,all_mac_origins=mac,parts=checks['parts'],GPU_kernel_calls=0,new_scores=0,model_loads=0,primary_seal_modified=False)

def main():
    result=json.loads((HERE/'RESULT.json').read_bytes());O.require(result['status']=='PASS_EXTRA_FULL60_INDEPENDENT_REVIEW' and result['formal_completed']==60,'Extra full review incomplete')
    primary=json.loads((ROOT/'independent_delivery/RESULT.json').read_bytes());core=primary['core_manifest_sha256']
    files=[]
    for name in ['review_once.py','full_score_checks.py','local_extra_evidence.py','publish_review_once.py','REVIEW_PLAN.json','CLAIM.json','MAC_AND_ORIGIN_VERIFICATION.json','RECOVERY46_VERIFICATION.json','LOCAL_EXTRA_EVIDENCE_VERIFICATION.json','AUDIT55.json','AUDIT29.json','RESULT.json']:
        p=HERE/name;files.append((p,row(p,'extra_final_review/'+name)))
    for name in list(R.local_extra_evidence.CURRENT_REVIEW_PINS)+['REBALANCE_MONITOR_EXTRA_BOUNDARIES.json','CURRENT_4GPU_HANDOFF.json']:
        p=ROOT/'operations'/name;files.append((p,row(p,'local_operations/'+name)))
    planning=ROOT.parent/'PACE_FOUR_GPU_REBALANCE_PLANNING_20261027'
    for name in ['HUMAN_ACCEPTED_REBALANCE.json','ORIGINAL_CONTROLLER_STOP_REQUEST_RECEIPT.json','PROPOSED_REBALANCE.json','CPU_PREPARATION_SCOPE_NOTES.json','CPU_PROMPT_TRANSPORT_SCOPE_NOTES.json']:
        p=planning/name;files.append((p,row(p,'rebalance_authorization/'+name)))
    for name in ['RESULT.json','LOCAL_SECOND_VERIFICATION.json','TRANSFER_AND_SCORE_VERIFICATION.json']:
        p=ROOT.parent/'PACE_MANUAL_PARTIAL_SCORE_20261027_LIFECYCLE_V1'/name;files.append((p,row(p,'separate_human_requested_partial_score53/'+name)))
    old_local=json.loads((HERE/'LOCAL_EXTRA_EVIDENCE_VERIFICATION.json').read_bytes())
    for x in old_local['old_start_reviews']:
        p=Path(x['path']);files.append((p,row(p,'old_local_start_reviews/'+p.parent.parent.name+'/'+p.name)))
    chunks=[files[i:i+23] for i in range(0,len(files),23)]
    plan=dict(status='PLANNED_BEFORE_PUBLICATION_CLAIM',parts=[dict(index=i+1,files=[x for p,x in part],count=len(part)+(i==len(chunks)-1),publication_plan_added=(i==len(chunks)-1)) for i,part in enumerate(chunks)],maximum_files_per_part=24,maximum_bytes_per_part=33554432,all_original46_bytes_already_archived_and_reverified=True,core_manifest_sha256=core)
    O.write_new(HERE/'PUBLICATION_PLAN.json',O.canonical(plan)+b'\n')
    chunks[-1].append((HERE/'PUBLICATION_PLAN.json',row(HERE/'PUBLICATION_PLAN.json','extra_final_review/PUBLICATION_PLAN.json')))
    for part in chunks:O.require(len(part)<=24 and sum(x['bytes'] for p,x in part)<=33554432,'Planned part exceeds original bound')
    O.write_new(HERE/'PUBLICATION_CLAIM.json',O.canonical(dict(status='PLANNED_BOUNDED_PARTS_CLAIMED_ONCE',plan_sha256=O.digest(HERE/'PUBLICATION_PLAN.json'),parts=len(chunks),unix=time.time()))+b'\n')
    try:
        t=O.load_transport(ROOT/'operations/remote.py','db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40')
        proofs=[publish(t,i+1,part,core) for i,part in enumerate(chunks)]
        second=second_review(t,proofs,core);O.write_new(HERE/'SECOND_INDEPENDENT_REVIEW.json',O.canonical(second)+b'\n')
        closing_names=['SECOND_INDEPENDENT_REVIEW.json','PUBLICATION_CLAIM.json']
        for index in range(1,len(proofs)+1):closing_names+=['PART%d_VERIFICATION.json'%index,'PART%d_LOCAL_INVENTORY.json'%index,'PART%d_CLAIM.json'%index]
        closing=[(HERE/name,row(HERE/name,'extra_final_review/'+name)) for name in closing_names]
        O.write_new(HERE/'CLOSURE_PUBLICATION_PLAN.json',O.canonical(dict(status='CLOSURE_PART_PLANNED_BEFORE_CLAIM',files=[x for p,x in closing],maximum_files_per_part=24,maximum_bytes_per_part=33554432))+b'\n')
        closing.append((HERE/'CLOSURE_PUBLICATION_PLAN.json',row(HERE/'CLOSURE_PUBLICATION_PLAN.json','extra_final_review/CLOSURE_PUBLICATION_PLAN.json')))
        proofs.append(publish(t,len(proofs)+1,closing,core))
        final_check=R.call(t,"print(json.dumps(dict(status='PASS_ALL_NEW_EXTRA_SIBLINGS_SECOND_FRESH_SHA',parts=[regular_manifest(x['archive_root'],x['manifest_sha256']) for x in P['parts']])))",dict(parts=proofs),'29')
        for part in chunks+[closing]:unchanged_files(part)
        value=dict(status='PASS_FULL60_PRIMARY_EXTRA_ALL_ORIGINS_AND_BOUNDED_SUPPLEMENTS_COMPLETELY_DELIVERED',source_manifest_sha256=O.digest(ROOT/'source_manifest.json'),permit_sha256=O.digest(ROOT/'GPU_PERMIT.json'),core_manifest_sha256=core,formal_completed=60,reused=54,new_completed=6,original_uninterrupted_60=False,extra_result_sha256=O.digest(HERE/'RESULT.json'),second_review_sha256=O.digest(HERE/'SECOND_INDEPENDENT_REVIEW.json'),supplements=proofs,final_archive_verification=final_check,scientific_source_or_primary_seal_modified=False,GPU_kernel_calls=0,new_scores=0,model_loads=0)
        O.write_new(HERE/'COMPLETION.json',O.canonical(value)+b'\n');print(json.dumps(value),flush=True)
    except BaseException as exc:
        O.write_new(HERE/'PUBLICATION_FIRST_FAILURE.json',O.canonical(dict(status='EXTRA_CPU_PUBLICATION_FAILURE_CLAIM_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),GPU_calls=0))+b'\n');raise

if __name__=='__main__':main()
