"""Fresh local hashes of immutable earlier review and authorization evidence."""
import json
from pathlib import Path

CURRENT_REVIEW_PINS = {
    'FORMAL_START_FOUR_GPU_FULL_INDEPENDENT_REVIEW.json':'5442e7aa943d3fda000cbeb4f8107e059e93546fa26b08d35d2290b5bffd0579',
    'FORMAL_START_FOUR_GPU_ARCHIVE29_INDEPENDENT_REVIEW.json':'fc3c7f95405ed03c7818e968e2be8ccfd0b9d7efcd03c94033baab1627088fab',
    'FORMAL_START_FOUR_GPU_COMPLETION_REVIEW.json':'7925a01fc67ce9293a2706525bd8ed88c0698a2b5b3deda75c532f393d602237',
    'INITIAL_FOUR_GPU_LIFECYCLE_INDEPENDENT_REVIEW.json':'83b5771c9f4bed64bb8184e2350801e5d4d14128647b6dd2bd4ac24e555b8d50',
    'INITIAL_SOURCE_BOOTSTRAP29_INDEPENDENT_REVIEW.json':'88077895024ef95cc3b09ff52668a29315c6c25bc2ac0890aeb617daffdae503',
    'INITIAL_CLOSED_LINEAGE55_INDEPENDENT_REVIEW.json':'c4ee873be10670816a52c2b033a7ae4612a88c352e698497284e7b32c427b22f',
    'INITIAL_CLOSED_LINEAGE29_INDEPENDENT_REVIEW.json':'93c34e11950a00ecdc662150d26f39d84170a3204e01948ad88920e006a2e661',
}

def review(root,here,O):
    base=root.parent
    lineage=json.loads((root/'provenance/CLOSED_LINEAGE_PINS.json').read_bytes())
    source=json.loads((root/'source_manifest.json').read_bytes())
    frozen_hashes={x['sha256'] for x in source['files']}
    records={}
    def check(path,expected=None):
        path=Path(path)
        O.require(path.is_file() and not path.is_symlink(),'Missing local immutable evidence: '+str(path))
        digest=O.digest(path)
        if expected:O.require(digest==expected,'Local immutable evidence differs: '+str(path))
        row=dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
        records[str(path)]=row
        return row
    for name,pin in CURRENT_REVIEW_PINS.items():
        row=check(root/'operations'/name,pin)
        O.require(json.loads(Path(row['path']).read_bytes())['status'].startswith('PASS'),'Current formal review not PASS')
    previous=base/'PACE_LIFECYCLE_FINAL_REVIEW_20261027'
    inventory=json.loads((previous/'SUPPLEMENT_LOCAL_INVENTORY.json').read_bytes())['files']
    frozen_previous=root/'provenance/PREVIOUS_LIFECYCLE_EXTRA_INDEPENDENT_REVIEW'
    check(previous/'SUPPLEMENT_LOCAL_INVENTORY.json',O.digest(frozen_previous/'SUPPLEMENT_LOCAL_INVENTORY.json'))
    prefixes={
        'extra_final_review':previous,
        'local_operations':base/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_LIFECYCLE_REPAIR_20261026/operations',
        'rebalance_authorization':base/'PACE_FOUR_GPU_REBALANCE_PLANNING_20261027',
        'separate_human_requested_partial_score53':base/'PACE_MANUAL_PARTIAL_SCORE_20261027_LIFECYCLE_V1',
    }
    for x in inventory:
        parts=Path(x['path']).parts;O.require(parts[0] in prefixes,'Unexpected previous evidence prefix')
        row=check(prefixes[parts[0]]/Path(*parts[1:]),x['sha256']);O.require(row['bytes']==x['bytes'],'Previous evidence size differs')
    O.require(len(inventory)==22,'Previous final extra evidence inventory count differs')
    for name in ('RESULT.json','SUPPLEMENT_VERIFICATION.json','RECOVERY46_VERIFICATION.json','AUDIT55.json','AUDIT29.json','MAC_AND_ORIGIN_VERIFICATION.json'):
        check(previous/name,O.digest(frozen_previous/name))
    old_reviews=[]
    for pin in lineage['roots']:
        folder=base/Path(pin['archive']).name
        if not folder.is_dir():continue
        for pattern in ('FORMAL_START*REVIEW*.json','INITIAL*INDEPENDENT*REVIEW*.json'):
            for p in sorted((folder/'operations').glob(pattern)):
                row=check(p)
                row=dict(row,recorded_status=json.loads(p.read_bytes()).get('status'),original_baseline_pin_available=row['sha256'] in frozen_hashes or any(x['sha256']==row['sha256'] for x in inventory))
                old_reviews.append(row)
    partial=json.loads((base/'PACE_MANUAL_PARTIAL_SCORE_20261027_LIFECYCLE_V1/RESULT.json').read_bytes())
    O.require(partial['N']==53 and partial['partial_descriptive_only'] and partial['full60_accuracy'] is False and partial['production_source_and_queue_unchanged'],'Old human-requested partial scope differs')
    for child in partial['children']:
        O.require(child['actual_reaped'] and child['actual_returncode']==0 and child['identity_gone'],'Old partial CPU owning child unresolved')
    plan=base/'PACE_FOUR_GPU_REBALANCE_PLANNING_20261027'
    for name in ('CPU_PROMPT_TRANSPORT_SCOPE_NOTES.json','CPU_PREPARATION_SCOPE_NOTES.json','HUMAN_ACCEPTED_REBALANCE.json','ORIGINAL_CONTROLLER_STOP_REQUEST_RECEIPT.json','PROPOSED_REBALANCE.json'):
        p=plan/name
        if p.exists():check(p)
    O.require(json.loads((plan/'HUMAN_ACCEPTED_REBALANCE.json').read_bytes())['human_answer']=='接受，等q24完成后重排四卡','Migration authorization differs')
    value=dict(status='PASS_ALL_LOCAL_START_REVIEWS_PREVIOUS22_MIGRATION_AND_53_PARTIAL_RECORDS_FRESH_SHA',files=list(records.values()),current_review_count=len(CURRENT_REVIEW_PINS),old_start_review_count=len(old_reviews),old_start_reviews=old_reviews,historical_review_statuses_preserved=True,no_invented_original_baseline_pins=True,old_final_review_supplement_files=22,partial_score_scope='53 original completed subset, descriptive only, no additional scoring',GPU_calls=0,production_mutations=0)
    O.write_new(here/'LOCAL_EXTRA_EVIDENCE_VERIFICATION.json',O.canonical(value)+b'\n')
    return value
