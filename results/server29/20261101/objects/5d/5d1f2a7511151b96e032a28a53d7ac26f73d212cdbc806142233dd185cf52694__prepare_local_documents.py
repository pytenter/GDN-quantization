"""Local-only prospective corpus audit/tokenization; never loads a model.

No network, GPU, recurrence, rotation metric, outcomes or Confirmation replay.
Fail closed if the frozen local corpus inventory has no eligible four-doc pool.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

TASK = 'QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT = REPO / 'experiments' / TASK
OLD = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1')
PREV_REPO = Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1')
PARENT = OLD/'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'
CUMSUM = OLD/'experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1'
PREV = PREV_REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1'
AMEND = PREV/'amendments/PREMEASUREMENT_INPUT_FREEZE_AMENDMENT_V1'
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
CACHE = Path('/data/ydai/qwen_model_cache/datasets')
CANON = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name, value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return p
def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(value)
    return p
def cmd(args,cwd=REPO):
    p=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return dict(command=args,cwd=str(cwd),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def seal(root,name):
    manifest=root/name;rows=[]
    for line in manifest.read_text().splitlines():
        expected,rel=line.split(None,1);rel=rel.lstrip('*');p=root/rel
        assert p.resolve().is_relative_to(root.resolve())
        actual=sha(p) if p.is_file() else None
        rows.append(dict(relative_path=rel,expected_sha256=expected,actual_sha256=actual,PASS=expected==actual))
    return dict(root=str(root),manifest=str(manifest),manifest_sha256=sha(manifest),sealed_artifact_count=len(rows),
                all_exact=all(r['PASS'] for r in rows),rows=rows)
def history():
    items={'parent':seal(PARENT,'FINAL_SHA256SUMS'),'cumsum':seal(CUMSUM,'FINAL_SHA256SUMS'),
           'preflight':seal(PREV,'SHA256SUMS'),'amendment':seal(AMEND,'SHA256SUMS')}
    assert all(s['all_exact'] for s in items.values()),'historical seal mismatch: STOP'
    pv=read(PARENT/'analysis/final_verdict.json');cv=read(CUMSUM/'provenance/final_verdict.json');av=read(PREV/'final_verdict.json');mv=read(AMEND/'final_verdict.json')
    assert pv['TASK_STATUS']=='STOP_STRICT_DETERMINISTIC_RUNTIME_UNAVAILABLE'
    assert cv['verdict']['DETERMINISTIC_CUMSUM_COMPATIBILITY_GATE']=='FAIL'
    assert av['verdict']['FINAL_ACTION']=='FAIL_AND_STOP'
    assert mv['verdict']['DISCOVERY_DOCUMENT_RULE_RECOVERY_GATE']=='FAIL'
    assert av['scientific_measurements']==0 and mv['verdict']['SCIENTIFIC_METRIC_EXECUTION_COUNT']==0
    assert pv['DISCOVERY_COMPLETE']=='0/48' and read(PARENT/'analysis/discovery_summary.json')['rotation_conditioned_metrics_computed'] is False
    return dict(OLD_EVIDENCE_INTEGRITY_GATE='PASS',SCIENTIFIC_MEASUREMENT_COUNT=0,
        SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_NEW_EXPERIMENT=0,seals=items,
        unchanged_verdicts=dict(parent=pv,cumsum=cv,preflight=av,amendment=mv),
        failed_amendment_audit_commit=cmd(['git','rev-parse','196e1da'],PREV_REPO)['stdout'].strip(),
        new_repo=str(REPO),new_branch=cmd(['git','rev-parse','--abbrev-ref','HEAD'])['stdout'].strip(),
        new_base_commit=cmd(['git','rev-parse','HEAD'])['stdout'].strip(),
        authoritative_original_paths_used=True,CONFIRMATION_EXECUTED=0)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('HF_DATASETS_OFFLINE')=='1'
    before=history();save('preflight_historical_evidence.json',before)
    # New selection procedure is authorized by the NEW task, never a recovered legacy rule.
    policy=dict(EXPERIMENT=TASK,NEW_PROSPECTIVE_DISCOVERY_DOCUMENT_PROTOCOL='YES',
        prospective_policy_written_before_tokenization=True,SCIENTIFIC_MEASUREMENT_COUNT=0,
        corpus_hierarchy=['A: available intended fresh parent corpus','B: unique eligible historical Qwen mechanism corpus',
                          'C: eligible local non-AIME non-training corpus with greatest eligible count; lexicographic descriptor tie'],
        stable_descriptor='<dataset_name>|<split>|<revision_or_manifest_sha>',
        minimum_tokens=32768,normalization='Exact decoded source text; no stripping, joining, repeating, chat template or semantic filtering',
        tokenizer='Exact local production Qwen tokenizer; encode(add_special_tokens=False); no truncation for eligibility',
        candidate_identity='Arrow immutable container SHA + record index + dataset document ID if provided; context/text SHA',
        duplicate_policy='Keep one identical-content representative with lexicographically smallest canonical_doc_id; others DUPLICATE',
        exclusions='AIME, Shared16, project rotation training/validation/calibration, used outcome-bearing mechanism docs, reserved confirmation docs',
        selection_namespace=TASK,selection_key='SHA256(namespace+"|"+candidate_pool_revision+"|"+canonical_doc_id)',
        ordering='selection_key ascending',hash_collision_policy='FAIL_AND_STOP if distinct IDs collide',Discovery_count=4,
        future_Confirmation='continue the same hash ranking after the four Discovery docs; do not materialize or execute now',
        fresh_docs_not_found_action='FAIL_AND_STOP; no download/concatenation/repetition or substituted shorter horizon',
        allowed_content_access='Only unreserved local candidate text for identity/eligibility tokenization; no answers, labels or endpoint outcomes',
        scientific_model_forward_count=0)
    save('protocol/prospective_document_protocol.json',policy)
    searches=[]
    for args in [
        ['find','/data/zypan/worktrees','-maxdepth','7','-iname','*pg19*'],
        ['find',str(CACHE),'-maxdepth','7','-iname','*pg19*'],
        ['find','/data/zypan/GDN-quantization','-maxdepth','6','-iname','*pg19*'],
        ['find','/data/zypan/worktrees','-maxdepth','6','-type','f','(','-iname','*raw*text*.jsonl','-o','-iname','*corpus*.jsonl',')'],
        ['find',str(CACHE),'-maxdepth','7','-type','f','-name','*.arrow']]:
        searches.append(cmd(args))
    excluded=[dict(path=str(CANON/'CALIBRATION_RAW_TEXTS.jsonl'),reason='Entire frozen project training/validation/calibration/heldout corpus; never eligible',content_read=False),
              dict(path=str(CANON/'results/rotation/qwen_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl'),
                   reason='Whole confirmation/mechanism panel reserved/used by another Qwen experiment; exclude without reading contents',content_read=False)]
    # PG19 is named as intended source but no local path exists in the surveyed roots.
    pg19_available=any(s['stdout'].strip() for s in searches[:3])
    assert not pg19_available,'PG19 discovered: stop this restricted preparation script and implement Priority A, do not silently use C'
    # Cached non-AIME LongBench contexts are distinct immutable local configurations.
    arrows=sorted((CACHE/'long_bench').glob('*/*/*/*.arrow'))
    assert arrows,'no local context Arrow files: STOP'
    inventory=dict(searches=searches,search_scope='Own project worktrees/data roots and project-shared Qwen cache; unrelated users directories not opened',
        Priority_A='Intended PG19 source named, no PG19 local path found in surveyed roots; unavailable',
        Priority_B='Existing Qwen WikiText calibration and confirmation/mechanism panels excluded wholesale; no eligible independent unique mechanism corpus recovered',
        priority_to_evaluate='C',available_local_Arrow_files=[dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)) for p in arrows],
        excluded_local_corpora=excluded,excluded_cache_names=['gsm8k','simplescaling___s1_k','HuggingFaceH4___aime_2024','math-ai___aime25'],
        excluded_cache_rationale='Reasoning/endpoint rather than fresh long text; AIME forbidden',
        downloaded_corpora=0,other_users_data_contents_read=False)
    save('candidate_corpus_inventory.json',inventory)
    runtime=read(PARENT/'provenance/runtime_identity.json')
    expected={Path(r['path']).name:r['sha256'] for r in runtime['model_identity']['files']}
    token_files=[]
    for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
        actual=sha(MODEL/name);assert actual==expected[name], 'production tokenizer hash mismatch'
        token_files.append(dict(path=str(MODEL/name),sha256=actual))
    from datasets import Dataset
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    save('protocol/tokenizer_identity.json',dict(files=token_files,tokenizer_class=type(tokenizer).__name__,add_special_tokens=False,
        no_chat_template=True,production_identity_gate='PASS',no_model_loaded=True))
    corpus_results=[];all_eligible={}
    for path in arrows:
        subset=path.relative_to(CACHE/'long_bench').parts[0]
        ds=Dataset.from_file(str(path))
        cols=ds.column_names
        assert 'context' in cols, 'context field unavailable; do not substitute question/answers'
        records=[];raw_by_id={}
        # Read ONLY context and optional stable record ID; answers/output/labels never read.
        selected_columns=ds.select_columns(['context']+(['_id'] if '_id' in cols else []))
        for index,row in enumerate(selected_columns):
            text=row['context'];docid=f'long_bench/{subset}/test/'+str(row.get('_id',f'row{index:06d}'))
            assert docid not in raw_by_id,'duplicate canonical document ID'
            valid=isinstance(text,str) and bool(text) and '\x00' not in text
            raw=text.encode('utf-8') if valid else b''
            record=dict(canonical_doc_id=docid,relative_path=path.name+'#row='+str(index),record_index=index,
                source_path=str(path),container_file_size=path.stat().st_size,file_size=len(raw),content_SHA256=hash_bytes(raw),
                readable_nonempty=valid,exclusion_status='NOT_IN_KNOWN_PROJECT_OR_RESERVED_CORPUS')
            records.append(record);raw_by_id[docid]=text if valid else ''
        assert len({r['canonical_doc_id'] for r in records})==len(records)
        manifest=save(f'corpora/{subset}/candidate_pool_manifest.json',dict(dataset_name='long_bench/'+subset,split='test',
            source_container_sha256=sha(path),source_container_path=str(path),document_count=len(records),records=records))
        revision=sha(manifest)
        representatives={}
        for r in sorted(records,key=lambda r:r['canonical_doc_id']):
            representatives.setdefault(r['content_SHA256'],r['canonical_doc_id'])
        eligibility=[];eligible=[]
        for r in records:
            duplicate=representatives[r['content_SHA256']]!=r['canonical_doc_id']
            token_count=len(tokenizer.encode(raw_by_id[r['canonical_doc_id']],add_special_tokens=False)) if r['readable_nonempty'] and not duplicate else None
            ok=r['readable_nonempty'] and not duplicate and token_count>=32768
            entry=dict(**r,token_count=token_count,duplicate=duplicate,eligible=ok)
            eligibility.append(entry)
            if ok:eligible.append(entry)
        save(f'corpora/{subset}/eligible_document_manifest.json',dict(dataset_name='long_bench/'+subset,split='test',CANDIDATE_POOL_REVISION=revision,
            candidate_count=len(records),eligible_count=len(eligible),records=eligibility,selection_metrics_used=False))
        result=dict(dataset_name='long_bench/'+subset,split='test',path=str(path),CANDIDATE_POOL_REVISION=revision,
            candidate_count=len(records),eligible_count=len(eligible),maximum_token_count=max((r['token_count'] or 0) for r in eligibility),
            context_only=True,no_model_forward=True,no_rotation_metric=True)
        corpus_results.append(result);all_eligible[subset]=dict(records=eligible,texts=raw_by_id,revision=revision)
        print(json.dumps(result),flush=True)
    save('eligible_document_manifest.json',dict(corpora=corpus_results,total_eligible=sum(r['eligible_count'] for r in corpus_results),
        minimum_tokens=32768,not_pooled_across_corpus_configurations=True))
    qualified=[r for r in corpus_results if r['eligible_count']>=4]
    if not qualified:
        finish_stop(before,corpus_results)
        return
    qualified.sort(key=lambda r:(-r['eligible_count'],r['dataset_name']+'|'+r['split']+'|'+r['CANDIDATE_POOL_REVISION']))
    chosen=qualified[0];subset=chosen['dataset_name'].split('/')[-1];data=all_eligible[subset]
    ranked=[]
    for record in data['records']:
        key=hash_bytes((TASK+'|'+data['revision']+'|'+record['canonical_doc_id']).encode('utf-8'))
        ranked.append(dict(**record,selection_key=key))
    assert len({r['selection_key'] for r in ranked})==len(ranked),'selection hash collision'
    ranked.sort(key=lambda r:r['selection_key']);picked=[]
    for rank,r in enumerate(ranked[:4],1):
        text=data['texts'][r['canonical_doc_id']];raw=text.encode('utf-8');text_name=f'inputs/discovery_{rank:02d}.txt'
        p=ROOT/text_name;p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f:f.write(raw)
        ids=tokenizer.encode(text,add_special_tokens=False)[:32768]
        tokens=save(f'inputs/discovery_{rank:02d}_tokens.json',ids)
        picked.append(dict(**r,selection_rank=rank,text_path=text_name,text_sha256=sha(p),token_path=str(tokens.relative_to(ROOT)),
            token_sha256=sha(tokens),frozen_token_count=len(ids)))
    save('protocol/frozen_discovery_documents.json',dict(status='SELECTED_AND_FROZEN',DISCOVERY_DOC_COUNT=4,
        DISCOVERY_DOCUMENT_SELECTION_GATE='PASS',CANDIDATE_CORPUS=chosen['dataset_name'],CANDIDATE_POOL_REVISION=data['revision'],documents=picked,
        CONFIRMATION_RULE_FROZEN='YES',CONFIRMATION_EXECUTED=0,confirmation_inputs_materialized=False))
    save('document_preparation_status.json',dict(status='DOCUMENTS_PREPARED_ANCHORS_AND_PREREGISTRATION_PENDING',
        chosen_corpus=chosen,scientific_measurement_count=0,no_scientific_replay_authorized_yet=True))
    print('DOCUMENT_PREPARATION_PASS; stop this CPU preparation script before implementation/gates',flush=True)

def finish_stop(before,corpora):
    verdict=dict(EXPERIMENT=TASK,OLD_PARENT_STOP_PRESERVED='YES',OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES',
        OLD_PREFLIGHT_FAIL_PRESERVED='YES',OLD_INPUT_FREEZE_AMENDMENT_FAIL_PRESERVED='YES',OLD_EVIDENCE_INTEGRITY_GATE='PASS',
        SCIENTIFIC_MEASUREMENT_COUNT_BEFORE_NEW_EXPERIMENT=0,NEW_PROSPECTIVE_DOCUMENT_PROTOCOL='YES',PROSPECTIVE_DOCUMENT_PROTOCOL_GATE='FAIL',
        CANDIDATE_CORPUS='NONE_QUALIFIED',CANDIDATE_POOL_REVISION=None,ELIGIBLE_DOCUMENT_COUNT=sum(r['eligible_count'] for r in corpora),
        DISCOVERY_DOCUMENT_SELECTION_GATE='FAIL',DISCOVERY_DOC_COUNT=0,ANCHOR_CONDITION_COUNT=0,ANCHOR_IDENTITY_GATE='NOT_EVALUATED',
        PREMEASUREMENT_COMMIT_SHA=None,DRIVE_FACTORIZATION_GATE='NOT_EVALUATED',CAPTURE_SCHEMA_GATE='NOT_EVALUATED',
        FP_REPLAY_PARITY_GATE='NOT_EVALUATED',NO_QDQ_ROTATION_INVARIANCE_GATE='NOT_EVALUATED',COUNTERFACTUAL_DECOMPOSITION_GATE='NOT_EVALUATED',
        REPLAY_ISOLATION_GATE='NOT_EVALUATED',REPLAY_DETERMINISM_GATE='NOT_EVALUATED',STREAMING_REDUCER_PARITY_GATE='NOT_EVALUATED',
        DISCOVERY_EXECUTION_AUTHORIZED='NO',DISCOVERY_COMPLETE='0/16',HORIZON_2048_COMPLETE='0/16',HORIZON_8192_COMPLETE='0/16',HORIZON_32768_COMPLETE='0/16',
        CONFIRMATION_RULE_FROZEN='YES',CONFIRMATION_EXECUTED=0,FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',
        NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,NEW_TRAINING='NO',OPTIMIZER_STEP=0,FINAL_ACTION='FAIL_AND_STOP')
    save('final_verdict.json',dict(verdict=verdict,reason='No surveyed allowed local corpus has four eligible independent contexts of32768 exact Qwen tokens; no download/concatenation/substitution authorized',
        corpus_results=corpora,scope='Selected server own/project-referenced local corpora; unrelated users corpora not accessed',
        not_legacy_selection_rule_failure=True,successful_freeze_commit_created=False))
    save('execution_status.json',dict(document_tokenization_only=True,model_forward_count=0,GPU_compute_started=False,drive_capture_count=0,
        replay_units=0,scientific_metric_execution_count=0,confirmation_inputs_materialized=0,anchor_stage='NOT_STARTED',
        no_interpolation_bank=True,no_cumsum_patch=True))
    save('protocol/future_confirmation_rule.json',dict(CONFIRMATION_RULE_FROZEN='YES',rule='Continue same namespace|revision|canonical_doc_id SHA256 ranking after four Discovery documents',
        confirmation_contents_materialized=False,CONFIRMATION_EXECUTED=0,requires_separate_authorization=True))
    report='# New prospective anchor-mechanism experiment: premeasurement STOP\n\n'
    report+='The NEW prospective hash-selection rule is explicitly authorized and documented. This is not a parent resume, legacy-rule recovery or12-condition interpolation study. '
    report+='Document readiness fails because no allowed local corpus in the surveyed own/project-shared roots provides four distinct fresh documents reaching32768 exact production Qwen tokens. '
    report+='No corpus was downloaded, no short documents concatenated/repeated, no token threshold or horizon relaxed.\n\n## Corpus eligibility\n\n'
    for c in corpora:report+=f"- {c['dataset_name']}: {c['eligible_count']}/{c['candidate_count']} eligible; maximum{c['maximum_token_count']} Qwen tokens.\n"
    report+='\n## Scientific boundaries\n\nAll old evidence/verdicts unchanged; audit196e1da retained. No anchors materialized/audited in this stopped attempt; no preregistration freeze commit; no model forward, capture, replay, FD metric, training, AIME or Confirmation. '
    report+='The report makes no history/amplification/cancellation or endpoint-causality conclusion. A later separately authorized input step must provide an existing qualified local corpus; this report permits no download.\n\n## Required questions\n\n'
    answers={'Q1':'YES: new outcome-independent prospective hierarchy, immutable pool revisions and namespace-hash selection documented before eligibility tokenization.',
        'Q2':'NO: no corpus qualifies with four32768-token documents.',
        'Q3':'NOT_EVALUATED: G2 not reached; no substitute anchors used.',
        'Q4':'NOT_EVALUATED: no scientific replay implemented or executed.',
        'Q5':'NOT_EVALUATED: no capture or FP replay parity.',
        'Q6':'NOT_EVALUATED: no vector/energy closure measurement.',
        'Q7':'NOT_EVALUATED: no C6R/H comparisons at any horizon.',
        'Q8':'NOT_EVALUATED: no C7/C6R decomposition.',
        'Q9':'NOT_EVALUATED: no Native/H decomposition.',
        'Q10':'NOT_EVALUATED: no horizon trend measurements.',
        'Q11':'Scope remains intrinsic frozen-drive recurrent dynamics only; full-model reasoning causality NOT_EVALUATED.',
        'Q12':'NO scientific hypothesis established by this input-readiness audit; Confirmation NOT_RUN.'}
    for q,a in answers.items():report+=q+': '+a+'\n\n'
    report+='## Final fields\n\n```text\n'+'\n'.join(k+' = '+('null' if v is None else str(v)) for k,v in verdict.items())+'\n```\n'
    write('FINAL_REPORT.md',report)
    after=history();save('final_historical_seal_revalidation.json',after)
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    write('SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':main()
