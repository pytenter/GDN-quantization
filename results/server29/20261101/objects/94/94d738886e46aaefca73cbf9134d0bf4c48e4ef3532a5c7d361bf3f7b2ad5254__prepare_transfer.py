#!/usr/bin/env python3
"""Freeze a project-unused, pinned GSM8K panel and official scorer before C7 transfer."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'protocol'
REV='3101c7d5072418e28b9008a6636bde82a006892c'
AIME=Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl')
DATA=P/'gsm8k_test_pinned.jsonl'
OFFICIAL=P/'gsm8k_official_dataset.py'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(text):return ' '.join(text.split())
def write(p,obj):
    if p.exists():raise RuntimeError(f'frozen output already exists {p}')
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def main():
    if sha(DATA)!='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14':raise RuntimeError('pinned dataset SHA mismatch')
    spec=importlib.util.spec_from_file_location('official_gsm8k_dataset',OFFICIAL)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    tests=[('#### 42','#### 42',True),('#### 1,234','#### 1234',True),('#### -2.5','#### -2.5',True),('No answer','#### 42',False),('#### 41','#### 42',False)]
    for output,gold,expected in tests:
        assert official.is_correct(output,{'answer':gold})==expected
    data=[json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    assert len(data)==1319
    aime=[json.loads(line) for line in AIME.read_text().splitlines() if line.strip()]
    aime_hashes={hashlib.sha256(norm(r['problem']).encode()).hexdigest() for r in aime}
    clean=[];excluded=[];seen=set()
    for i,row in enumerate(data):
        problem_sha=hashlib.sha256(norm(row['question']).encode()).hexdigest()
        pid=f'gsm8k-test-{i:04d}'
        if problem_sha in aime_hashes or problem_sha in seen:
            excluded.append(pid);continue
        seen.add(problem_sha)
        if official.extract_answer(row['answer'])==official.INVALID_ANS:raise RuntimeError('invalid gold answer')
        clean.append({'problem_id':pid,'question':row['question'],'answer':row['answer'],'problem_sha256':problem_sha,'id_sha256':hashlib.sha256(pid.encode()).hexdigest()})
    clean.sort(key=lambda r:r['id_sha256'])
    chosen=clean[:160]
    assert len(chosen)==160
    for i,row in enumerate(chosen):row['partition']='TRANSFER_SCREEN' if i<32 else 'TRANSFER_CHECK' if i<64 else 'END_TO_END_CONFIRM'
    panel=P/'reasoning_panel.jsonl'
    if panel.exists():raise RuntimeError('panel already frozen')
    panel.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in chosen))
    protocol={
      'status':'FROZEN_BEFORE_C7_TRANSFER',
      'dataset':'GSM8K official main test', 'repository':'https://github.com/openai/grade-school-math','revision':REV,
      'dataset_sha256':sha(DATA),'official_scorer_source_sha256':sha(OFFICIAL),'panel_sha256':sha(panel),
      'original_records':1319,'eligible_records':len(clean),'excluded_exact_overlap_or_duplicate_ids':excluded,
      'problem_ids':{s:[r['problem_id'] for r in chosen if r['partition']==s] for s in ['TRANSFER_SCREEN','TRANSFER_CHECK','END_TO_END_CONFIRM']},
      'project_prior_usage_audit':{'scoped_roots':['/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments','/data/zypan/results','/data/zypan/reports','/root/autodl-tmp/GDN-formal-c5-c6-h64-v1/experiments'], 'local_GSM8K_inventory':'NOT_FOUND','prior_GSM8K_experiment_references':'NOT_FOUND','old_MATH500_screen_check':'excluded by selecting a distinct benchmark'},
      'AIME_exact_normalized_problem_overlap':{'count':0,'AIME_dataset_sha256':sha(AIME),'rule':'SHA256 of whitespace-normalized raw problem strings; prompt prefix excluded'},
      'model_pretraining_contamination':'NOT_ASSESSED',
      'prompt':{'system':None,'user_suffix':'\nSolve the problem step by step. Finish your response with a final line in the exact format #### <number>.','template':'canonical tokenizer.apply_chat_template with enable_thinking=True and add_generation_prompt=True'},
      'generation':{'runtime':'canonical manual HF single-GPU Qwen path','seed':1,'do_sample':False,'max_new_tokens':32768,'stop':'canonical EOS only','precision':'BF16 model / FP32 rotated recurrent state','conditions_have_identical_decoding_policy':True,'early_stop_on_claim':False},
      'scorer':'unmodified pinned official grade_school_math.dataset.is_correct/extract_answer; exact string numeric claim with comma removal',
      'scorer_self_tests':'PASS',
      'trajectory_freeze':'FP-only fixed generation for SCREEN/CHECK; freeze token IDs, text, length, finish reason and hashes before candidate replay',
      'capture_schedule':{'absolute_generated_positions_zero_based':[31,127,511,1023,2047,4095,8191,16383,32767],'normalized_generated_positions':[0.05,0.25,0.5,0.75,0.95],'rule':'take valid unique union; each problem has equal aggregate weight; no skipped intermediate updates'},
      'T1':'C7 selected final-hidden relative MSE improves over H on Shared16 and both frozen transfer splits; no checkpoint tuning',
      'non_AIME_e2e_gate':'only T1','AIME_gate':'C7 paired rescues strictly greater than regressions versus H on non-AIME confirm',
      'primary_paired_test':'two-sided exact paired binomial/McNemar; paired problem bootstrap 10000 replicates seed0 for effect uncertainty',
      'no_posthoc_weighted_score':True,
    }
    write(P/'transfer_preregistration.json',protocol)
    print(json.dumps({'status':'PASS','records':160,'partition_counts':[32,32,96],'overlap':0,'panel_sha256':sha(panel)}))

if __name__=='__main__':main()
