#!/usr/bin/env python3
"""Complete only Git/audit after valid CSV CRLF blocked default whitespace check. No model imports."""
import json
from pathlib import Path
import subprocess
import posttrain_closure_audit as A

ROOT=A.ROOT
REPO=A.REPO
CRLF_CHECK=['git','--no-pager','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check']

def run(args):return subprocess.check_output(args,cwd=REPO,text=True).strip()

def main():
    A.require(run(['git','rev-parse','HEAD'])==A.HEAD,'HEAD changed; stop')
    A.require(not (ROOT/'logs/posttrain_closure_commit.json').exists(),'commit already recorded; stop')
    A.require(A.read('analysis/posttrain_closure_gates.json')['status']=='PASS','scientific gates not PASS')
    failure=A.read('logs/posttrain_closure_finalize.error.json')
    A.require("['git', 'diff', '--cached', '--check']" in failure['error'],'different failure; stop')
    staged=run(['git','diff','--cached','--name-only']).splitlines()
    prefix=str(ROOT.relative_to(REPO))+'/'
    A.require(len(staged)==33 and all(p.startswith(prefix) for p in staged),'unexpected preexisting staged scope')
    A.require(not run(['git','diff','--name-only']),'tracked mutation; stop')
    raw=subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    warnings=[l for l in raw.stdout.splitlines() if ': trailing whitespace.' in l]
    A.require(raw.returncode==2 and len(warnings)==1091,'different whitespace failure; stop')
    csvs={l.rsplit(':',2)[0] for l in warnings};counts={}
    for rel in sorted(csvs):
        path=REPO/rel;lines=path.read_bytes().splitlines(keepends=True)
        A.require(path.suffix=='.csv' and all(l.endswith(b'\r\n') and l[:-2].rstrip(b' \t')==l[:-2] for l in lines),'actual trailing spaces or non-CRLF; stop')
        counts[rel]={'records_including_header':len(lines),'sha256':A.sha(path),'standard_CRLF_only':True,'no_actual_trailing_space_or_tab':True}
    A.require(sum(x['records_including_header'] for x in counts.values())==1091,'warning count mismatch')
    A.require(run(CRLF_CHECK)=='','format-aware whitespace check failed')
    original=ROOT/'manifests/posttrain_closure_artifact_sha256.txt';hashes={}
    for i,line in enumerate(original.read_text().splitlines()):
        digest,path=line.split('  ',1);actual=A.identity(path,digest);hashes[path]=actual
        if i%50==0:print(json.dumps({'event':'completion_hash_recheck','files':i+1}),flush=True)
    pre=A.read('provenance/preclosure_training_artifact_inventory.json')
    A.require(all(hashes[r['path']]==r for r in pre['files']),'protected byte/size/mtime changed')
    for p in ROOT.rglob('*.json'):json.loads(p.read_text())
    audit={'status':'PASS','cause':'Python csv.DictWriter standard CRLF; default Git whitespace check classified CR as trailing blank','default_check_exitcode':2,'default_warning_count':1091,'files':counts,'format_aware_check_command':CRLF_CHECK,'format_aware_check_exitcode':0,'permanent_git_configuration_changed':False,'files_reformatted':0,'scientific_computation_repeated':False,'original_failure_preserved':A.identity(ROOT/'logs/posttrain_closure_finalize.error.json'),'original_pipeline_exitcode_preserved':(ROOT/'logs/posttrain_closure_pipeline.exitcode').read_text().strip(),'original_manifest_reverified_files':len(hashes),'protected_inventory_reverified_count':len(pre['files']),'TRAINING_ARTIFACT_BYTE_IDENTITY':'PASS','no_new_GPU_work':True,'semantic_or_numerical_retry':False,'backward_calls':0,'optimizer_step_calls':0,'3090_deployment_started':False,'new_AIME_generation':0}
    A.save('analysis/posttrain_git_crlf_check_closure.json',audit)
    supplement=ROOT/'reports/QWEN_A800_POSTTRAIN_SHARED16_REPRODUCTION_CLOSURE_V1_GIT_ADDENDUM.md'
    with supplement.open('x') as f:
        f.write('# Git finalization addendum\n\nAll scientific gates and all 112 validation documents completed PASS before the original finalizer stopped. The only failure was Git default whitespace interpretation of standard CSV CRLF (1091 lines in four files). The original exitcode=1/error/source/manifest are preserved. No runner, validation, export, teacher cache or training was repeated or changed.\n\nA one-command Git `core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol` override correctly recognizes CRLF while retaining actual trailing-space and tab checks. It passes; no permanent configuration or scientific data formatting changed. All 526 manifest entries and 500 protected files are reverified; size/mtime unchanged. JSON parsing passes.\n\nThe one local closure commit recorded in logs/posttrain_closure_commit.json completes this task; the earlier failed pipeline exitcode is historical, not overwritten. The supplementary completion hash manifest covers this addendum, the preserved finalization failure and the completion helper/audit. The post-commit identity log is excluded from pre-commit manifests to avoid self-reference. No push, no deployment, no AIME. STOP.\n')
    extras=[Path(__file__),ROOT/'analysis/posttrain_git_crlf_check_closure.json',supplement,ROOT/'logs/posttrain_closure_finalize.error.json',ROOT/'logs/posttrain_closure_pipeline.exitcode',original]
    completion=ROOT/'manifests/posttrain_closure_completion_artifact_sha256_v1.txt'
    with completion.open('x') as f:
        for p in sorted(extras):f.write(A.sha(p)+'  '+str(p)+'\n')
    for line in completion.read_text().splitlines():
        digest,path=line.split('  ',1);A.require(A.sha(path)==digest,'supplement hash changed')
    newfiles=[str(p.relative_to(REPO)) for p in [*extras,completion]]
    subprocess.run(['git','add','-f','--',*newfiles],cwd=REPO,check=True)
    finalscope=run(['git','diff','--cached','--name-only']).splitlines()
    A.require(set(finalscope)==set(staged)|set(newfiles),'unexpected final staged scope')
    check=run(CRLF_CHECK);stat=run(['git','--no-pager','diff','--cached','--stat'])
    A.require(not run(['git','diff','--name-only']),'unstaged source changes')
    subprocess.run(['git','commit','-m','Close A800 Shared16 reproduction and export audits; preserve hash and CRLF incidents; stop before endpoints'],cwd=REPO,check=True)
    A.save('logs/posttrain_closure_commit.json',{'status':'PASS','branch':run(['git','branch','--show-current']),'pre_task_HEAD':A.HEAD,'new_commit_SHA':run(['git','rev-parse','HEAD']),'git_status':run(['git','status','--short']),'diff_check':check,'diff_check_command':CRLF_CHECK,'diff_stat':stat,'staged_scope':finalscope,'artifact_root':str(ROOT),'push_performed':False,'original_failed_pipeline_not_restarted':True,'stop':True})
    print('CLOSURE_COMPLETE_STOP '+run(['git','rev-parse','HEAD']),flush=True)

if __name__=='__main__':main()
