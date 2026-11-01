"""One fresh local exact-math scoring pass; gold never leaves this CPU host."""
import argparse,hashlib,json,math,os,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scoring'),str(ROOT/'operations')]
from exact_math_scorer import grade,VERSION
from transport import run as remote_cpu

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def compare(rows,a,b):
    aa={r['problem_idx']:r for r in rows if r['config_id']==a};bb={r['problem_idx']:r for r in rows if r['config_id']==b}
    if len(aa)!=33 or aa.keys()!=bb.keys():return dict(status='UNAVAILABLE_CONFIG_NOT_RUN',A=a,B=b,N=0)
    n=dict(both_correct=0,A_only_correct=0,B_only_correct=0,both_wrong=0)
    for k in sorted(aa):
        x,y=aa[k]['correct'],bb[k]['correct']
        n['both_correct' if x and y else 'A_only_correct' if x else 'B_only_correct' if y else 'both_wrong']+=1
    total=n['A_only_correct']+n['B_only_correct'];small=min(n['A_only_correct'],n['B_only_correct'])
    p=min(1,2*sum(math.comb(total,k) for k in range(small+1))/2**total) if total else 1.0
    return dict(status='AVAILABLE_EXPLORATORY',A=a,B=b,N=33,**n,net_A_minus_B=n['A_only_correct']-n['B_only_correct'],accuracy_difference=(n['A_only_correct']-n['B_only_correct'])/33,exact_McNemar_pvalue=p)
def summarize(rows):
    arms={}
    fp=sum(r['correct'] for r in rows if r['config_id']=='C0')
    for c in ['C0','C1','C2','C3']:
        rr=[r for r in rows if r['config_id']==c]
        if not rr:arms[c]=dict(N=0,status='NOT_RUN_STOP_ROTATION_PARITY_FAILED',correct=None,accuracy=None);continue
        assert len(rr)==33
        lengths=[r['generated_token_count'] for r in rr]
        arms[c]=dict(N=33,status='COMPLETE33',correct=sum(r['correct'] for r in rr),accuracy=sum(r['correct'] for r in rr)/33,
            delta_correct_vs_FP=sum(r['correct'] for r in rr)-fp,mean_tokens=statistics.mean(lengths),median_tokens=statistics.median(lengths),
            EOS=sum(r['finish_reason']=='eos' for r in rr),length_cap=sum(r['finish_reason']=='length' for r in rr),
            mean_repeated_32gram_ratio=statistics.mean(r['repeated_32gram_ratio'] for r in rr),
            pooled_generation_TPS=sum(lengths)/sum(r['generation_wall_s'] for r in rr),
            pooled_decode_forward_TPS=sum(r['actual_decode_forwards'] for r in rr)/sum(r['decode_and_sampling_s'] for r in rr))
    return dict(arms=arms,primary_A=compare(rows,'C1','C0'),primary_B=compare(rows,'C2','C1'),primary_C=compare(rows,'C3','C1'))

def verify_remote_decode(root,source_sha,permit_sha):
    # Remote code sees only frozen token IDs and original tokenizer, no gold.
    code=r'''
import hashlib,json,os,sys
from pathlib import Path
r=Path(REMOTE_ROOT);sys.path[:0]=[str(r/'src'),'/data/zypan/transformers-qwen35']
import execution_common as C
C.source(r,SOURCE_SHA);assert C.sha(r/'GPU_PERMIT.json')==PERMIT_SHA
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',local_files_only=True,trust_remote_code=True)
rows=[]
for j in json.loads((r/'FORMAL_JOBS.json').read_text())['jobs']:
 p=r/'formal'/j['id']/'RESULT.json';x=json.loads(p.read_text());v=x['result'];ids=v['generated_token_ids']
 full=t.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 response=t.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 assert full==v['full_output'] and response==v['response']
 rows.append(dict(unit=j['id'],RESULT_sha256=C.sha(p),full_output_sha256=hashlib.sha256(full.encode()).hexdigest(),token_count=len(ids)))
print(json.dumps(dict(status='PASS_FRESH_CPU_ORIGINAL_TOKENIZER_SAME_PAYLOAD',rows=rows,visible_GPU='',gold_accessed=False)))
'''
    binding='REMOTE_ROOT='+repr(str(root))+'\nSOURCE_SHA='+repr(source_sha)+'\nPERMIT_SHA='+repr(permit_sha)+'\n'
    return remote_cpu(binding+code,timeout=None)
def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    local=Path(a.payload_root);remote=Path(a.remote_root)
    cpu_manifest=json.loads((ROOT/'CPU_SCORING_SOURCE_MANIFEST.json').read_text())
    for x in cpu_manifest['files']:
        p=ROOT/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    assert sha(local/'FORMAL_SOURCE_MANIFEST.json')==a.source_sha and sha(local/'GPU_PERMIT.json')==a.permit_sha
    verified=verify_remote_decode(remote,a.source_sha,a.permit_sha);checks={x['unit']:x for x in verified['rows']};assert len(checks)==99
    gold={x['problem_idx']:x['answer'] for x in map(json.loads,(ROOT/'scoring/GOLD_CPU_ONLY.jsonl').read_text().splitlines())}
    requests={x['problem_idx']:x for x in map(json.loads,(ROOT/'protocol/HMMT_REQUESTS.jsonl').read_text().splitlines())}
    rows=[]
    for j in json.loads((local/'FORMAL_JOBS.json').read_text())['jobs']:
        target=local/'formal'/j['id'];p=target/'RESULT.json';x=json.loads(p.read_text());assert sha(p)==checks[j['id']]['RESULT_sha256']
        digest=x.pop('payload_sha256');assert objsha(x)==digest and x['job']==j
        assert x['source_manifest_sha256']==a.source_sha and x['permit_sha256']==a.permit_sha
        close=json.loads((target/'OWNING_CLOSED.json').read_text());arc=json.loads((target/'ARCHIVE29.json').read_text())
        assert close['owning_Popen_exit']==0 and close['release_observed'] and close['RESULT_sha256']==sha(p) and arc['status']=='PASS_29_DURABLE_NOREPLACE'
        r=x['result'];ids=r['generated_token_ids'];assert len(ids)==r['generated_token_count'] and objsha(ids)==r['generated_token_ids_sha256']
        assert r['actual_decode_forwards']==len(ids)-1 and r['seed']==1 and r['status']=='COMPLETE'
        text=r['full_output'];assert hashlib.sha256(text.encode()).hexdigest()==checks[j['id']]['full_output_sha256']
        score=grade(text,gold[j['problem_idx']],r['finish_reason'])
        count=max(0,len(ids)-31);repeat=(count-len({tuple(ids[i:i+32]) for i in range(count)}))/count if count else 0.0
        rows.append(dict(unit=j['id'],problem_idx=j['problem_idx'],problem_type=requests[j['problem_idx']]['problem_type'],config_id=j['config_id'],
            seed=1,correct=score['correct'],parser_status=score['parser_status'],score=score,
            generated_token_count=len(ids),finish_reason=r['finish_reason'],generation_wall_s=r['wall_seconds'],decode_and_sampling_s=r['decode_and_sampling_seconds'],actual_decode_forwards=r['actual_decode_forwards'],
            repeated_32gram_ratio=repeat,RESULT_sha256=sha(p),raw_payload_sha256=digest,source_artifact=str(p)))
        print(json.dumps(dict(phase='LOCAL_EXACT_SCORE_SAME_PAYLOAD',count=len(rows),unit=j['id'])),flush=True)
    assert len(rows)==99
    output=dict(status='PASS99_ADMITTED_LOCAL_CPU_EXACT_SCORE_C2_NOT_RUN',scorer=VERSION,source_sha256=a.source_sha,permit_sha256=a.permit_sha,
        CPU_scoring_manifest_sha256=sha(ROOT/'CPU_SCORING_SOURCE_MANIFEST.json'),rows=rows,summary=summarize(rows),
        tokenizer_fresh_CPU_decode=verified,unresolved_rule='PARSE_FAILED/PARSE_TIMEOUT/UNRESOLVED and extraction failures count incorrect; frozen before formal')
    with Path(a.output).open('x') as f:json.dump(output,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--payload-root',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--output',required=True);main(p.parse_args())
