"""Independent CPU FrozenV4 over preserved original15 plus this attempt's legal remainder."""
import os,sys,json,importlib.util
from pathlib import Path
R=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(R/'src'));import execution_control as C
from record_validation import validate
C.verify_sources(R);C.model_guard(R)
session=C.read(R/'SESSION_RESULT.json');old=Path(C.read(R/'ROOT.json')['original_root'])
sources=C.read(R/'provenance/P1_LEGAL_PAYLOAD_MANIFEST.json');original_jobs={j['id']:j for j in C.read(old/'FORMAL_JOB_MANIFEST.json')['jobs']}
new_jobs={j['id']:j for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']}
spec=importlib.util.spec_from_file_location('frozen_v4',R/'protocol/aime26_scorer_v4.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
gold={x['problem_id']:x for x in map(json.loads,(R/'protocol/aime26_frozen.jsonl').read_text().splitlines())}
sys.path.insert(0,'/data/zypan/transformers-qwen35/src');from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
rows=[]
def score(root,job,provenance):
 p=root/'formal'/job['id']/'RESULT.json';interval=C.read(p.parent/'RESOURCE_INTERVAL.json')
 C.require(interval['exit_returncode']==0 and interval['release']=='PASS' and not C.same_process(interval['identity']),'Score only actual exited/released legal sample')
 wrapper=C.read(p);row=validate(wrapper,job,root)
 C.require(tok.decode(row['generated_token_ids'],skip_special_tokens=True)==row['response'],'Exact payload CPU re-decode differs')
 g=gold[job['problem_id']];x=m.extract_v4(row['response'],row['finish_reason'],g['problem'])
 return dict(problem_id=job['problem_id'],sample_id=job['id'],condition='P1',seed=1,correct=bool(m.compare_with_gold(x,g['answer'])),extracted_answer=x.normalized_value,extraction=x.to_dict(),generated_token_count=row['generated_token_count'],finish_reason=row['finish_reason'],gpu_uuid=row['gpu_uuid'],wall_seconds=row['wall_seconds'],TPS=row['TPS'],payload_sha256=wrapper['payload_sha256'],source=provenance,result_sha256=C.sha(p))
for x in sources['records']:
 for f in x['files']:
  p=old/f['path'];C.require(p.stat().st_size==f['bytes'] and C.sha(p)==f['sha256'],'Original completed source changed')
 rows.append(score(old,original_jobs[x['unit']],'ORIGINAL_PRE_CANCELLATION'))
for unit in session['completed']:rows.append(score(R,new_jobs[unit],'HUMAN_AUTHORIZED_GPU0_REMAINDER'))
rows.sort(key=lambda x:x['problem_id']);C.require(len(rows)==15+session['count'] and len({x['problem_id'] for x in rows})==len(rows),'Duplicate/missing completed question')
C.atomic_new(out,dict(status='PASS_INDEPENDENT_CPU_ORIGINAL_FROZEN_V4_SAME_TOKEN_REDECODE',scores=rows,scorer_sha256=C.sha(R/'protocol/aime26_scorer_v4.py'),old_cancellation_unchanged=True))
