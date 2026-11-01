"""Real CPU owning waits and two fresh FrozenV4 payload passes; zero GPU credit."""
import argparse,copy,json,os,shlex,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import execution_common as C
import experiment_parent as P

def run(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    with tempfile.TemporaryDirectory(prefix='unroll_delivery_cpu_',dir='/dev/shm') as td:
        root=Path(td);(root/'runtime').mkdir();(root/'formal').mkdir();(root/'protocol').mkdir()
        original=P.ROOT;P.ROOT=root
        try:
            waits=[]
            for name,body,expected in [('exit0','import time;time.sleep(.1)',0),('exit7','import time,sys;time.sleep(.1);sys.exit(7)',7),('sigterm','import time,os,signal;time.sleep(.1);os.kill(os.getpid(),signal.SIGTERM)',-15),('leaderZ','import ctypes,threading,time;threading.Thread(target=lambda:time.sleep(.18)).start();ctypes.CDLL(None).syscall(60,0)',0)]:
                rc=P.owned([sys.executable,'-B','-c',body],name);assert rc==expected
                waits.append(dict(case=name,owning_Popen_exit=rc))
            bash=[]
            for code,body in [(0,'import time;time.sleep(.1)'),(7,'import time,sys;time.sleep(.1);sys.exit(7)'),(143,'import time,os,signal;time.sleep(.1);os.kill(os.getpid(),signal.SIGTERM)')]:
                command=[sys.executable,'-B','-c',body]
                text=shlex.join(command)+' &\nchild=$!\nwait "$child"\nrc=$?\nprintf "%s\\n" "$rc"\n'
                actual=subprocess.run(['bash','-c',text],capture_output=True,text=True,check=True)
                assert int(actual.stdout)==code;bash.append(dict(actual_Bash_wait=int(actual.stdout),not_inferred=True))
            sys.path.insert(0,'/data/zypan/transformers-qwen35')
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
            ids=tokenizer.encode('The final answer is 123.',add_special_tokens=False);text=tokenizer.decode(ids,skip_special_tokens=True)
            C.atomic(root/'GPU_PERMIT.json',dict(CPU_FIXTURE_ONLY=True));permit=C.sha(root/'GPU_PERMIT.json');source='CPU_SYNTHETIC_NO_FORMAL_CREDIT'
            jobs=[dict(id=f'CPU_NON_AIME_{i:02d}_{p}',problem_id=f'CPU_NON_AIME_{i:02d}',period=p) for i in range(30) for p in (8,16)]
            C.atomic(root/'FORMAL_JOBS.json',dict(jobs=jobs));C.atomic(root/'runtime/FORMAL_SESSION_RESULT.json',dict(status='COMPLETE60_TRUE_EXIT_RELEASE_AND_ARCHIVE',completed=[x['id'] for x in jobs],CPU_FIXTURE_ONLY=True))
            (root/'protocol/aime26_frozen.jsonl').write_text('\n'.join(json.dumps(dict(problem_id=f'CPU_NON_AIME_{i:02d}',problem='Synthetic CPU check, not AIME. Return the fixture value.',answer=123)) for i in range(30)))
            (root/'protocol/aime26_scorer_v4.py').write_bytes((ROOT/'protocol/aime26_scorer_v4.py').read_bytes())
            for job in jobs:
                target=root/'formal'/job['id'];target.mkdir()
                result=dict(status='COMPLETE',seed=1,nonfinite=False,runtime_error=None,generated_token_ids=ids,generated_token_ids_sha256=C.objsha(ids),generated_token_count=len(ids),response=text,finish_reason='eos',wall_seconds=1.0,prompt_seconds=.1,decode_and_sampling_seconds=.9,memory={},quantizer_audit=dict(storage_by_layer={}))
                value=dict(job=job,result=result,model_load_wall_s=0.0,source_manifest_sha256=source,permit_sha256=permit,CPU_FIXTURE_ONLY=True);value['payload_sha256']=C.objsha(value);C.atomic(target/'RESULT.json',value)
                C.atomic(target/'OWNING_CLOSED.json',dict(owning_Popen_exit=0,release_observed=True,RESULT_sha256=C.sha(target/'RESULT.json'),source_sha256=source,permit_sha256=permit,dispatch_wall_s=1.0,CPU_MOCK_ONLY=True))
                C.atomic(target/'ARCHIVE29.json',dict(status='PASS_29_DURABLE_NOREPLACE',CPU_MOCK_ONLY=True))
            scores=[]
            for n in [1,2]:
                dest=root/('SCORE'+str(n)+'.json')
                body='import sys,argparse;from pathlib import Path;sys.path[:0]='+repr([str(ROOT/'scripts'),str(ROOT/'src')])+';import score_cpu as S;S.ROOT=Path('+repr(str(root))+');S.C.source=lambda *a,**kw:None;raise SystemExit(S.run(argparse.Namespace(source_sha='+repr(source)+',permit_sha='+repr(permit)+',output=Path('+repr(str(dest))+'))))'
                rc=P.owned([sys.executable,'-B','-c',body],'fresh_score'+str(n));assert rc==0
                scores.append(json.loads(dest.read_text()))
            assert scores[0]==scores[1] and scores[0]['summary']['arms']['8']['correct']==30 and scores[0]['summary']['arms']['16']['correct']==30
            # Exercise the actual parent->two scorers->reports->manifest->archive
            # control flow. These60 inputs are CPU synthetic fixtures, not AIME.
            for folder in ['src','scripts','configs','calibration','performance','analysis','validation']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts and 'cpu_failure_scopes' not in p.parts:
                        dest=root/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            (root/'operations').mkdir()
            paths=[p for folder in ['src','scripts','configs','calibration','performance','analysis','validation','protocol'] for p in (root/folder).rglob('*') if p.is_file()]
            C.atomic(root/'FORMAL_SOURCE_MANIFEST.json',dict(CPU_FIXTURE_ONLY=True,files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=C.sha(p)) for p in paths]))
            source=C.sha(root/'FORMAL_SOURCE_MANIFEST.json')
            for job in jobs:
                p=root/'formal'/job['id']/'OWNING_CLOSED.json';x=json.loads(p.read_text());x['source_sha256']=source;p.write_text(json.dumps(x))
                result_path=root/'formal'/job['id']/'RESULT.json';value=json.loads(result_path.read_text());value['source_manifest_sha256']=source;value.pop('payload_sha256');value['payload_sha256']=C.objsha(value);result_path.write_text(json.dumps(value))
                x['RESULT_sha256']=C.sha(result_path);p.write_text(json.dumps(x))
            original_publish=P.publish;original_apps=C.apps;published=[]
            def fake_archive(rr,paths,relative):
                assert all((Path(rr)/n).is_file() for n in paths)
                published.append(relative)
                return dict(status='PASS_29_DURABLE_NOREPLACE',CPU_MOCK_ONLY=True,files=len(paths))
            try:
                P.publish=fake_archive;C.apps=lambda:[]
                P.finish(argparse.Namespace(source_sha=source,permit_sha=permit))
            finally:P.publish=original_publish;C.apps=original_apps
            assert published==['FINAL_DELIVERY'] and (root/'paired_results.csv').is_file()
            assert json.loads((root/'DELIVERY_READY.json').read_text())['two_fresh_CPU_scores_equal']
            for line in (root/'MANIFEST.sha256').read_text().splitlines():
                digest,relative=line.split('  ',1);assert C.sha(root/relative)==digest
        finally:P.ROOT=original
    result=dict(status='PASS_REAL_CPU_PARENT_WAITS_BASH_AND_TWO_FRESH_SCORERS',waits=waits,Bash_waits=bash,two_fresh_same_payload_FrozenV4_equal=True,actual_parent_finish_two_more_fresh_scorers_reports_CSV_manifest=True,synthetic_non_AIME_payloads=60,actual_GPU_credit=0,archive_and_source_are_CPU_mocks=True,real_archive_qualification_is_separate=True)
    C.atomic(output,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
