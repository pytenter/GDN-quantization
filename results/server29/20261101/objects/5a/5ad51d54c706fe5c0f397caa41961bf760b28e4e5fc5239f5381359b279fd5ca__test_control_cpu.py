"""Real CPU Popen lifecycle/queue fixtures, no CUDA or model/science credit."""
import argparse,contextlib,copy,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import execution_common as C
import formal_controller as M
from score_cpu import summarize

WORKER='''import argparse,ctypes,hashlib,json,os,pathlib,sys,threading,time
p=argparse.ArgumentParser();p.add_argument('--unit');p.add_argument('--source-sha');p.add_argument('--permit-sha');a=p.parse_args()
r=pathlib.Path(__file__).resolve().parents[1];job=next(x for x in json.loads((r/'FORMAL_JOBS.json').read_text())['jobs'] if x['id']==a.unit)
time.sleep(.06)
if MODE=='exit7' and job['id']=='cpu01_8':sys.exit(7)
result=dict(status='PASS',job=job,source_manifest_sha256=a.source_sha,result=dict(seed=1),CPU_FIXTURE_ONLY=True)
result['payload_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
(r/'formal'/a.unit/'RESULT.json').write_text(json.dumps(result))
print('CPU fixture finished',flush=True)
if MODE=='leaderZ':
 threading.Thread(target=lambda:time.sleep(.18)).start()
 ctypes.CDLL(None).syscall(60,0)
'''

def jobs():
    return [dict(id=f'cpu{i:02d}_{p}',problem_id=f'cpu{i:02d}',period=p,seed=1,max_new_tokens=81920,gpu=(i-1)%4,gpu_uuid=C.UUIDS[(i-1)%4],input_ids_sha256='CPU_FIXED_INPUT_'+str(i)) for i in range(1,31) for p in (8,16)]

def fixture(mode):
    with tempfile.TemporaryDirectory(prefix='pace_unroll_control_',dir='/dev/shm') as td:
        r=Path(td);(r/'scripts').mkdir();(r/'formal').mkdir();(r/'validation').mkdir();(r/'runtime').mkdir()
        jj=jobs();(r/'FORMAL_JOBS.json').write_text(json.dumps(dict(jobs=jj)))
        (r/'FORMAL_SOURCE_MANIFEST.json').write_text(json.dumps(dict(files=[])))
        (r/'GPU_PERMIT.json').write_text(json.dumps(dict(CPU_FIXTURE_ONLY=True)))
        (r/'validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json').write_text(json.dumps(dict(formal_admission=True,new_formal_target=60,CPU_FIXTURE_ONLY=True)))
        (r/'scripts/formal_worker.py').write_text(WORKER.replace('MODE',repr(mode)))
        original=dict(ROOT=M.ROOT,source=C.source,model=C.model_pins,apps=C.apps,publish=M.publish,sleep=M.time.sleep)
        publications=[]
        def fake_publish(root,paths,relative):
            assert all((Path(root)/x).is_file() for x in paths)
            publications.append(relative)
            return dict(status='PASS_29_DURABLE_NOREPLACE',CPU_MOCK_ONLY=True,files=len(paths))
        try:
            M.ROOT=r;C.source=lambda root,expected:None;C.model_pins=lambda root,full=False:[];C.apps=lambda:[];M.publish=fake_publish
            original_sleep=original['sleep'];M.time.sleep=lambda seconds:original_sleep(min(seconds,.01))
            with (r/'controller_stdout.log').open('w') as log, contextlib.redirect_stdout(log):
                rc=M.run(argparse.Namespace(source_sha='CPU_FIXTURE_SOURCE',permit_sha=C.sha(r/'GPU_PERMIT.json'),resume_verified=False))
        finally:
            M.ROOT=original['ROOT'];C.source=original['source'];C.model_pins=original['model'];C.apps=original['apps'];M.publish=original['publish'];M.time.sleep=original['sleep']
        events=[json.loads(x) for x in (r/'runtime/CONTROLLER_EVENTS.jsonl').read_text().splitlines()]
        closed=list((r/'formal').glob('*/OWNING_CLOSED.json'))
        try:
            if mode=='exit7':
                assert rc==1 and (r/'runtime/FIRST_FAILURE.json').is_file() and not (r/'runtime/FORMAL_SESSION_RESULT.json').exists()
                assert any(x.get('true_exit')==7 for x in events if x['phase']=='OWNED_STOP_WAIT')
            else:
                assert rc==0 and len(closed)==60 and len(publications)==61
                assert all(json.loads(x.read_text())['owning_Popen_exit']==0 for x in closed)
                if mode=='leaderZ':assert any(x['phase'].startswith('EXIT_PENDING') for x in events),'Fixture did not observe actual leaderZ'
        except BaseException:
            target=ROOT/'validation/cpu_failure_scopes'/(mode+'_'+str(time.time_ns()))
            shutil.copytree(r,target)
            print(json.dumps(dict(status='CPU_FIXTURE_FAILURE_RETAINED',case=mode,evidence=str(target),failure=json.loads((r/'runtime/FIRST_FAILURE.json').read_text()) if (r/'runtime/FIRST_FAILURE.json').exists() else None)),flush=True)
            raise
        return dict(case=mode,actual_returncode=rc,actual_closed=len(closed),archive_calls_mocked=len(publications),actual_EXIT_PENDING_events=sum(x['phase'].startswith('EXIT_PENDING') for x in events),GPU_calls=0,model_loads=0)

def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    rows=[fixture(x) for x in ['normal','leaderZ','exit7']]
    roster=jobs();negative=[]
    for name,edit in [('duplicate',lambda x:x.__setitem__(1,x[0])),('bad_seed',lambda x:x[0].__setitem__('seed',2)),('bad_period',lambda x:x[0].__setitem__('period',32)),('wrong_UUID',lambda x:x[0].__setitem__('gpu_uuid','FOREIGN')),('short_cap',lambda x:x[0].__setitem__('max_new_tokens',1))]:
        bad=copy.deepcopy(roster);edit(bad);rejected=False
        try:M.validate_jobs(bad)
        except AssertionError:rejected=True
        assert rejected;negative.append(dict(case=name,rejected=True))
    # Paired exact McNemar and arm membership; no actual answers involved.
    fake=[dict(problem_id=f'cpu{i:02d}',period=p,correct=(i%3==0 if p==8 else i%5==0),generated_token_count=100+i,finish_reason='eos',generation_wall_s=1+i,repeated_32gram_ratio=0) for i in range(30) for p in (8,16)]
    summary=summarize(fake);assert summary['arms']['8']['correct']==10 and summary['arms']['16']['correct']==6 and sum(summary['paired_counts'].values())==30
    result=dict(status='PASS_REAL_CPU_CONTROL_AND_PAIRED_STATISTICS_FIXTURES',rows=rows,negative_jobs=negative,statistical_fixture_pass=True,actual_GPU_credit=0,archive_is_mocked_not_real_archive_qualification=True)
    C.atomic(output,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)
