import argparse,ctypes,hashlib,json,os,pathlib,sys,threading,time
p=argparse.ArgumentParser();p.add_argument('--unit');p.add_argument('--source-sha');p.add_argument('--permit-sha');a=p.parse_args()
r=pathlib.Path(__file__).resolve().parents[1];job=next(x for x in json.loads((r/'FORMAL_JOBS.json').read_text())['jobs'] if x['id']==a.unit)
time.sleep(.06)
if 'normal'=='exit7' and job['id']=='cpu01_8':sys.exit(7)
result=dict(status='PASS',job=job,source_manifest_sha256=a.source_sha,result=dict(seed=1),CPU_FIXTURE_ONLY=True)
result['payload_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
(r/'formal'/a.unit/'RESULT.json').write_text(json.dumps(result))
print('CPU fixture finished',flush=True)
if 'normal'=='leaderZ':
 threading.Thread(target=lambda:time.sleep(.18)).start()
 ctypes.CDLL(None).syscall(60,0)
