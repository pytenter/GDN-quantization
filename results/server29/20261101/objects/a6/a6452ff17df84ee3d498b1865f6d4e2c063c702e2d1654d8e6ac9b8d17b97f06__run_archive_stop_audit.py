#!/usr/bin/env python3
"""Metadata-only one-shot coordinator; four independent remote CPU raw hashes."""
import hashlib,json,shlex,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'independent_delivery';D=O/'receipts';root=json.loads((R/'ROOT.json').read_bytes());remote=root['remote_root'];py=root['remote_python'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',root['ssh_control_path'],root['remote_user']+'@'+root['remote_host']]
def run(code,args=(),payload=None):
 p=subprocess.run(ssh+[shlex.join([py,'-B','-c',code]+list(args))],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[:4000])
 return p.stdout

def main():
 release=json.loads((D/'ARCHIVE_RELEASE_OBSERVATION.json').read_bytes());assert not release['compute_apps'].strip()and not release['actual_process_identities']
 expected={'archive_root':root['archive_root'],'cards':root['cards'],'execution_run_id':root['execution_run_id']};raw=(json.dumps(expected,sort_keys=True,indent=2)+'\n').encode();p=D/'EXPECTED_FROZEN_IDENTITIES.json'
 with p.open('xb')as f:f.write(raw)
 stage="from pathlib import Path;import sys,hashlib,json;p=Path(sys.argv[1]);raw=sys.stdin.buffer.read();p.parent.mkdir(parents=True,exist_ok=True);assert(not p.exists())or p.read_bytes()==raw;f=None if p.exists()else p.open('xb');f.write(raw)if f else None;f.close()if f else None;print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))"
 desc=json.loads(run(stage,[remote+'/independent_delivery/receipts/EXPECTED_FROZEN_IDENTITIES.json'],raw));assert desc['sha256']==hashlib.sha256(raw).hexdigest()
 body=(O/'scripts/archive_stop_audit_cpu.py').read_bytes();rp=remote+'/independent_delivery/archive_stop_audit_cpu.py';staged=json.loads(run(stage,[rp],body));assert staged['sha256']==hashlib.sha256(body).hexdigest();(D/'STAGED_INDEPENDENT_AUDIT_SCRIPT.json').write_text(json.dumps(staged,indent=2)+'\n')
 def action(mode,card=None):
  args=['--run-root',remote,'--mode',mode]+(['--card',card]if card else[]);p=subprocess.run(ssh+[shlex.join([py,'-B',rp]+args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if p.returncode:raise RuntimeError(mode+' '+str(card)+' '+p.stderr.decode(errors='replace')[:4000])
  x=json.loads(p.stdout);(D/(mode+'_'+(card or'global')+'.json')).write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True);return x
 pre=action('precheck');results=[]
 with ThreadPoolExecutor(max_workers=4)as pool:
  futures=[pool.submit(action,'card','gpu'+str(i))for i in range(4)]
  for f in as_completed(futures):results.append(f.result())
 inventory=action('inventory');downloads=[]
 get="from pathlib import Path;import sys;p=Path(sys.argv[1]);a=p.stat();assert a.st_size<16*1024*1024;raw=p.read_bytes();b=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns);sys.stdout.buffer.write(raw)"
 descriptors=[x['receipt']for x in [pre]+results+[inventory]]
 for x in results:
  temp=json.loads(run(get,[x['receipt']['path']]));descriptors.append(temp['catalog_mirror_receipt'])
 for desc in descriptors:
  raw=run(get,[desc['path']]);assert len(raw)==desc['bytes']and hashlib.sha256(raw).hexdigest()==desc['sha256'];dest=D/Path(desc['path']).name
  with dest.open('xb')as f:f.write(raw)
  assert dest.stat().st_size==desc['bytes']and hashlib.sha256(dest.read_bytes()).hexdigest()==desc['sha256'];downloads.append(dict(local=str(dest),remote=desc['path'],bytes=len(raw),sha256=desc['sha256'],status='PASS_EXACT_SMALL_METADATA_DOWNLOAD'))
 v=json.loads((D/'PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json').read_bytes());summary=dict(status='PASS_INDEPENDENT_STORAGE_DELIVERY_SCIENCE_REMAINS_INCOMPLETE_STOP',files=v['files'],bytes=v['bytes'],classifications=v['classifications'],published_expected_files=v['published_expected_files'],downloaded_metadata=downloads,no_large_raw_downloaded=True,no_GPU_calls=True,remote_script_sha256=staged['sha256'],finished_unix=time.time())
 with(D/'ARCHIVE_AUDIT_SUMMARY.json').open('x')as f:json.dump(summary,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in summary.items()if k!='downloaded_metadata'}),flush=True)
if __name__=='__main__':main()
