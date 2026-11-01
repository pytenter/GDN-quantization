#!/usr/bin/env python3
"""Mac metadata-only coordinator for independent post-STOP CPU archive audit."""
import base64,hashlib,json,shlex,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'independent_delivery';root=json.loads((R/'ROOT.json').read_text());remote=root['remote_root'];py=root['remote_python'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',root['ssh_control_path'],root['remote_user']+'@'+root['remote_host']]
def run(code,args=(),payload=None):
 cmd=ssh+[shlex.join([py,'-B','-c',code]+list(args))];p=subprocess.run(cmd,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[:4000])
 return p.stdout
def main():
 src=O/'scripts/archive_stop_audit_cpu.py';body=src.read_bytes();rp=remote+'/independent_delivery/archive_stop_audit_cpu.py'
 stage="from pathlib import Path;import sys,hashlib,json;p=Path(sys.argv[1]);raw=sys.stdin.buffer.read();p.parent.mkdir(parents=True,exist_ok=True);assert (not p.exists()) or p.read_bytes()==raw;f=None if p.exists() else p.open('xb');f.write(raw) if f else None;f.close() if f else None;print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))"
 result=json.loads(run(stage,[rp],body));assert result['sha256']==hashlib.sha256(body).hexdigest();(O/'receipts/STAGED_INDEPENDENT_AUDIT_SCRIPT.json').write_text(json.dumps(result,indent=2)+'\n')
 def action(mode,card=None):
  args=['--run-root',remote,'--mode',mode]+(['--card',card] if card else []);p=subprocess.run(ssh+[shlex.join([py,'-B',rp]+args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if p.returncode:raise RuntimeError(mode+' '+str(card)+' '+p.stderr.decode(errors='replace')[:4000])
  v=json.loads(p.stdout);name=(mode+'_'+(card or 'global')+'.json');(O/'receipts'/name).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True);return v
 pre=action('precheck');results=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(action,'card','gpu'+str(i)):'gpu'+str(i) for i in range(4)}
  for f in as_completed(futures):results.append(f.result())
 inventory=action('inventory');all_results=[pre]+results+[inventory];downloads=[]
 for x in all_results:
  desc=x['receipt'];raw=run('from pathlib import Path;import sys;p=Path(sys.argv[1]);a=p.stat();raw=p.read_bytes();b=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns);sys.stdout.buffer.write(raw)',[desc['path']]);assert len(raw)==desc['bytes'] and hashlib.sha256(raw).hexdigest()==desc['sha256'];dest=O/Path(desc['path']).name
  with dest.open('xb') as f:f.write(raw)
  downloads.append(dict(local=str(dest),remote=desc['path'],bytes=len(raw),sha256=desc['sha256'],status='PASS_EXACT_SMALL_METADATA_DOWNLOAD'))
 v=json.loads((O/'PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json').read_text());summary=dict(status='PASS_INDEPENDENT_STORAGE_DELIVERY_SCIENCE_REMAINS_INCOMPLETE_STOP',files=v['files'],bytes=v['bytes'],classifications=v['classifications'],published_expected_files=v['published_expected_files'],downloaded_metadata=downloads,no_large_raw_downloaded=True,no_GPU_calls=True,remote_script_sha256=result['sha256'],finished_unix=time.time())
 (O/'ARCHIVE_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='downloaded_metadata'}),flush=True)
if __name__=='__main__':main()
