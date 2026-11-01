#!/usr/bin/env python3
"""Independent post-STOP CPU storage audit; no torch/CUDA/NVML/process queries."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.dont_write_bytecode=True
EXPECTED_ARCHIVE_IO='99b526ddac7488edf6536b8ee6ddba4f39b079af5a968d3288f7ef56519d183a'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save_new(p,v):
 raw=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def module(root,card):
 croot=root/card;manifest=read(croot/'SOURCE_MANIFEST.json');p=croot/'scripts/archive_io.py';row=next(x for x in manifest['execution_sources'] if x['path']=='scripts/archive_io.py')
 require(sha(p)==row['sha256']==EXPECTED_ARCHIVE_IO and p.stat().st_size==row['bytes'],'Frozen archive source identity mismatch')
 auth=read(croot/'AUTHORIZATION.json');require(sha(croot/'ARCHIVE_CONFIG.json')==auth['archive_config_sha256'],'Frozen archive config changed')
 sys.path.insert(0,str(croot/'scripts'));import archive_io as A
 require(Path(A.__file__).resolve()==p.resolve(),'Wrong archive module imported');return A,croot

def ssh29(config,code,payload):
 command=shlex.join([config.get('remote_python','python3'),'-B','-c',code]);cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',config['control_path'],config['host'],command]
 p=subprocess.run(cmd,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 require(p.returncode==0,'Archive CPU audit command failed:'+p.stderr.decode(errors='replace')[:3000]);return json.loads(p.stdout)
PRECHECK=r'''from pathlib import Path
import json,sys
v=json.load(sys.stdin);print(json.dumps([dict(card=k,path=str(Path(v['archive_root'])/k/'ASSET_CATALOG.json'),exists=(Path(v['archive_root'])/k/'ASSET_CATALOG.json').exists()) for k in v['cards']]))'''
INVENTORY=r'''from pathlib import Path
import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
v=json.load(sys.stdin);root=Path(v['archive_root']);expected={x['path']:x for x in v['expected']};paths=sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
def inspect(p):
 if p.is_symlink():raise RuntimeError('Unexpected archive symlink:'+str(p))
 a=p.stat();name=str(p)
 if name in expected:
  x=expected[name]
  if a.st_size!=x['bytes']:raise RuntimeError('Published bytes changed after fresh verification:'+name)
  return dict(path=name,bytes=a.st_size,sha256=x['sha256'],classification='PUBLISHED_REFERENCE_RAW_BYTES_VERIFIED_SCIENCE_INCOMPLETE',SHA_proof='FRESH_FROZEN_PER_CARD_REMOTE_STREAM_VERIFIER',scientific_qualification=False)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.stat()
 if (a.st_ino,a.st_size,a.st_mtime_ns)!=(z.st_ino,z.st_size,z.st_mtime_ns):raise RuntimeError('Uncommitted/engineering asset changed during hash:'+name)
 role='STORAGE_CATALOG_METADATA' if p.name=='ASSET_CATALOG.json' else 'ENGINEERING_ONLY_PROBE' if 'engineering_transport_probe' in p.parts else 'UNCOMMITTED_TEMP_RAW' if '.tmp.' in p.name else 'UNCOMMITTED_ORPHAN_RAW_NO_TOKEN_RECEIPT' if p.suffix=='.bin' else 'UNREFERENCED_ARCHIVE_ASSET_NOT_SCIENTIFICALLY_QUALIFIED'
 return dict(path=name,bytes=z.st_size,sha256=h.hexdigest(),classification=role,SHA_proof='FRESH_INDEPENDENT_CPU_STREAM_SHA',scientific_qualification=False)
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect,paths))
missing=sorted(set(expected)-{x['path'] for x in rows})
if missing:raise RuntimeError('Expected published raw missing:'+str(missing))
totals={}
for x in rows:
 t=totals.setdefault(x['classification'],dict(files=0,bytes=0));t['files']+=1;t['bytes']+=x['bytes']
print(json.dumps(dict(status='PASS_PHYSICAL_ARCHIVE_ACCOUNTING_STORAGE_ONLY_SCIENCE_INCOMPLETE',archive_root=str(root),unix=time.time(),files=len(rows),bytes=sum(x['bytes'] for x in rows),classifications=totals,rows=rows,missing_published=[],no_deletion=True)))'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--mode',choices=['precheck','card','inventory'],required=True);p.add_argument('--card',choices=['gpu0','gpu1','gpu2','gpu3']);a=p.parse_args();root=a.run_root;out=root/'independent_delivery';out.mkdir(exist_ok=True)
 require(read(root/'SESSION_RESULT.json')['status']=='FAIL_CLOSED_STOP' and (root/'FIRST_FAILURE.json').exists(),'Expected real STOP not present')
 card=a.card or 'gpu0';A,croot=module(root,card);config=A.load_archive_config(croot)
 if a.mode=='precheck':
  rows=ssh29(config,PRECHECK,dict(archive_root=config['archive_root'],cards=['gpu0','gpu1','gpu2','gpu3']));require(all(not x['exists'] for x in rows),'Catalog already exists: do not publish again')
  result=dict(status='PASS_ALL_FOUR_CATALOGS_ABSENT_BEFORE_ONE_SHOT_PUBLICATION',rows=rows,archive_module_sha256=EXPECTED_ARCHIVE_IO,unix=time.time());receipt=save_new(out/'ARCHIVE_CATALOG_PRECHECK.json',result)
 elif a.mode=='card':
  require(a.card is not None,'--card required');require((out/'ARCHIVE_CATALOG_PRECHECK.json').exists(),'Precheck missing');target=out/(card+'_ACCEPTED_RAW_VERIFICATION.json');require(not target.exists(),'Independent card receipt already exists; no republish')
  exists=ssh29(config,PRECHECK,dict(archive_root=config['archive_root'],cards=[card]));require(not exists[0]['exists'],'Card catalog exists; no duplicate publication')
  start=time.time();result=A.final_verify_archive(croot,publish_catalog=True);result.update(scope='FROZEN_ARCHIVE_STORAGE_BYTES_ONLY_NOT_SCIENTIFIC_PASS',real_scientific_status='FAIL_CLOSED_STOP_INCOMPLETE',card_key=card,started_unix=start,finished_unix=time.time(),archive_module_sha256=EXPECTED_ARCHIVE_IO,archive_config_sha256=config['config_sha256'],no_GPU_calls=True,no_large_Mac_download=True)
  receipt=save_new(target,result)
 else:
  expected=[];proofs=[]
  for k in ['gpu0','gpu1','gpu2','gpu3']:
   rp=out/(k+'_ACCEPTED_RAW_VERIFICATION.json');r=read(rp);proofs.append(dict(path=str(rp),bytes=rp.stat().st_size,sha256=sha(rp)))
   for x in r['cards']:expected.extend(x['objects'])
  result=ssh29(config,INVENTORY,dict(archive_root=config['archive_root'],expected=expected));result.update(source_proofs=proofs,published_expected_files=len(expected),all_real_GPU_units_scientifically_incomplete=True,no_GPU_calls=True,no_large_Mac_download=True);receipt=save_new(out/'PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json',result)
 print(json.dumps(dict(status='PASS_INDEPENDENT_ARCHIVE_AUDIT_RECEIPT',mode=a.mode,card=a.card,receipt=receipt,objects=result.get('objects',result.get('files')),bytes=result.get('bytes'),classifications=result.get('classifications')),allow_nan=False))
if __name__=='__main__':main()
