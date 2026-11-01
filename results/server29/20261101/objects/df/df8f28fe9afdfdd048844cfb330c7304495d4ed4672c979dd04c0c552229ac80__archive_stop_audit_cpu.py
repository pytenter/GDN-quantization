#!/usr/bin/env python3
"""Independent repair-STOP archive audit; only CPU filesystem/SSH reads+catalog seal."""
import argparse,base64,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
EXPECTED_ARCHIVE_IO='99b526ddac7488edf6536b8ee6ddba4f39b079af5a968d3288f7ef56519d183a'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save_new(p,value=None,raw=None):
 if raw is None:raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def validate(root,card):
 expected=read(root/'independent_delivery/receipts/EXPECTED_FROZEN_IDENTITIES.json');row=next(x for x in expected['cards']if x['card_key']==card);croot=root/card
 for key,rel in [('source_manifest_sha256','SOURCE_MANIFEST.json'),('authorization_sha256','AUTHORIZATION.json'),('archive_config_sha256','ARCHIVE_CONFIG.json')]:require(sha(croot/rel)==row[key],'Frozen binding changed:'+card+':'+rel)
 manifest=read(croot/'SOURCE_MANIFEST.json');p=croot/'scripts/archive_io.py';source=next(x for x in manifest['execution_sources']if x['path']=='scripts/archive_io.py');require(sha(p)==source['sha256']==EXPECTED_ARCHIVE_IO and p.stat().st_size==source['bytes'],'Frozen archive module changed')
 config=read(croot/'ARCHIVE_CONFIG.json');auth=read(croot/'AUTHORIZATION.json');require(auth['archive_config_sha256']==row['archive_config_sha256']and config['card_key']==card and config['archive_root']==expected['archive_root'],'Archive config identity mismatch')
 return croot

def module(root,card):
 croot=validate(root,card);p=croot/'scripts/archive_io.py';spec=importlib.util.spec_from_file_location('_frozen_archive_io_audit',p);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);return A,croot

def ssh29(config,code,payload):
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',config['control_path'],config['host'],shlex.join([config.get('remote_python','python3'),'-B','-c',code])]
 p=subprocess.run(cmd,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE);require(p.returncode==0,'Archive CPU command failed:'+p.stderr.decode(errors='replace')[:3000]);return json.loads(p.stdout)
PRECHECK="""from pathlib import Path
import json,sys
v=json.load(sys.stdin);print(json.dumps([dict(card=k,path=str(Path(v['archive_root'])/k/'ASSET_CATALOG.json'),exists=(Path(v['archive_root'])/k/'ASSET_CATALOG.json').exists())for k in v['cards']]))
"""
FETCH="""from pathlib import Path
import json,sys,base64,hashlib
v=json.load(sys.stdin);p=Path(v['path']);a=p.stat();assert a.st_size<1024*1024;b=p.read_bytes();z=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns);print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())))
"""
INVENTORY=r'''from pathlib import Path
import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
v=json.load(sys.stdin);root=Path(v['archive_root']);expected={x['path']:x for x in v['expected']};paths=sorted(p for p in root.rglob('*')if p.is_file()or p.is_symlink())
def inspect(p):
 if p.is_symlink():raise RuntimeError('Unexpected archive symlink:'+str(p))
 a=p.stat();name=str(p)
 if name in expected:
  x=expected[name]
  if a.st_size!=x['bytes']:raise RuntimeError('Published bytes changed after fresh verification:'+name)
  role='PUBLISHED_FAILURE_RAW_EVIDENCE'if x['role']=='failure_raw_archive'else'PUBLISHED_REFERENCE_RAW_EVIDENCE'
  return dict(path=name,bytes=a.st_size,sha256=x['sha256'],classification=role,raw_receipt_role=x['role'],SHA_proof='FRESH_FROZEN_PER_CARD_REMOTE_STREAM_VERIFIER',scientific_qualification_from_storage_proof=False)
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.stat()
 if(a.st_ino,a.st_size,a.st_mtime_ns)!=(z.st_ino,z.st_size,z.st_mtime_ns):raise RuntimeError('Physical asset changed during hash:'+name)
 role='STORAGE_CATALOG_METADATA'if p.name=='ASSET_CATALOG.json'else'ENGINEERING_ONLY_PROBE'if'engineering_transport_probe'in p.parts else'UNCOMMITTED_TEMP_RAW'if'.tmp.'in p.name else'UNCOMMITTED_ORPHAN_RAW_NO_TOKEN_RECEIPT'if p.suffix=='.bin'else'UNREFERENCED_ARCHIVE_ASSET_NOT_SCIENTIFICALLY_QUALIFIED'
 return dict(path=name,bytes=z.st_size,sha256=h.hexdigest(),classification=role,SHA_proof='FRESH_INDEPENDENT_CPU_STREAM_SHA',scientific_qualification_from_storage_proof=False)
with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(inspect,paths))
missing=sorted(set(expected)-{x['path']for x in rows})
if missing:raise RuntimeError('Expected published raw missing:'+str(missing))
totals={}
for x in rows:
 t=totals.setdefault(x['classification'],dict(files=0,bytes=0));t['files']+=1;t['bytes']+=x['bytes']
print(json.dumps(dict(status='PASS_PHYSICAL_ARCHIVE_ACCOUNTING_STORAGE_ONLY_SCIENCE_INCOMPLETE',archive_root=str(root),unix=time.time(),files=len(rows),bytes=sum(x['bytes']for x in rows),classifications=totals,rows=rows,missing_published=[],no_deletion=True)))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--mode',choices=['precheck','card','inventory'],required=True);p.add_argument('--card',choices=['gpu0','gpu1','gpu2','gpu3']);a=p.parse_args();root=a.run_root;out=root/'independent_delivery/receipts';out.mkdir(parents=True,exist_ok=True)
 require(read(root/'SESSION_RESULT.json')['status']=='FAIL_CLOSED_STOP'and(root/'FIRST_FAILURE.json').exists(),'Real STOP evidence missing');card=a.card or'gpu0';A,croot=module(root,card);config=A.load_archive_config(croot)
 if a.mode=='precheck':
  exists=ssh29(config,PRECHECK,dict(archive_root=config['archive_root'],cards=['gpu0','gpu1','gpu2','gpu3']));require(all(not x['exists']for x in exists),'Catalog already exists: never republish')
  rosters={};summary=[]
  for k in ['gpu0','gpu1','gpu2','gpu3']:
   validate(root,k);rows=A.enumerate_archive_objects(root/k);rosters[k]=rows;totals={}
   for row in rows:
    v=totals.setdefault(row['role'],dict(files=0,bytes=0));v['files']+=1;v['bytes']+=row['bytes']
   summary.append(dict(card_key=k,objects=len(rows),roles=totals))
  result=dict(status='PASS_ALL_FOUR_CATALOGS_ABSENT_BEFORE_ONE_SHOT_PUBLICATION',rows=exists,expected_objects_by_card=rosters,expected_summary=summary,archive_module_sha256=EXPECTED_ARCHIVE_IO,unix=time.time());receipt=save_new(out/'ARCHIVE_CATALOG_PRECHECK.json',result)
 elif a.mode=='card':
  require(a.card is not None,'Card required');pre=read(out/'ARCHIVE_CATALOG_PRECHECK.json');expected=pre['expected_objects_by_card'][card];target=out/(card+'_ACCEPTED_RAW_VERIFICATION.json');require(not target.exists(),'Receipt exists; no duplicate final verify/publish')
  exists=ssh29(config,PRECHECK,dict(archive_root=config['archive_root'],cards=[card]));require(not exists[0]['exists'],'Catalog exists; no duplicate publication');require(A.enumerate_archive_objects(croot)==expected,'Expected raw roster changed after STOP precheck')
  started=time.time();result=A.final_verify_archive(croot,publish_catalog=True);require(result['objects']==len(expected),'Fresh raw count mismatch')
  catalog=result['cards'][0]['catalog'];mirror=ssh29(config,FETCH,dict(path=catalog['path']));require(mirror['bytes']==catalog['bytes']and mirror['sha256']==catalog['sha256'],'Independent catalog mirror changed');raw=base64.b64decode(mirror['base64']);require(hashlib.sha256(raw).hexdigest()==catalog['sha256'],'Catalog mirror decode mismatch');local_catalog=save_new(out/(card+'_ASSET_CATALOG.json'),raw=raw)
  result.update(scope='FROZEN_ARCHIVE_STORAGE_BYTES_ONLY_NOT_SCIENTIFIC_PASS',overall_scientific_status='FAIL_CLOSED_STOP_INCOMPLETE',card_key=card,expected_object_receipts=expected,catalog_mirror_receipt=local_catalog,started_unix=started,finished_unix=time.time(),archive_module_sha256=EXPECTED_ARCHIVE_IO,archive_config_sha256=config['config_sha256'],no_GPU_calls=True,no_large_Mac_download=True);receipt=save_new(target,result)
 else:
  expected=[];proofs=[]
  for k in ['gpu0','gpu1','gpu2','gpu3']:
   rp=out/(k+'_ACCEPTED_RAW_VERIFICATION.json');r=read(rp);proofs.append(dict(path=str(rp),bytes=rp.stat().st_size,sha256=sha(rp)));roles={x['path']:x['role']for x in r['expected_object_receipts']}
   for c in r['cards']:
    expected.extend(dict(x,role=roles[x['path']])for x in c['objects'])
  result=ssh29(config,INVENTORY,dict(archive_root=config['archive_root'],expected=expected));result.update(source_proofs=proofs,published_expected_files=len(expected),overall_science_incomplete=True,no_GPU_calls=True,no_large_Mac_download=True);receipt=save_new(out/'PHYSICAL_ARCHIVE_INVENTORY_AND_UNCOMMITTED.json',result)
 print(json.dumps(dict(status='PASS_INDEPENDENT_ARCHIVE_AUDIT_RECEIPT',mode=a.mode,card=a.card,receipt=receipt,objects=result.get('objects',result.get('files')),expected_summary=result.get('expected_summary'),bytes=result.get('bytes'),classifications=result.get('classifications')),allow_nan=False),flush=True)
if __name__=='__main__':main()
