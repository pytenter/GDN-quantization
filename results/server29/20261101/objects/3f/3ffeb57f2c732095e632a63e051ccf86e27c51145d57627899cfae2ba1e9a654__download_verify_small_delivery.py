#!/usr/bin/env python3
"""One-shot sealed small delivery55 ->Mac and55 ->29. Default preview.

Only --run with the root-observed sealed descriptor SHA starts transfer. No GPU,
model, raw tensor, deletion, scientific execution, retry or old-root write exists.
"""
import argparse
import base64
import ctypes
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import select
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
import unicodedata
import transfer_primitives as P

sys.dont_write_bytecode = True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
PRIMITIVES_SHA='0214931830433d9e9d7fab27f418d21b838e3464d1a4307b7375c5a535d67948'
PARENT_SHA='64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457'
MAX_FILE=64*1024**2
MAX_TOTAL=256*1024**2
MAX_BUNDLE=128*1024**2
MAX_FILES=20000
ANCHORS={'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'}
EXTERNAL='LOCAL_DELIVERY_VERIFICATION.json'
FORBIDDEN={'.pt','.pth','.bin','.safetensors','.npy','.npz','.tensor','.so','.dylib','.dll'}
SETTINGS={'indexed_files','indexed_bytes','manifest_sha256','SHA256SUMS_sha256','seal_sha256','payload_roster_sha256'}
REPLACED={'scripts/restoration_controller.py','worker/restoration_worker.py','qualification/GPU_PLAN.json','USER_REQUEST.txt'}


def require(ok, text):
    if not ok: raise RuntimeError(text)


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(data): return hashlib.sha256(data).hexdigest()


def relative(name):
    require(isinstance(name,str) and name and not any(c in name for c in '\r\n\x00\\'), 'Invalid relative path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==name and len(name)<4096,'Unsafe relative path: '+name)
    require(Path(name).suffix.lower() not in FORBIDDEN,'Model/raw/native object prohibited: '+name)
    return p


def path_at(root,name):
    p=relative(name); q=Path(root)
    for part in p.parts:
        q=q/part; require(not q.is_symlink(),'Symlink prohibited: '+str(q))
    require(Path(root).resolve() in q.resolve().parents,'Path escaped destination')
    return q


def raw(path,limit=MAX_FILE):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Not regular file: '+str(path))
    a=path.stat();require(a.st_size<=limit,'File exceeds bounded size: '+str(path))
    with path.open('rb') as stream: data=stream.read(limit+1)
    b=path.stat(); require(len(data)<=limit and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'File grew/changed: '+str(path))
    require(not data.startswith(b'\x7fELF'),'Native ELF prohibited: '+str(path))
    return data


def load(path): return json.loads(raw(path))
def projection_key(name): return unicodedata.normalize('NFC',name).casefold()


def verify_delivery(directory,settings,source_sha,with_receipt=False):
    """Verify logical bytes; explicitly distinguish Mac projection inode aliases."""
    d=Path(directory);require(SETTINGS<=set(settings),'Missing frozen delivery settings')
    m=load(path_at(d,'SHA256_MANIFEST.json'));require(m.get('task_id')==TASK,'Manifest task differs')
    rows=m['files'];expected={}
    for row in rows:
        relative(row['path']);require(row['path'] not in expected and row['path'] not in ANCHORS|{EXTERNAL},'Duplicate/recursive indexed path')
        require(type(row['bytes']) is int and 0<=row['bytes']<=MAX_FILE and len(row['sha256'])==64,'Invalid indexed size/hash')
        expected[row['path']]=row
    require(len(rows)==len(expected)==m['files_count']==settings['indexed_files'] and len(rows)<=MAX_FILES,'Manifest count differs')
    require(sum(x['bytes'] for x in rows)==m['total_bytes']==settings['indexed_bytes'] and m['total_bytes']<=MAX_TOTAL,'Manifest byte total differs')
    for name,key in [('SHA256_MANIFEST.json','manifest_sha256'),('SHA256SUMS.txt','SHA256SUMS_sha256'),('FINAL_SEAL_RECEIPT.json','seal_sha256')]:
        require(digest(raw(path_at(d,name)))==settings[key],'Anchor hash differs: '+name)
    names=set(expected)|ANCHORS; logical=[];alias_groups={}
    if with_receipt:names.add(EXTERNAL)
    for name in sorted(names):
        p=path_at(d,name);data=raw(p);row=dict(path=name,bytes=len(data),sha256=digest(data));logical.append(row)
        if name in expected:require(row==expected[name],'Indexed bytes/hash differs: '+name)
        alias_groups.setdefault(projection_key(name),[]).append(row)
    require(sum(x['bytes'] for x in logical)<=MAX_TOTAL,'Logical delivery total exceeds bound')
    for group in alias_groups.values():
        require(len({(x['bytes'],x['sha256']) for x in group})==1,'Differing-byte case/Unicode projection collision')
    physical=[];physical_name_aliases=[]
    for p in sorted(d.rglob('*')):
        require(not p.is_symlink(),'Actual delivery symlink')
        if p.is_file():
            name=str(p.relative_to(d));relative(name);require(projection_key(name) in alias_groups,'Unindexed physical file: '+name)
            if name not in names:
                require(sys.platform=='darwin','Non-Mac unindexed physical case variant: '+name)
                inode=(p.stat().st_dev,p.stat().st_ino)
                require(any((path_at(d,x['path']).stat().st_dev,path_at(d,x['path']).stat().st_ino)==inode for x in alias_groups[projection_key(name)]),'Unindexed distinct physical case variant: '+name)
                physical_name_aliases.append(dict(physical_name=name,listed_logical_names=[x['path'] for x in alias_groups[projection_key(name)]],same_inode_platform_projection=True))
            b=raw(p);require(any(x['bytes']==len(b) and x['sha256']==digest(b) for x in alias_groups[projection_key(name)]),'Physical projection differs')
            physical.append(dict(path=name,device=p.stat().st_dev,inode=p.stat().st_ino))
    require({projection_key(x['path']) for x in physical}==set(alias_groups),'Missing physical projection')
    pairs=[line.split('  ',1) for line in raw(path_at(d,'SHA256SUMS.txt')).decode().splitlines()]
    want={n:x['sha256'] for n,x in expected.items()};want['SHA256_MANIFEST.json']=settings['manifest_sha256']
    require(all(len(x)==2 for x in pairs) and len(pairs)==len(want) and len({n for h,n in pairs})==len(pairs) and {n:h for h,n in pairs}==want,'SUMS logical roster differs')
    base_rows=[x for x in logical if x['path']!=EXTERNAL]
    require(digest(canonical(base_rows))==settings['payload_roster_sha256'],'Logical payload roster hash differs')
    seal=load(path_at(d,'FINAL_SEAL_RECEIPT.json'));report=raw(path_at(d,'FINAL_SCIENTIFIC_REPORT.md'))
    require(seal.get('task_id')==TASK and seal.get('manifest_sha256')==settings['manifest_sha256'] and seal.get('SHA256SUMS_sha256')==settings['SHA256SUMS_sha256'],'Seal task/manifest/SUMS differs')
    require(seal.get('FINAL_SCIENTIFIC_REPORT_sha256')==digest(report),'Final Chinese report hash differs')
    freeze_path='payload/SOURCE_FREEZE.json';freeze=load(path_at(d,freeze_path));require(digest(raw(path_at(d,freeze_path)))==source_sha==seal.get('source_freeze_sha256'),'Actual407 freeze hash differs')
    source_descriptor=seal.get('source_freeze',{});require(source_descriptor.get('path')==freeze_path and source_descriptor.get('bytes')==expected[freeze_path]['bytes'] and source_descriptor.get('sha256')==source_sha and source_descriptor.get('files_verified')==407,'Seal source407 descriptor differs')
    frozen={r['path']:r for r in freeze['files']};require(len(frozen)==len(freeze['files'])==407,'Source407 unique roster differs')
    parent_path='payload/provenance/PARENT_EFFECTIVE_SOURCE_FREEZE.json';parent=load(path_at(d,parent_path));require(digest(raw(path_at(d,parent_path)))==PARENT_SHA,'Parent effective freeze differs')
    old={r['path']:r for r in parent['files']};require(set(old)==set(frozen) and len(old)==407,'Parent/new source path roster differs')
    require(all(frozen[n]==old[n] for n in set(frozen)-REPLACED) and all(frozen[n]!=old[n] for n in REPLACED),'Source403+4 lineage differs')
    for name,row in frozen.items():require(expected.get('payload/'+name)==dict(row,path='payload/'+name),'Indexed source407 binding differs: '+name)
    canonical_names=[n for n in frozen if n.startswith('candidate/dependency_by_path_sha256/')]
    require(len(canonical_names)==194 and len({(path_at(d,'payload/'+n).stat().st_dev,path_at(d,'payload/'+n).stat().st_ino) for n in canonical_names})==194,'Canonical194 source copies must retain independent physical files')
    anchor_map={'analysis/FINAL_INDEPENDENT_RESULT_REVIEW.json':'report_review_sha256','analysis/POST_GPU_RELEASE_ADMISSION.json':'release_proof_sha256','archive/ARCHIVE_VERIFICATION.json':'archive_verification_sha256'}
    for name,key in anchor_map.items():require(name in expected and seal.get(key)==expected[name]['sha256'],'Seal final audit anchor differs: '+name)
    review=load(path_at(d,'analysis/FINAL_INDEPENDENT_RESULT_REVIEW.json'));require(str(review.get('status','')).startswith('PASS') and review.get('findings')==[] and type(review.get('checks_total')) is int and review['checks_total']>0 and review.get('checks_passed')==review['checks_total'],'Final independent review not complete PASS')
    review_rows=review.get('receipts');require(isinstance(review_rows,list),'Final review receipt list missing');review_map={x['path']:x for x in review_rows};require(len(review_map)==len(review_rows),'Duplicate review receipt path')
    required_review={'FINAL_SCIENTIFIC_REPORT.md','RUNTIME_RESTORATION_DECISION.json','RESOURCE_LEDGER.json','analysis/FINAL_SOURCE_DEPENDENCY_VERIFICATION.json','analysis/POST_GPU_RELEASE_ADMISSION.json','archive/ARCHIVE_VERIFICATION.json'}
    require(required_review<=set(review_map),'Final review six required receipt bindings missing')
    for name,row in review_map.items():require(expected.get(name)==row,'Final review exact SHA/bytes differs: '+name)
    release=load(path_at(d,'analysis/POST_GPU_RELEASE_ADMISSION.json'));require(release.get('schema')=='QWEN_RECOVERY_RELEASE_ADMISSION_V1' and release.get('status')=='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED' and release.get('all_owned_processes_exited') is True and release.get('all_owned_GPU_resources_released') is True,'Actual release admission not PASS')
    aliases=[]
    for key,group in alias_groups.items():
        if len(group)>1:
            identities={(path_at(d,x['path']).stat().st_dev,path_at(d,x['path']).stat().st_ino) for x in group}
            aliases.append(dict(logical_paths=[x['path'] for x in group],identical_bytes=group[0]['bytes'],identical_sha256=group[0]['sha256'],physical_inodes=len(identities),projection_only_alias_reuse=len(identities)<len(group)))
    return dict(status='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES',settings=settings,destination=str(d),indexed_files=len(rows),indexed_bytes=m['total_bytes'],logical_payload_files=len(base_rows),logical_files_with_external_receipt=len(names),physical_files=len(physical),physical_unique_inodes=len({(x['device'],x['inode']) for x in physical}),case_projection_aliases=aliases,physical_name_projection_aliases=physical_name_aliases,payload_roster_sha256=settings['payload_roster_sha256'],source407_verified=407,inherited_sources403_verified=403,amended_sources4_verified=4,canonical_source194_independent_physical_files=194,source_freeze_sha256=source_sha,FINAL_SCIENTIFIC_REPORT_sha256=digest(report),logical_manifest_bytes_equality_only_not_physical_roster_equality=True,no_large_raw_model_or_native_object=True)


def extract_safe(bundle,destination,settings,source_sha):
    dest=Path(destination);require(not dest.exists(),'Extract destination exists: no retry/no overwrite');require(Path(bundle).stat().st_size<=MAX_BUNDLE,'Compressed bundle exceeds bound');dest.mkdir(parents=True)
    seen=set();projected={};total=0;aliases=[]
    with tarfile.open(bundle,'r:gz') as tf:
        for member in tf:
            relative(member.name);require(member.isfile() and not member.issym() and not member.islnk() and not member.sparse,'Only regular non-sparse tar members allowed')
            require(member.name not in seen and len(seen)<MAX_FILES+4 and 0<=member.size<=MAX_FILE,'Tar duplicate/count/size bound');seen.add(member.name);total+=member.size;require(total<=MAX_TOTAL,'Tar logical total bound')
            src=tf.extractfile(member);p=path_at(dest,member.name);p.parent.mkdir(parents=True,exist_ok=True)
            data=src.read(MAX_FILE+1);require(len(data)==member.size and not data.startswith(b'\x7fELF'),'Tar member bytes/native object differ');key=projection_key(member.name);identity=(len(data),digest(data))
            if key in projected:require(projected[key][1]==identity,'Differing-byte projection collision')
            if p.exists():
                require(key in projected and raw(p)==data,'Existing path is not a byte-identical in-this-bundle alias');aliases.append(dict(path=member.name,original=projected[key][0],bytes=identity[0],sha256=identity[1]))
            else:
                with p.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
                p.chmod(member.mode&0o777)
            projected.setdefault(key,(member.name,identity))
    proof=verify_delivery(dest,settings,source_sha);proof['extract_projection_reuse']=aliases;return proof


def publish_local_no_clobber(source,destination):
    source,destination=Path(source),Path(destination);require(not destination.exists(),'Local delivery already exists')
    libc=ctypes.CDLL(None,use_errno=True)
    if sys.platform=='darwin':
        fn=libc.renamex_np;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(os.fsencode(source),os.fsencode(destination),4)
    else:
        fn=getattr(libc,'renameat2',None);require(fn is not None,'Atomic no-clobber directory publish unavailable')
        fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(source),-100,os.fsencode(destination),1)
    if rc!=0:raise OSError(ctypes.get_errno(),'Atomic no-clobber local publication failed; incoming retained')


def remote_check_program():
    functions=[require,canonical,digest,relative,path_at,raw,load,projection_key,verify_delivery]
    constants=dict(TASK=TASK,MAX_FILE=MAX_FILE,MAX_TOTAL=MAX_TOTAL,MAX_FILES=MAX_FILES,PARENT_SHA=PARENT_SHA)
    code='import hashlib,json,os,sys,unicodedata\nfrom pathlib import Path,PurePosixPath\n'
    code+='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'
    code+='ANCHORS='+repr(ANCHORS)+'\nEXTERNAL='+repr(EXTERNAL)+'\nFORBIDDEN='+repr(FORBIDDEN)+'\nSETTINGS='+repr(SETTINGS)+'\nREPLACED='+repr(REPLACED)+'\n'
    code+='\n\n'.join(inspect.getsource(fn) for fn in functions)
    return code+'\ns=json.load(sys.stdin);print(json.dumps(verify_delivery(Path(s["destination"]),s["settings"],s["source_sha256"],s.get("with_receipt",False)),ensure_ascii=False,allow_nan=False))\n'


REMOTE_VERIFY64=P.REMOTE_VERIFY.replace('member.size<16*1024**2','member.size<=64*1024**2').replace('total<512*1024**2','total<=256*1024**2')
require(REMOTE_VERIFY64!=P.REMOTE_VERIFY and 'member.size<16*1024**2' not in REMOTE_VERIFY64,'Remote64MiB bound adaptation missing')
META_READ="import hashlib,json,sys;from pathlib import Path;s=json.loads(sys.argv[1]);p=Path(s['path']);assert p.is_file()and not p.is_symlink();a=p.stat();assert a.st_size<=1048576;b=p.read_bytes();z=p.stat();assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns);print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),document=json.loads(b))))"
BUNDLE_READ="import hashlib,json,sys;from pathlib import Path;s=json.loads(sys.argv[1]);p=Path(s['path']);a=p.stat();assert p.is_file()and not p.is_symlink()and a.st_size==s['bytes']<=134217728;h=hashlib.sha256();n=0\nwith p.open('rb')as f:\n for b in iter(lambda:f.read(1048576),b''):\n  n+=len(b);assert n<=s['bytes'];h.update(b);sys.stdout.buffer.write(b)\nz=p.stat();assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)and n==s['bytes']and h.hexdigest()==s['sha256'];sys.stdout.buffer.flush()"


class Transport(P.Transport):
    def call29(self,code,settings,data=b''):
        body=data if code==P.REMOTE_UPLOAD else P.encoded(settings);args=[json.dumps(settings)] if code==P.REMOTE_UPLOAD else []
        wrapper="import base64,json,shlex,subprocess,sys;v=json.load(sys.stdin);p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',v['control'],v['host'],shlex.join(['/usr/bin/python3','-B','-c',v['code']]+v['args'])],input=base64.b64decode(v['body']),capture_output=True,timeout=600);sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr);sys.exit(p.returncode)"
        return self.call55(wrapper,data=P.encoded(dict(control=self.root['archive_control_from55'],host=self.root['remote_user']+'@'+self.root['archive_host'],code=code,args=args,body=base64.b64encode(body).decode())))

    def forward_bundle29(self,path55,path29,description):
        require(description['bytes']<=MAX_BUNDLE,'Forward compressed bound');settings=dict(path=path29,bytes=description['bytes'],sha256=description['sha256'])
        wrapper="import json,pathlib,shlex,subprocess,sys;v=json.load(sys.stdin);p=pathlib.Path(v['source']);assert p.is_file()and not p.is_symlink()and p.stat().st_size==v['settings']['bytes']<=134217728\nwith p.open('rb')as stream:r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',v['control'],v['host'],shlex.join(['/usr/bin/python3','-B','-c',v['code'],json.dumps(v['settings'])])],stdin=stream,capture_output=True,timeout=600)\nsys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)"
        return self.call55(wrapper,data=P.encoded(dict(source=path55,settings=settings,control=self.root['archive_control_from55'],host=self.root['remote_user']+'@'+self.root['archive_host'],code=P.REMOTE_UPLOAD)))

    def download55(self,description,destination):
        require(description['bytes']<=MAX_BUNDLE,'Bundle bound');target=Path(destination);tmp=target.with_name(target.name+'.incoming');require(not target.exists() and not tmp.exists(),'Download path already exists')
        command=self.ssh+[shlex.join([self.py,'-B','-c',BUNDLE_READ,json.dumps(description)])]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE);h=hashlib.sha256();count=0;deadline=time.monotonic()+600
        try:
            with tmp.open('xb') as stream:
                while True:
                    left=deadline-time.monotonic();require(left>0,'Bounded small-download I/O deadline exceeded')
                    ready,_,_=select.select([process.stdout],[],[],min(left,10))
                    if not ready:continue
                    block=os.read(process.stdout.fileno(),1048576)
                    if not block:break
                    count+=len(block);require(count<=description['bytes'] and count<=MAX_BUNDLE,'Download exceeded exact compressed bound');h.update(block);stream.write(block)
                stream.flush();os.fsync(stream.fileno())
            rc=process.wait(timeout=max(1,deadline-time.monotonic()));error=process.stderr.read(8192).decode(errors='replace');require(rc==0,'Bounded bundle download failed: '+error)
            require(count==description['bytes'] and h.hexdigest()==description['sha256'],'Compressed bundle exact SHA/bytes mismatch');os.link(tmp,target);tmp.unlink()
            return dict(status='PASS_DOWNLOADED_FIXED_SMALL_BUNDLE_SHA_BYTES',path=str(target),bytes=count,sha256=h.hexdigest())
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise


def descriptor_admission(state,bundle_meta,local_meta,pinned_sha):
    require(bundle_meta['sha256']==pinned_sha,'Root-observed sealed descriptor SHA differs')
    b=bundle_meta['document'];v=local_meta['document'];g=state['remote_root']
    require(b.get('task_id')==TASK and b.get('status')=='PASS_POST_RELEASE_SMALL_DELIVERY_SEALED_BUNDLE' and b.get('destination')==g+'/final_delivery','Actual sealed bundle status/root differs')
    desc=b['bundle'];require(desc['path']==g+'/SMALL_DELIVERY.tar.gz' and type(desc['bytes']) is int and 0<desc['bytes']<=MAX_BUNDLE and len(desc['sha256'])==64,'Fixed bundle path/size differs')
    settings=b['settings'];require(SETTINGS<=set(settings) and settings==v.get('settings') and v.get('status')=='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES','Sealer descriptor/actual55 settings differ')
    require(type(settings['indexed_files']) is int and 0<settings['indexed_files']<=MAX_FILES and type(settings['indexed_bytes']) is int and 0<settings['indexed_bytes']<=MAX_TOTAL,'Bounded indexed payload count/bytes differs')
    return b


def run(root,pinned_sha,transport_factory=Transport):
    root=Path(root).resolve();state=load(root/'ROOT.json');require(state.get('task_id')==TASK and state.get('local_root')==str(root),'Local run ROOT identity differs')
    require(state.get('remote_host')=='202.38.247.55' and state.get('archive_host')=='202.38.247.29' and state.get('remote_user')=='zypan','Authorized three-host identity differs')
    require(state.get('remote_python')=='/data/ydai/miniconda3/envs/bitdecode/bin/python','Exact55 Python identity differs')
    require(digest(raw(Path(P.__file__)))==PRIMITIVES_SHA,'Copied immutable transfer primitive differs')
    operations=root/'operations';operations.mkdir(exist_ok=True);log=operations/'SMALL_DELIVERY_DOWNLOAD_STATE.jsonl';require(not log.exists(),'One-shot download state exists: inspect failure; do not retry')
    require(not (root/'delivery').exists() and not (root/'delivery.incoming').exists(),'Existing local delivery/incoming must not be overwritten')
    def record(event,**data):
        with log.open('a') as stream:stream.write(json.dumps(dict(event=event,unix=time.time(),**data),sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    def save(name,value):return P.new_bytes(operations/name,P.encoded(value))
    record('START_ONCE_ROOT_EXPLICIT_SEALED_DESCRIPTOR',sealed_descriptor_sha256=pinned_sha,GPU_queries=0,GPU_dispatches=0,primitive_sha256=PRIMITIVES_SHA,old_remote_verify_sha256=digest(P.REMOTE_VERIFY.encode()),new_remote_verify64_sha256=digest(REMOTE_VERIFY64.encode()),limits=dict(compressed=MAX_BUNDLE,file=MAX_FILE,total=MAX_TOTAL,files=MAX_FILES))
    try:
        transport=transport_factory(state,record);g=state['remote_root'];a=state['archive_root'];source_sha=state['source_freeze_sha256']
        meta=transport.call55(META_READ,[json.dumps(dict(path=g+'/operations/SMALL_DELIVERY_BUNDLE.json'))]);local_meta=transport.call55(META_READ,[json.dumps(dict(path=g+'/operations/SMALL_DELIVERY_LOCAL_VERIFICATION.json'))]);bundle=descriptor_admission(state,meta,local_meta,pinned_sha)
        save('DOWNLOADED_SEALER_DESCRIPTOR_EVIDENCE.json',dict(bundle=meta,server55_verification=local_meta));settings=bundle['settings'];description=bundle['bundle'];check=remote_check_program()
        proof55=transport.call55(check,data=P.encoded(dict(destination=g+'/final_delivery',settings=settings,source_sha256=source_sha)));save('DELIVERY_SERVER55_INDEPENDENT_VERIFICATION.json',proof55);record('SERVER55_LOGICAL_PAYLOAD407_VERIFIED')
        download=transport.download55(description,operations/'SMALL_DELIVERY.tar.gz');save('DELIVERY_BUNDLE_DOWNLOAD_VERIFICATION.json',download)
        extracted=extract_safe(operations/'SMALL_DELIVERY.tar.gz',root/'delivery.incoming',settings,source_sha);save('DELIVERY_LOCAL_EXTRACT_VERIFICATION.json',extracted);publish_local_no_clobber(root/'delivery.incoming',root/'delivery');proof_local=verify_delivery(root/'delivery',settings,source_sha);record('MAC_LOGICAL_PAYLOAD407_VERIFIED',logical_files=proof_local['logical_payload_files'],physical_files=proof_local['physical_files'],case_projection_aliases=proof_local['case_projection_aliases'])
        archive_bundle=a+'/SMALL_DELIVERY_SOURCE_AND_REPORTS.tar.gz';archive_dest=a+'/SMALL_DELIVERY';up=transport.forward_bundle29(description['path'],archive_bundle,description);save('DELIVERY_ARCHIVE29_BUNDLE_UPLOAD.json',up)
        publish=transport.call29(REMOTE_VERIFY64,dict(settings,bundle=archive_bundle,destination=archive_dest,bundle_bytes=description['bytes'],bundle_sha256=description['sha256']));save('DELIVERY_ARCHIVE29_ATOMIC_PUBLICATION.json',publish)
        proof29=transport.call29(check,dict(destination=archive_dest,settings=settings,source_sha256=source_sha));save('DELIVERY_ARCHIVE29_INDEPENDENT_VERIFICATION.json',proof29);record('ARCHIVE29_LOGICAL_PAYLOAD407_VERIFIED')
        again=verify_delivery(root/'delivery',settings,source_sha);proof55_again=transport.call55(check,data=P.encoded(dict(destination=g+'/final_delivery',settings=settings,source_sha256=source_sha)));save('DELIVERY_SERVER55_FINAL_BEFORE_EXTERNAL_RECEIPT.json',proof55_again)
        require(all(x['payload_roster_sha256']==settings['payload_roster_sha256'] and x['source407_verified']==407 for x in [proof55,proof55_again,proof29,again]),'Three-host logical equality differs')
        external=dict(task_id=TASK,status='PASS_VERIFIED_POST_RELEASE_SCIENTIFIC_SMALL_DELIVERY_THREE_HOSTS',scientific_status=bundle['scientific_status'],verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,indexed_files=settings['indexed_files'],indexed_bytes=settings['indexed_bytes'],source_freeze_sha256=source_sha,FINAL_SCIENTIFIC_REPORT_sha256=again['FINAL_SCIENTIFIC_REPORT_sha256'],all_three_hosts_logical_manifest_SHA_bytes_equal=True,physical_roster_identical_not_claimed=True,Mac_physical_files=again['physical_files'],Mac_projection_aliases=again['case_projection_aliases'],server55_verification=proof55_again,Mac_verification=again,archive29_verification=proof29,bundle=description,external_receipt_excluded_from_indexed_manifest=True,replication_proof_separate='operations/EXTERNAL_RECEIPT_REPLICATION.json',GPU_queries=0,GPU_dispatches=0,no_model_or_raw_tensor_download=True,no_scientific_program_or_old_seal_mutation=True)
        ext=P.new_bytes(root/'delivery'/EXTERNAL,P.encoded(external));record('EXTERNAL_RECEIPT_CREATED_AFTER_THREE_HOST_VERIFICATION',receipt=ext);body=raw(root/'delivery'/EXTERNAL)
        transport.call55(P.REMOTE_UPLOAD,[json.dumps(dict(path=g+'/final_delivery/'+EXTERNAL,bytes=ext['bytes'],sha256=ext['sha256']))],body);transport.call29(P.REMOTE_UPLOAD,dict(path=archive_dest+'/'+EXTERNAL,bytes=ext['bytes'],sha256=ext['sha256']),body)
        r55=transport.call55(P.REMOTE_RECEIPT_CHECK,data=P.encoded(dict(destination=g+'/final_delivery',bytes=ext['bytes'],sha256=ext['sha256'])));r29=transport.call29(P.REMOTE_RECEIPT_CHECK,dict(destination=archive_dest,bytes=ext['bytes'],sha256=ext['sha256']));final=verify_delivery(root/'delivery',settings,source_sha,True);require(digest(raw(root/'delivery'/EXTERNAL))==ext['sha256'],'Local external receipt changed')
        replication=dict(status='PASS_EXTERNAL_RECEIPT_REPLICATED_EXACT_ALL_THREE_HOSTS',local=ext,server55=r55,archive29=r29,logical_final_files=final['logical_files_with_external_receipt'],Mac_physical_final_files=final['physical_files'],no_hash_cycle=True,logical_manifest_bytes_equality_only_not_physical_roster_equality=True);proof=save('EXTERNAL_RECEIPT_REPLICATION.json',replication);record('COMPLETE',external_receipt=ext,external_replication_proof=proof)
        return dict(status=external['status'],LOCAL_DELIVERY_VERIFICATION=ext,replication_proof=proof,GPU_queries=0,GPU_dispatches=0)
    except BaseException as exc:
        record('FAILED_RETAIN_EVIDENCE_NO_AUTOMATIC_RETRY',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--run',action='store_true');p.add_argument('--sealed-descriptor-sha256');args=p.parse_args()
    if not args.run:
        print(json.dumps(dict(status='PREVIEW_ONLY_NO_NETWORK_NO_WRITES_NO_GPU',root=str(args.root.resolve()),maximum_compressed_bytes=MAX_BUNDLE,maximum_file_bytes=MAX_FILE,maximum_total_bytes=MAX_TOTAL,requires_root_observed_sealed_descriptor_SHA=True)));return
    pin=args.sealed_descriptor_sha256;require(isinstance(pin,str) and len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'--run requires exact root-observed sealed descriptor SHA256')
    print(json.dumps(run(args.root,pin),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
