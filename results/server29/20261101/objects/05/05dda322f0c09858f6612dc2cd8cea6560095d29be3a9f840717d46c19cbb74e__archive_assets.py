"""Lossless new-document archive; no legacy asset deletion or model path changes."""
import csv, hashlib, json, os, shlex, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TARGET='zypan@202.38.247.29'
REMOTE_BASE='/data/zypan/qwen_functional_replication_archive/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1'
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','22',TARGET]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(n,x):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def remote_py(source):
    p=subprocess.run(SSH+['python3 -'],input=source,text=True,capture_output=True)
    if p.returncode:raise RuntimeError('ARCHIVE_REMOTE_FAILURE '+p.stderr)
    return json.loads(p.stdout)

def preflight():
    p=ROOT/'engineering/ARCHIVE_IO_PROBE.bin'
    with p.open('xb') as f:f.write(bytes(range(256))*4096);f.flush();os.fsync(f.fileno())
    expected=sha(p)
    remote_py('import os,json\np='+repr(REMOTE_BASE)+"\nos.makedirs(p,exist_ok=True)\nv=os.statvfs(p)\nprint(json.dumps({'free_bytes':v.f_bavail*v.f_frsize}))")
    destination=TARGET+':'+REMOTE_BASE+'/ARCHIVE_IO_PROBE.bin'
    subprocess.run(['rsync','-a','--ignore-existing','-e',shlex.join(SSH[:-1]),str(p),destination],check=True)
    result=remote_py("import pathlib,hashlib,json\np=pathlib.Path("+repr(REMOTE_BASE+'/ARCHIVE_IO_PROBE.bin')+")\nprint(json.dumps({'SHA':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}))")
    assert result['SHA']==expected and result['bytes']==p.stat().st_size
    save('engineering/ARCHIVE_IO_PREFLIGHT.json',{'status':'PASS','local_path':str(p),'remote_path':REMOTE_BASE+'/ARCHIVE_IO_PROBE.bin','source_SHA':expected,'independent_remote_verification':result,'password_stored':False})

def audit_and_archive(doc):
    # Full CPU FP64 audit of all 384 retained positions in both arms, before relocation.
    import torch
    from replication_worker import P, W
    P.torch=torch;torch.set_num_threads(1)
    ref=load(ROOT/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json')
    anchor=load(ROOT/f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json')
    branches=[load(ROOT/f'students/doc_{doc:02d}/{arm}/BRANCH_MANIFEST.json') for arm in W.ARMS]
    spec=load(ROOT/'FROZEN_SCIENCE_SPEC.json')
    assert ref['status']==anchor['status']=='COMPLETE' and ref['consumed']==32768 and anchor['consumed']==8192
    assert len(ref['chunks'])==96
    arts=list(ref['chunks'])+[ref['logits'],ref['endpoint_states'],anchor['artifact']]
    for b in branches:
        assert b['status']=='COMPLETE' and b['consumed']==32768 and b['sample_count']==384
        assert b['continuation_QDQ_calls']==24576 and b['complete_source_anchor_immutable'] and b['every_token_QDQ'] and b['persistent_INT8_writeback']
        assert b['identity']['FP_reference_identity']==ref['reference_identity'] and b['identity']['anchor_checkpoint_identity']==anchor['anchor_identity']
        for k,v in [('model_SHA',spec['model_SHA']),('C6R_rotation_SHA',spec['C6R_rotation_SHA']),('quantizer_SHA',spec['quantizer_SHA']),('runtime_identity',spec['runtime_identity']),('GPU_UUID',spec['GPU_UUID']),('science_spec_SHA',sha(ROOT/'FROZEN_SCIENCE_SPEC.json')),('code_commit',spec['code_commit'])]:assert b['identity'][k]==v,(k,doc)
        d=spec['documents'][doc-1]
        for k,v in [('document_id',d['document_id']),('document_raw_SHA',d['raw_text_SHA']),('document_token_SHA',d['token_SHA']),('intervention_policy_identity',W.P.digest(spec['intervention'])),('position_schedule_identity',W.P.digest(W.POSITIONS))]:assert b['identity'][k]==v,(k,doc)
        arts += b['artifacts']+[b['sampled_logits']]
    for a in arts:assert sha(a['path'])==a['sha256'],('ARTIFACT_SHA_FAILURE',a['path'])
    fp=torch.load(ref['logits']['path'],map_location='cpu',weights_only=True)['logits'];audits=[]
    for arm,b in zip(W.ARMS,branches):
        student=torch.load(b['sampled_logits']['path'],map_location='cpu',weights_only=True)['logits']
        with (ROOT/f'students/doc_{doc:02d}/{arm}/SAMPLED_FULLVOCAB_FUTURE_KL.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==384 and [int(x['token_index']) for x in rows]==W.POSITIONS
        assert all(int(x['full_vocabulary_size'])==248320 and int(x['consumed_tokens'])==int(x['token_index'])+1 and int(x['predicted_token_index'])==int(x['token_index'])+1 for x in rows)
        assert math_fsum([float(x['KL_FP32']) for x in rows])/384==b['K']
        errors=[abs(float(rows[i]['KL_FP32'])-P.kl64(fp[i],student[i])) for i in range(384)]
        bound=load(ROOT/'KL_REDUCER_QUALIFICATION.json')['FP32_max_abs_error']
        audits.append({'condition':arm,'position_count':384,'max_abs_KL32_vs_FP64_error':max(errors),'frozen_bound':bound})
        assert max(errors)<=bound,'FORMAL_KL_NUMERICAL_AUDIT_FAIL'
        del student
    del fp
    save(f'archive/DOC_{doc:02d}_NUMERICAL_AUDIT.json',{'status':'PASS','document_index':doc,'rows':audits,'threshold_changed':False,'GPU_forward_count':0})
    paths=[]
    for top in ['references','anchors','students']:
        paths += [p for p in (ROOT/f'{top}/doc_{doc:02d}').rglob('*') if p.is_file()]
    assert not any(p.name.endswith('.tmp') for p in paths)
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
    save(f'archive/DOC_{doc:02d}_ASSET_CATALOG.json',{'status':'DOCUMENT_COMPLETE_NO_SCIENCE_CLASSIFICATION','document_index':doc,'files':files,'FP_identity':ref['reference_identity'],'anchor_identity':anchor['anchor_identity'],'science_spec_SHA':sha(ROOT/'FROZEN_SCIENCE_SPEC.json')})
    partial=REMOTE_BASE+f'/.doc_{doc:02d}.partial';final=REMOTE_BASE+f'/doc_{doc:02d}'
    remote_py('import os,json\np='+repr(partial)+'\nf='+repr(final)+"\nassert not os.path.exists(p) and not os.path.exists(f)\nos.mkdir(p)\nprint(json.dumps({'created':p}))")
    subprocess.run(['rsync','-aL','--relative','-e',shlex.join(SSH[:-1])]+[f'{top}/doc_{doc:02d}/' for top in ['references','anchors','students']]+[TARGET+':'+partial+'/'],cwd=ROOT,check=True)
    catalog=ROOT/f'archive/DOC_{doc:02d}_ASSET_CATALOG.json'
    subprocess.run(['rsync','-a','-e',shlex.join(SSH[:-1]),str(catalog),TARGET+':'+partial+'/ASSET_CATALOG.json'],check=True)
    verify="""import json,pathlib,hashlib,os,time
p=pathlib.Path(PARTIAL);cat=json.loads((p/'ASSET_CATALOG.json').read_text())
for x in cat['files']:
 f=p/x['path'];h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 assert f.stat().st_size==x['bytes'] and h.hexdigest()==x['sha256'],x['path']
receipt={'status':'PASS_ALL_FILE_SHA_BYTES','file_count':len(cat['files']),'bytes':sum(x['bytes'] for x in cat['files']),'catalog_SHA':hashlib.sha256((p/'ASSET_CATALOG.json').read_bytes()).hexdigest(),'time_unix':time.time(),'final_directory':FINAL}
with (p/'ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
os.rename(p,FINAL);print(json.dumps(receipt))
""".replace('PARTIAL',repr(partial)).replace('FINAL',repr(final))
    receipt=remote_py(verify);assert receipt['catalog_SHA']==sha(catalog)
    save(f'archive/DOC_{doc:02d}_ARCHIVE_VERIFICATION.json',receipt)
    # Only this new complete document's verified tensor copies are relocated.
    moved=[]
    for f in files:
        if f['path'].endswith('.pt'):
            path=ROOT/f['path'];assert str(path).startswith(str(ROOT)+'/') and path.exists()
            assert sha(path)==f['sha256'];path.unlink()
            moved.append(dict(f,current_server='202.38.247.29',current_path=final+'/'+f['path'],original_manifest_path=str(path)))
    save(f'archive/DOC_{doc:02d}_CURRENT_ASSET_LOCATIONS.json',{'status':'VERIFIED_LOSSLESS_RELOCATION','document_index':doc,'assets':moved,'old_scientific_assets_touched':False,'metadata_retained_on_compute_server':True,'not_compressed_or_rounded':True})
    return receipt

def math_fsum(values):
    import math
    return math.fsum(values)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='preflight':preflight()
    else:print(json.dumps(audit_and_archive(int(sys.argv[1]))))
