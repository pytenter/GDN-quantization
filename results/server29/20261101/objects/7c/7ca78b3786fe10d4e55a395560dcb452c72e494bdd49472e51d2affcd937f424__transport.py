"""Copy-only runtime transport and development qualification. No holdout measurement."""
import argparse, hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path
sys.dont_write_bytecode = True

SOURCE = Path('/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1')
PARENT_ROOT = Path('/data01/user2/worktrees/ling-propagation-surrogate-validation-v1')
PD = PARENT_ROOT/'experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1'
P = Path('/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1')
V4 = P/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC'
ENV = P/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env'
DEST = Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1')
MIRROR = DEST/'inherited'
GPUS = [3,5]
UUIDS = {3:'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae',5:'GPU-07d9c381-888d-1111-a944-56abcc583573'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def output(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as s:json.dump(x,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')
def mapped(p):
    p=str(p)
    if p==str(SOURCE) or p.startswith(str(SOURCE)+'/'):return Path(p.replace(str(SOURCE),str(DEST),1))
    if p.startswith(str(ENV)+'/'):return Path(p.replace(str(ENV),'/data/zypan/envs/ling-kda',1))
    if p.startswith('/data01/user2/models/Ling-3.0-tiny/'):return Path(p.replace('/data01/user2/models/Ling-3.0-tiny','/data/zypan/models/Ling-3.0-tiny',1))
    return MIRROR/p.lstrip('/')

def build():
    own=SOURCE/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
    assert not (own/'run/STARTED.json').exists() and not list((own/'raw/confirmation').glob('*.json'))
    items={i['path']:i for i in read(PD/'manifests/premeasurement_sha256.json')['files']}
    extras=[PD/'reports/FINAL_REPORT.json',PD/'reports/FINAL_REPORT.md',PD/'manifests/premeasurement_sha256.json',PD/'manifests/final_sha256_inventory.json']
    extras+=list((PD/'protocol').glob('*'))
    extras+=list((PD/'scripts').glob('*.py'))
    extras+=[V4/'runtime'/f'evaluation_gpu{g}_A.json' for g in [0,1]]
    extras+=[V4/'runtime/historical_fixed_forward_manifest.json',V4/'runtime/qualified_backward_selection.json']
    for p in extras:
        if p.is_file():items[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    out=own/'transport';out.mkdir(exist_ok=True)
    archive=out/'inherited_inputs.tar'
    with tarfile.open(archive,'x') as t:
        for i in items.values():
            assert sha(i['path'])==i['sha256'],i['path']
            if '/runtime/env/' not in i['path'] and '/models/' not in i['path']:
                t.add(i['path'],arcname=i['path'].lstrip('/'),recursive=False)
    output(out/'transport_inventory.json',dict(gate='PASS',source_parent_commit='dc602d2d2d3b78346b6e89d9b6a71a9ad9ddc53b',
        source_manifest_sha256=sha(PD/'manifests/premeasurement_sha256.json'),files=list(items.values()),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,copy_only=True,holdout_measurements=0))
    print('TRANSPORT_BUILD_PASS',archive.stat().st_size,len(items),flush=True)

def verify():
    inv=read(DEST/'transport_inventory_v2.json');checked=[]
    for i in inv['files']:
        p=mapped(i['path']);assert p.stat().st_size==i['bytes'] and sha(p)==i['sha256'],('TRANSPORT_IDENTITY_FAIL',i['path'],str(p))
        checked.append(dict(source_path=i['path'],transport_path=str(p),sha256=i['sha256'],bytes=i['bytes']))
    import importlib.metadata as meta
    expected=read(mapped(V4/'runtime/v4_deterministic_runtime_manifest.json'))['package_environment']['current']
    got={p:meta.version(p) for p in expected['packages']}
    assert got==expected['packages'] and sys.version==expected['python']
    output(DEST/'transport_identity_PASS.json',dict(gate='PASS',files=checked,python=sys.version,packages=got,
        scientific_source_bytes_changed=False,package_installations=0,holdout_measurements=0))
    print('TRANSPORT_RUNTIME_IDENTITY_PASS',len(checked),flush=True)

def addendum():
    own=SOURCE/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
    old=read(own/'transport/transport_inventory.json');items={i['path']:i for i in old['files']}
    extra=list((V4/'runtime').glob('*.json'))
    archive=own/'transport/transport_addendum_v2.tar'
    with tarfile.open(archive,'x') as t:
        for p in extra:
            if str(p) in items:continue
            items[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
            t.add(p,arcname=str(p).lstrip('/'),recursive=False)
    output(own/'transport/transport_inventory_v2.json',dict(gate='PASS',files=list(items.values()),
        original_transport_inventory=sha(own/'transport/transport_inventory.json'),addendum_sha256=sha(archive),
        ENGINEERING_ONLY_RECOVERY='previous transport omitted runtime authority manifest; byte-identical runtime JSON addendum only',
        package_changes=0,scientific_measurements=0,original_archive_retained=True))

def checkout_export():
    own=SOURCE/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    scope=str(own.relative_to(SOURCE))
    entries=subprocess.check_output(['git','ls-tree','-r','-t','HEAD'],cwd=SOURCE,text=True).splitlines()
    objects={commit,subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=SOURCE,text=True).strip()}
    for entry in entries:
        info,name=entry.split('\t',1);mode,kind,oid=info.split()
        if kind=='tree' or name.startswith(scope+'/'):objects.add(oid)
    stage=own/'transport/checkouts'/commit;stage.mkdir(parents=True,exist_ok=True)
    pack=stage/'premeasurement_shallow.pack'
    with pack.open('xb') as stream:
        subprocess.run(['git','pack-objects','--stdout'],cwd=SOURCE,input=('\n'.join(sorted(objects))+'\n').encode(),stdout=stream,check=True)
    paths=subprocess.check_output(['git','ls-files','--',scope],cwd=SOURCE,text=True).splitlines()
    archive=stage/'premeasurement_checkout.tar'
    with tarfile.open(archive,'x') as t:
        for rel in paths:t.add(SOURCE/rel,arcname=rel,recursive=False)
    output(stage/'premeasurement_checkout_receipt.json',dict(commit=commit,branch='exp/ling-long-horizon-propagation-confirm-v1',
        source=str(SOURCE),target=str(DEST),pack_sha256=sha(pack),archive_sha256=sha(archive),paths=paths,
        scope='same commit+tree; inherited unrelated blobs omitted and skip-worktree; operational shallow checkout only'))
    print('CHECKOUT_EXPORT_PASS',commit,len(paths),str(stage),flush=True)

def checkout_import():
    receipt=read(DEST/'premeasurement_checkout_receipt.json')
    assert sha(DEST/'premeasurement_shallow.pack')==receipt['pack_sha256']
    assert sha(DEST/'premeasurement_checkout.tar')==receipt['archive_sha256']
    assert not (DEST/'.git').exists()
    d=DEST/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
    assert not (d/'run/STARTED.json').exists() and not list((d/'raw/confirmation').glob('*.json'))
    assert not list((d/'run/traces').glob('*/trace_freeze.json'))
    subprocess.run(['git','init','-b',receipt['branch'],str(DEST)],check=True)
    with (DEST/'premeasurement_shallow.pack').open('rb') as f:subprocess.run(['git','unpack-objects'],cwd=DEST,stdin=f,check=True)
    # Generated Git shallow boundary and ignore metadata, not scientific files.
    (DEST/'.git/shallow').write_text(receipt['commit']+'\n')
    (DEST/'.git/info/exclude').write_text('/*\n')
    subprocess.run(['git','update-ref','refs/heads/'+receipt['branch'],receipt['commit']],cwd=DEST,check=True)
    subprocess.run(['git','read-tree','HEAD'],cwd=DEST,check=True)
    allfiles=subprocess.check_output(['git','ls-files','-z'],cwd=DEST)
    subprocess.run(['git','update-index','-z','--skip-worktree','--stdin'],input=allfiles,cwd=DEST,check=True)
    with tarfile.open(DEST/'premeasurement_checkout.tar','r') as t:
        for entry in t.getmembers():
            p=(DEST/entry.name).resolve();assert p.is_relative_to(DEST) and entry.isfile()
            assert entry.name in receipt['paths'];p.parent.mkdir(parents=True,exist_ok=True)
            payload=t.extractfile(entry).read()
            if p.exists():
                if p.read_bytes()==payload:continue
                assert '/scripts/' in entry.name,('FROZEN_NONCODE_OBJECT_DIFFERS',entry.name)
                # Engineering source updates before any formal measurement;
                # exact previous commit archive remains preserved.
                tmp=p.with_name(p.name+'.prospective_update')
                with tmp.open('xb') as s:s.write(payload)
                os.replace(tmp,p)
            else:
                with p.open('xb') as s:s.write(payload)
    tracked=('\0'.join(receipt['paths'])+'\0').encode()
    subprocess.run(['git','update-index','-z','--no-skip-worktree','--stdin'],input=tracked,cwd=DEST,check=True)
    assert subprocess.check_output(['git','status','--porcelain'],cwd=DEST,text=True).strip()==''
    assert (d/'manifests/premeasurement_sha256.json').is_file()
    output(d/'run/premeasurement_commit.json',dict(commit=receipt['commit'],transported_same_commit=True,source_authority=receipt['source']))
    print('CHECKOUT_IMPORT_CLEAN_PASS',receipt['commit'],flush=True)

def remap_value(value):
    if isinstance(value,dict):return {k:remap_value(v) for k,v in value.items()}
    if isinstance(value,list):return [remap_value(v) for v in value]
    if isinstance(value,str) and value.startswith('/data01/user2/'):return str(mapped(value))
    return value

def scientific_modules():
    import importlib.util
    parent=mapped(PD);sys.path.insert(0,str(parent/'scripts'))
    spec=importlib.util.spec_from_file_location('common',parent/'scripts/common.py')
    c=importlib.util.module_from_spec(spec);sys.modules['common']=c;spec.loader.exec_module(c)
    # Operational relocation only. Exact scientific source bytes are immutable.
    c.D=DEST/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1';c.R=DEST
    c.TASK='LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
    c.P=mapped(P);c.OLD=mapped(c.OLD);c.V4=mapped(V4);c.V3=mapped(c.V3)
    c.ENV=Path('/data/zypan/envs/ling-kda');c.PYTHON=c.ENV/'bin/python';c.MODEL=Path('/data/zypan/models/Ling-3.0-tiny')
    c.PARENTS=[mapped(p) for p in c.PARENTS];c.UUIDS=UUIDS
    original_load=c.load;c.load=lambda p:remap_value(original_load(p))
    def hardware():
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'],text=True).splitlines()
        rows=[]
        for line in lines:
            a=[v.strip() for v in line.split(',')];rows.append(dict(physical_gpu=int(a[0]),gpu_uuid=a[1],name=a[2],total_mib=int(a[3]),used_mib=int(a[4]),driver=a[5]))
        index={r['physical_gpu']:r for r in rows}
        for g in GPUS:assert index[g]['gpu_uuid']==UUIDS[g] and '3090' in index[g]['name']
        return dict(host=subprocess.check_output(['hostname'],text=True).strip(),gpus=rows)
    c.hardware=hardware
    import capture,replay_core,worker,surrogate_statistics
    return c,capture,replay_core,worker,surrogate_statistics

def canary(gpu):
    assert gpu in GPUS and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    c,cap,core,_,_=scientific_modules();hw=c.hardware()
    assert hw['gpus'][gpu]['used_mib']<2000 and core.torch.cuda.device_count()==1
    c.runtime();source=c.V3/'inputs/traces/wikitext2raw-train-000.pt'
    ids=[int(v) for v in core.torch.load(source,map_location='cpu',weights_only=False)['token_ids'][:128]]
    runs=[]
    for repeat in range(2):
        model=cap.Capture()
        try:trace=model.document(ids,c.D/'runtime'/f'3090_canary_gpu{gpu}_repeat{repeat}',block=32,keep_step_hashes=True)
        finally:model.close()
        state=core.torch.zeros((18,16,128,128),device='cuda',dtype=core.torch.float32)
        with core.torch.inference_mode():
            for entry in trace['chunks']:
                x=core.torch.load(entry['artifact']['path'],map_location='cuda',weights_only=False)
                for t in range(x['q'].shape[1]):
                    rec={k:x[k][:,t:t+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                    state=core.update_pre_q(state,rec)
                assert core.torch.equal(state,x['fp_after']),('3090_CAPTURE_REPLAY_BITWISE_PARITY_FAIL',gpu,repeat)
        runs.append(trace)
    assert runs[0]['step_hashes']==runs[1]['step_hashes'],('3090_REPEAT_BITWISE_FAIL',gpu)
    expected=c.load(c.V4/'runtime/evaluation_gpu0_A.json')['numeric']['FP']
    cross=all(runs[0]['step_hashes'][t]==expected[t]['state'] for t in range(4))
    output(c.D/'runtime'/f'3090_canary_gpu{gpu}.json',dict(gate='PASS',physical_gpu=gpu,gpu_uuid=UUIDS[gpu],
        CAPTURE_REPLAY_BITWISE_PARITY='PASS',SAME_GPU_REPEAT_BITWISE='PASS',V4_FP_RUNTIME_BYTE_IDENTITY='PASS' if cross else 'FAIL',
        development_source=c.row(source),trace_manifests=[c.row(c.D/'runtime'/f'3090_canary_gpu{gpu}_repeat{i}'/'trace_freeze.json') for i in range(2)],
        no_holdout_documents=True,holdout_measurements=0,hardware=hw,
        fp_policy='All new confirmation FP traces captured on assigned3090; no4090 FP traces substituted or mixed. Cross-hardware bitwise result separately retained, never asserted PASS by inference.'))
    print('3090_QUALIFICATION_PASS',gpu,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','addendum','verify','canary','checkout_export','checkout_import']);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='build':build()
    elif a.mode=='verify':verify()
    elif a.mode=='addendum':addendum()
    elif a.mode=='checkout_export':checkout_export()
    elif a.mode=='checkout_import':checkout_import()
    else:canary(a.gpu)
