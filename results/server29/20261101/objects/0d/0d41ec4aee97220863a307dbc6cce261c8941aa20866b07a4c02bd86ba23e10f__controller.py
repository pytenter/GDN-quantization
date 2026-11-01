"""Operational clean-worktree transport and strictly ordered gate controller."""
import hashlib, importlib.util, json, os, subprocess, sys, tarfile, time, traceback
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parents[1]
ROOT=D.parents[1]
TASK='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
SOURCE=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1')
TARGET=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v1')
RD=TARGET/'experiments'/TASK
SSH=['ssh','-n','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes','-o','ServerAliveInterval=15','zypan@202.38.247.29']
SCP=['scp','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes']
PY='/data/zypan/envs/ling-kda/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def command(s):return subprocess.run(SSH+[s],check=True)
def module():
    sp=importlib.util.spec_from_file_location('new_observability_bridge',D/'scripts/bridge.py')
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def source_guard():
    m=module();m.guard()
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True)
    assert all(l[3:].startswith('experiments/'+TASK+'/') for l in status.splitlines())
    legacy=read(D/'historical_assets/asset_manifest.json')
    k=legacy['legacy_worktree']
    before=(D/'historical_assets/git_status_porcelain_v1.txt').read_bytes()
    assert subprocess.check_output(['git','status','--porcelain=v1'],cwd=k)==before,'LEGACY_STATUS_CHANGED'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=k,text=True).strip()==legacy['legacy_HEAD']
    for item in legacy['files']:assert sha(item['path'])==item['sha256'],'LEGACY_ASSET_CHANGED'
    return m
def export():
    m=source_guard();receipt=read(D/'premeasurement_commit.json');commit=receipt['commit']
    scope='experiments/'+TASK
    # Clean sparse worktree: committed experiment files and historical byte
    # snapshots are present; inherited runtime files retain their audited mapping.
    entries=subprocess.check_output(['git','ls-tree','-r','-t','HEAD'],cwd=ROOT,text=True).splitlines()
    objects={commit,subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()}
    for entry in entries:
        info,name=entry.split('\t',1);mode,kind,oid=info.split()
        if kind=='tree' or name.startswith(scope+'/'):objects.add(oid)
    folder=D/'transport';folder.mkdir(exist_ok=True)
    with (folder/'premeasurement.pack').open('xb') as f:
        subprocess.run(['git','pack-objects','--stdout'],cwd=ROOT,input=('\n'.join(sorted(objects))+'\n').encode(),stdout=f,check=True)
    dump(folder/'checkout_receipt.json',dict(commit=commit,source=str(ROOT),target=str(TARGET),scope=scope,
         sparse_clean_worktree=True,pack_sha256=sha(folder/'premeasurement.pack'),legacy_dirty_tree_used_as_source=False,
         historical_dirty_assets_disclosed=True))
    command('test ! -e '+str(TARGET)+' && mkdir -p '+str(TARGET))
    for name in ['premeasurement.pack','checkout_receipt.json']:
        subprocess.run(SCP+[str(folder/name),'zypan@202.38.247.29:'+str(TARGET/name)],check=True)
    command('cd '+str(TARGET)+' && '+PY+' -c '+__import__('shlex').quote(
        "import json,hashlib,subprocess,pathlib; r=json.load(open('checkout_receipt.json')); "
        "assert hashlib.sha256(open('premeasurement.pack','rb').read()).hexdigest()==r['pack_sha256']; "
        "subprocess.run(['git','init','--bare','authority.git'],check=True); "
        "f=open('premeasurement.pack','rb'); subprocess.run(['git','--git-dir=authority.git','unpack-objects'],stdin=f,check=True); "
        "pathlib.Path('authority.git/shallow').write_text(r['commit']+'\\n'); "
        "subprocess.run(['git','--git-dir=authority.git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v1',r['commit']],check=True); "
        "subprocess.run(['git','--git-dir=authority.git','worktree','add','--no-checkout','checkout','exp/ling-long-horizon-observability-bridge-v1'],check=True); "
        "subprocess.run(['git','-C','checkout','config','core.sparseCheckout','true'],check=True); "
        "sp=subprocess.check_output(['git','-C','checkout','rev-parse','--git-path','info/sparse-checkout'],text=True).strip(); "
        "sp=pathlib.Path(sp); sp=sp if sp.is_absolute() else pathlib.Path('checkout')/sp; "
        "assert sp.resolve().is_relative_to(pathlib.Path('.').resolve()); sp.parent.mkdir(parents=True,exist_ok=True); "
        "sp.write_text('/'+r['scope']+'/\\n'); "
        "subprocess.run(['git','-C','checkout','read-tree','-mu','HEAD'],check=True)"))
    # Worktree is actually at checkout/, so relocate only this new directory's
    # operational roots. The source files themselves are byte-identical.
    for f in [D/'premeasurement_commit.json']:
        subprocess.run(SCP+[str(f),'zypan@202.38.247.29:'+str(TARGET/'checkout'/scope/f.name)],check=True)
    command('test -z "$(git -C '+str(TARGET/'checkout')+' status --porcelain=v1 --untracked-files=no)"')
    dump(folder/'transport_PASS.json',dict(gate='PASS',commit=commit,target_worktree=str(TARGET/'checkout'),
         source_old_directory_written=False,pack_sha256=sha(folder/'premeasurement.pack')))
def remote_pipeline():
    import shlex
    # Local to the new clean remote worktree. No historical worktree is a cwd.
    m=module();m.guard();co,t,c,cap,core,stats=m.modules()
    launch=read(D/'runtime_manifest.json')['expected']['launch_profile']
    env=os.environ.copy()
    for k,v in launch.items():
        if v is None:env.pop(k,None)
        else:env[k]=str(t.mapped(v)) if isinstance(v,str) and v.startswith('/data01/user2/') else str(v)
    env.update(CUDA_VISIBLE_DEVICES='6',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    stages=[['verify'],['smoke','--repeat','0'],['smoke','--repeat','1'],['parity'],['gate'],['discovery'],['analyze']]
    for args in stages:
        log=D/'discovery/logs'/('stage_'+'_'.join(args)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb') as f:subprocess.run([PY,str(D/'scripts/bridge.py'),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('STAGE_COMPLETE',args,flush=True)
    print('REMOTE_DISCOVERY_DONE_FINAL_ACTION_STOP',flush=True)
def main():
    mode=sys.argv[1]
    if mode=='export':export()
    elif mode=='remote_pipeline':remote_pipeline()
    else:raise RuntimeError('UNKNOWN_MODE')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        p=D/'controller_STOP.json'
        if not p.exists():dump(p,dict(error=repr(e),traceback=traceback.format_exc(),FINAL_ACTION='FAIL_AND_STOP',no_auto_retry=True))
        raise
