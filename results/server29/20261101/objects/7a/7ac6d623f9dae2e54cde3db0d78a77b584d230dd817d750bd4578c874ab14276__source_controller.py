"""Source-side transport/parent guard/finalization. No scientific metric changes."""
from pathlib import Path
import json, os, subprocess, sys, tarfile, time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import coherence as m
SSH=['ssh','-n','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes','-o','ServerAliveInterval=15','zypan@202.38.247.29']
SCP=['scp','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes']
RD=m.TARGET/'experiments'/m.TASK
PY='/data/zypan/envs/ling-kda/bin/python'

def command(s):return subprocess.run(SSH+[s],check=True)
def guard():
    r=m.read(m.D/'premeasurement_commit.json')
    assert m.run(['git','rev-parse','HEAD'],m.ROOT)==r['commit']
    status=m.run(['git','status','--porcelain','--untracked-files=all'],m.ROOT)
    for line in status.splitlines():assert line[3:].startswith('experiments/'+m.TASK+'/'),('UNRELATED_DIRTY_CHANGE',line)
    for f in m.read(m.D/'premeasurement_sha256.json')['files']:assert m.row(f['path'])==f
    return r

def main():
    receipt=guard()
    command('test ! -e '+str(m.TARGET)+' && mkdir -p '+str(RD))
    archive=m.D/'transport/frozen_checkout.tar';archive.parent.mkdir(exist_ok=True)
    fixed=[Path(r['path']) for r in m.read(m.D/'premeasurement_sha256.json')['files']]+[m.D/'premeasurement_sha256.json',m.D/'premeasurement_commit.json']
    with tarfile.open(archive,'x') as t:
        for f in fixed:t.add(f,arcname=str(f.relative_to(m.D)),recursive=False)
    subprocess.run(SCP+[str(archive),'zypan@202.38.247.29:'+str(RD/'frozen_checkout.tar')],check=True)
    command('cd '+str(RD)+' && tar -xf frozen_checkout.tar && test "$(sha256sum frozen_checkout.tar | cut -d\  -f1)" = '+m.sha(archive))
    m.dump(m.D/'transport/receipt.json',dict(gate='PASS',archive=m.row(archive),source_commit=receipt['commit'],target=str(RD)))
    command('cd '+str(RD)+' && env PYTHONDONTWRITEBYTECODE=1 '+PY+' scripts/coherence.py controller')
    guard()
    identity=m.read(m.D/'source_identity.json');oldfiles=identity['protected_historical_files']
    for f in oldfiles:assert m.row(f['path'])==f,('PROTECTED_HISTORY_CHANGED',f['path'])
    expected={str(Path(f['path'])) for f in oldfiles}
    actual={str(f) for root in [m.P,m.C] for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    assert actual==expected,('PARENT_MEMBERSHIP_CHANGED',list(actual-expected),list(expected-actual))
    for f in identity['inputs']:assert m.row(f['path'])==f,('INHERITED_INPUT_CHANGED',f['path'])
    audit=m.D/'provenance/postmeasurement_parent_integrity.json'
    m.dump(audit,dict(gate='PASS',protected_files=len(oldfiles),inherited_inputs=len(identity['inputs']),source_identity_sha256=m.sha(m.D/'source_identity.json'),source_commit=receipt['commit'],original_global_gate='FAIL_PRESERVED',FP_mixed=False))
    subprocess.run(SCP+[str(audit),'zypan@202.38.247.29:'+str(RD/'provenance/postmeasurement_parent_integrity.json')],check=True)
    command('cd '+str(RD)+' && env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6 LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8 '+PY+' scripts/coherence.py analyze')
    # Package scalar results/metadata only. Large frozen FP traces stay on assigned host.
    command('cd '+str(RD)+" && tar --exclude='./discovery/traces' --exclude='./smoke/repeat0/trace' --exclude='./smoke/repeat1/trace' --exclude='./results.tar' --exclude='./frozen_checkout.tar' -cf results.tar .")
    archive2=m.D/'transport/results.tar'
    subprocess.run(SCP+['zypan@202.38.247.29:'+str(RD/'results.tar'),str(archive2)],check=True)
    remote_sha=subprocess.check_output(SSH+['sha256sum '+str(RD/'results.tar')],text=True).split()[0]
    assert m.sha(archive2)==remote_sha
    with tarfile.open(archive2) as t:
        for member in t.getmembers():
            if member.isdir():continue
            assert member.isfile()
            dest=(m.D/member.name).resolve();assert dest.is_relative_to(m.D.resolve())
            b=t.extractfile(member).read()
            if dest.exists():assert dest.read_bytes()==b,('SOURCE_SYNC_CONFLICT',str(dest))
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(b)
    # Every transported inventory member must match the remote scientific manifest.
    checked=0;remote_only=[]
    for line in (m.D/'sha256_manifest.txt').read_text().splitlines():
        h,rel=line.split('  ',1);p=m.D/rel
        if p.exists():assert m.sha(p)==h;checked+=1
        else:
            assert rel.startswith(('discovery/traces/','smoke/repeat0/trace/','smoke/repeat1/trace/'))
            remote_only.append(rel)
    m.dump(m.D/'transport/final_byte_identity.json',dict(gate='PASS',archive_sha256=remote_sha,local_entries_checked=checked,large_trace_entries_retained_on_assigned_host=remote_only,source_commit=receipt['commit'],FINAL_ACTION='STOP_AFTER_DISCOVERY'))
    guard()
    subprocess.run(['git','add','--','experiments/'+m.TASK],cwd=m.ROOT,check=True)
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=m.ROOT,check=True)
    subprocess.run(['git','commit','-m','experiment: finalize error coherence Discovery with same-GPU6 references'],cwd=m.ROOT,check=True)
    print('FINAL_COMMIT',m.run(['git','rev-parse','HEAD'],m.ROOT),'FINAL_ACTION STOP_AFTER_DISCOVERY',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:m.stop('source_controller',e);raise
