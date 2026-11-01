"""CPU launcher compatibility supplement; never edits the frozen experiment.

The original tmux command was rejected by its option parser, before any
scientific process existed. Its claim/receipt stay immutable. This supplement
allows one actual launch only after independent proof of zero prior dispatch.
"""
import hashlib
import base64
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'f51840870f017786bf761a53804a81bfa5c4a941a2207f136349761e0272256f'
PERMIT = '32ca98b22157ec6b5c03ea648c39cc8f6e88b3d773e3ee245380fd53c5dbf2d7'
GRANT = 'f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3'
REMOTE = '/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
TMUX = 'qwen-pace48-scale-repair-gpu0-20261025'
FAILED_CLAIM = '1a8815db7e9bf5ae522eeb71fd23dcef5982866b99e3adff8dbbbd4ce0991c2f'
FAILED_RECEIPT = 'd6a57fa2cadec2c11b4833b7561c466b6f30126a46e3829f0e22f13722294982'
REVIEW = '5a5b3918894c60280f71a2fdeb7707a5fd1fd7111fd84e99b4928940a8772a72'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reviewpath = ROOT / 'operations/TMUX_COMPAT_NO_SCIENTIFIC_DISPATCH_INDEPENDENT_REVIEW.json'
    review = json.loads(reviewpath.read_text())
    assert sha(reviewpath) == REVIEW
    assert review['status'] == 'PASS_INDEPENDENT_TMUX_USAGE_FAILURE_NO_SCIENTIFIC_DISPATCH'
    assert review['source_sha256'] == SOURCE and review['permit_sha256'] == PERMIT
    assert review['human_authorization_sha256'] == GRANT
    assert review['claim_sha256'] == FAILED_CLAIM and review['launch_receipt_sha256'] == FAILED_RECEIPT
    assert not any(review['markers'].values()) and not review['current_execution_entries']
    assert not review['current_trajectory_entries'] and not review['exact_newroot_processes']
    assert not review['selected_GPU0_apps'] and review['scientific_dispatches'] == 0
    assert sha(ROOT / 'source_manifest.json') == SOURCE and sha(ROOT / 'GPU_PERMIT.json') == PERMIT
    assert not (ROOT / 'operations/CPU_TMUX_COMPAT_ACTUAL_LAUNCH_RECEIPT_LOCAL.json').exists()
    p = ROOT / 'operations/remote.py'
    assert sha(p) == 'db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
    spec = importlib.util.spec_from_file_location('tmux_compat_pinned_transport', p)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    prefix = '\n'.join(name+'='+repr(value) for name, value in [
        ('REMOTE', REMOTE), ('TMUX', TMUX), ('SOURCE', SOURCE), ('PERMIT', PERMIT),
        ('GRANT', GRANT), ('REVIEW_SHA', sha(reviewpath)),
        ('FAILED_CLAIM_SHA', FAILED_CLAIM), ('FAILED_RECEIPT_SHA', FAILED_RECEIPT),
        ('SUPPLEMENT_SHA', sha(Path(__file__))),
        ('SUPPLEMENT_B64', base64.b64encode(Path(__file__).read_bytes()).decode()),
        ('REVIEW_B64', base64.b64encode(reviewpath.read_bytes()).decode())])
    program = r'''
import base64,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
r=Path(REMOTE);sys.path.insert(0,str(r/'src'))
import pace_control as C
assert C.sha(r/C.SOURCE)==SOURCE and C.sha(r/'GPU_PERMIT.json')==PERMIT
assert C.sha(r/'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json')==GRANT
C.authorize(r);C.model_guard(r);C.storage_guard(r)
sys.path.insert(0,str(r/'operations'))
import freeze_static_once as F
F.validate_final_cpu_admission(r)
old=C.read(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json')
assert old['source_manifest_sha256']==SOURCE and old['permit_sha256']==PERMIT
assert old['returncode']==1 and old['stderr'].startswith('usage: new-session')
assert old['argv']==['tmux','new-session','-d','-s',TMUX,'--','/bin/sh',str(r/'scripts/run_offline.sh')]
assert old['claim_sha256']==C.sha(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json')
assert C.sha(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json')==FAILED_CLAIM_SHA
assert C.sha(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json')==FAILED_RECEIPT_SHA
assert not any((r/n).exists() for n in ['OFFLINE_PARENT_STARTED.json','OFFLINE_PARENT_RESULT.json','SESSION_STARTED.json','SESSION_RESULT.json','CONTINUATION_ADMISSION.json','SCIENCE_FREEZE.json','FIRST_FAILURE.json','PARENT_CONTROLLER_DISPATCH.json','PARENT_FINISHER_DISPATCH.json','DELIVERY_MANIFEST.json','CPU_TMUX_COMPAT_LAUNCH_CLAIM.json'])
assert not (r/'execution').exists() and not any((r/c/'trajectories').exists()for c in ('R2_PACE48_DP8','R3_PACE48_DP1'))
scoped=[]
for directory in Path('/proc').iterdir():
 if not directory.name.isdigit():continue
 try:
  argv=[x.decode() for x in (directory/'cmdline').read_bytes().split(b'\0') if x]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if any(REMOTE in x for x in argv):
  try:identity=C.process_identity(int(directory.name))
  except(FileNotFoundError,ProcessLookupError):continue
  scoped.append(identity)
assert not scoped,'Existing new-root process forbids a second launch'
assert subprocess.run(['tmux','has-session','-t',TMUX],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
# Keep the operator source and its independent zero-dispatch review as separately
# named CPU provenance, outside every frozen runtime import directory.
parent=r/'operations/operator_supplements'
for path in (r/'operations',parent):assert not path.is_symlink()
parent.mkdir(exist_ok=True)
supplement=parent/SUPPLEMENT_SHA;supplement.mkdir(exist_ok=False)
for name,encoded,expected in [('CPU_ONLY_TMUX_COMPAT_LAUNCH_RECOVERY.py',SUPPLEMENT_B64,SUPPLEMENT_SHA),('TMUX_COMPAT_NO_SCIENTIFIC_DISPATCH_INDEPENDENT_REVIEW.json',REVIEW_B64,REVIEW_SHA)]:
 data=base64.b64decode(encoded,validate=True);assert hashlib.sha256(data).hexdigest()==expected
 with (supplement/name).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert C.sha(supplement/name)==expected
assert C.sha(r/C.SOURCE)==SOURCE and C.sha(r/'GPU_PERMIT.json')==PERMIT
obs=C.device()
claim=dict(status='CLAIMED_ONCE_CPU_TMUX_COMPAT_FIRST_ACTUAL_SCIENTIFIC_LAUNCH',source_manifest_sha256=SOURCE,permit_sha256=PERMIT,human_authorization_sha256=GRANT,independent_zero_dispatch_review_sha256=REVIEW_SHA,operator_supplement_sha256=SUPPLEMENT_SHA,operator_provenance_directory=str(supplement),original_failed_launch_receipt_sha256=C.sha(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json'),original_claim_preserved=True,zero_prior_scientific_dispatch=True,no_frozen_source_modified=True,NVML_before=obs,unix=time.time(),no_automatic_retry=True)
C.atomic_new(r/'CPU_TMUX_COMPAT_LAUNCH_CLAIM.json',claim)
# This old tmux accepts one shell-command string, with no modern --/argv form.
argv=['tmux','new-session','-d','-s',TMUX,'exec '+shlex.join(['/bin/sh',str(r/'scripts/run_offline.sh')])]
completed=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True)
receipt=dict(status='CPU_TMUX_COMPAT_FIRST_ACTUAL_LAUNCH_COMMAND_OBSERVED',argv=argv,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,source_manifest_sha256=SOURCE,permit_sha256=PERMIT,operator_supplement_sha256=SUPPLEMENT_SHA,claim_sha256=C.sha(r/'CPU_TMUX_COMPAT_LAUNCH_CLAIM.json'),original_failed_launch_receipt_sha256=claim['original_failed_launch_receipt_sha256'],original_claim_preserved=True,unix=time.time())
if completed.returncode==0:
 pane=subprocess.run(['tmux','list-panes','-t',TMUX,'-F','#{pane_pid}'],capture_output=True,text=True)
 receipt['initial_pane_query']=dict(returncode=pane.returncode,stdout=pane.stdout,stderr=pane.stderr)
 receipt['initial_pane_identities']=[]
 for value in pane.stdout.splitlines():
  try:receipt['initial_pane_identities'].append(C.process_identity(int(value)))
  except (FileNotFoundError,ProcessLookupError,ValueError,OSError) as error:receipt['initial_pane_identities'].append(dict(pid_hint=value,status='TRANSITION_OR_EXIT_UNRESOLVED_USE_OWN_PARENT_RECEIPT',error=repr(error)))
C.atomic_new(r/'CPU_TMUX_COMPAT_ACTUAL_LAUNCH_RECEIPT.json',receipt)
print(json.dumps(receipt))
'''
    receipt = transport.run(prefix+'\n'+program, role='55',
                            python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=60)
    with (ROOT/'operations/CPU_TMUX_COMPAT_ACTUAL_LAUNCH_RECEIPT_LOCAL.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))
    assert receipt['returncode'] == 0, 'Compatibility launch failed; preserve claims, never retry automatically'


if __name__ == '__main__':
    main()
