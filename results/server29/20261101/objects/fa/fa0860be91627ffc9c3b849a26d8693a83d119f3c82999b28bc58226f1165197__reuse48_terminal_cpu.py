"""Meaningful CPU-only negative tests of terminal accounting and safe transfer.
Mocks/temporary files have no GPU qualification credit; no network calls.
"""
import ast,copy,hashlib,importlib.util,io,json,pathlib,tarfile,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
D=load('new_reuse48_terminal',ROOT/'operations/deliver_reuse48_terminal_cpu_once.py')
O=load('new_reuse48_observer',ROOT/'operations/read_reuse48_snapshot_once.py')

def fixture(full=True):
 plan=json.loads((ROOT/'GPU_EXECUTION_PLAN.json').read_text());jobs=json.loads((ROOT/'FORMAL_JOB_MANIFEST.json').read_text())['jobs'];units=[x for x in plan['qualification_units'] if x['id'] in D.FRESH_IDS]+(jobs if full else [])
 intervals=[dict(unit=u['id'],gpu=u['gpu'],identity=dict(pid=100+i,starttick=200+i,cmdline=['python',D.REMOTE+'/scripts/gpu_unit_worker.py','--root',D.REMOTE,'--unit',u['id']]),exit_returncode=0,release='PASS') for i,u in enumerate(units)]
 ids=[x['unit'] for x in intervals];ledger=dict(intervals=intervals,completed_units=ids.copy());session=dict(status='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL' if full else 'FAIL_CLOSED_STOP',completed=ids.copy(),own_all_UUID_release=True,reused_qualification_count=48)
 return plan,jobs,session,ledger,ids

class TerminalAccounting(unittest.TestCase):
 def test_full_is48reused4fresh90formal(self):
  r=D.classify_terminal(*fixture());self.assertEqual((r['qualification_PASS'],r['reused_qualification'],r['fresh_qualification'],r['formal_complete'],r['dispatched_intervals']),(52,48,4,90,94))
 def test_partial_does_not_claim_fullaccuracy(self):
  r=D.classify_terminal(*fixture(False));self.assertFalse(r['full']);self.assertEqual(r['formal_complete'],0)
 def test_nonzero_positive_RESULT_not_legal(self):
  a=list(fixture(False));a[3]['intervals'][-1]['exit_returncode']=-11;a[2]['completed'].pop();a[3]['completed_units'].pop();a[4].pop();r=D.classify_terminal(*a);self.assertEqual(r['fresh_qualification'],3)
 def test_release_unresolved_rejected(self):
  a=list(fixture());a[3]['intervals'][0]['release']='UNRESOLVED'
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_old48_current_interval_rejected(self):
  a=list(fixture(False));a[3]['intervals'][0]['unit']='GPU1_W_FIELDS'
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_duplicate_interval_rejected(self):
  a=list(fixture());a[3]['intervals'].append(copy.deepcopy(a[3]['intervals'][0]))
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_unauthorized_GPU0_rejected(self):
  a=list(fixture());a[3]['intervals'][0]['gpu']=0
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_missing_starttick_rejected(self):
  a=list(fixture());del a[3]['intervals'][0]['identity']['starttick']
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_completed_set_mismatch_rejected(self):
  a=list(fixture());a[2]['completed'].pop()
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_full_without_all94_rejected(self):
  a=list(fixture());a[3]['intervals'].pop();a[3]['completed_units'].pop();a[2]['completed'].pop();a[4].pop()
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_unknown_session_rejected(self):
  a=list(fixture());a[2]['status']='RETRY'
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_formal_without_fresh4_rejected(self):
  a=list(fixture());a[3]['intervals'].pop(0);a[3]['completed_units'].pop(0);a[2]['completed'].pop(0);a[4].pop(0)
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)
 def test_reused_count_not_49(self):
  a=list(fixture());a[2]['reused_qualification_count']=49
  with self.assertRaises(RuntimeError):D.classify_terminal(*a)

class SafeTransfer(unittest.TestCase):
 def test_paths_preserve_case_separately(self):
  rows=D.path_map([dict(path='a/RESULT.json',bytes=1,sha256='a'*64),dict(path='a/result.json',bytes=1,sha256='b'*64)]);self.assertNotEqual(rows[0]['local_path'].casefold(),rows[1]['local_path'].casefold())
 def test_bad_paths(self):
  for path in ['../x','/tmp/x','x/../a','x\\y','.', '', 'x\x00y']:
   with self.subTest(path=path),self.assertRaises(RuntimeError):D.safe(path)
 def test_duplicate_logical_map_rejected(self):
  with self.assertRaises(RuntimeError):D.path_map([dict(path='x'),dict(path='x')])
 def test_dynamic_source_rows_SHAbytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp).resolve()/'x';p.write_bytes(b'123');row=dict(path='x',bytes=3,sha256=hashlib.sha256(b'123').hexdigest());self.assertEqual(D.verify_rows(p.parent,[row]),3);p.write_bytes(b'124')
   with self.assertRaises(RuntimeError):D.verify_rows(p.parent,[row])
 def test_symlink_roster_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp).resolve();(p/'x').write_bytes(b'1');(p/'y').symlink_to(p/'x')
   with self.assertRaises(RuntimeError):D.verify_rows(p,[dict(path='y',bytes=1,sha256=hashlib.sha256(b'1').hexdigest())])
 def _tar(self,kind='regular',wrong=False,extra=False):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp);row=dict(path='RESULT.json',bytes=3,sha256=hashlib.sha256(b'123').hexdigest());mapped=D.path_map([row]);archive=p/'a.tar'
   with tarfile.open(archive,'w') as t:
    m=tarfile.TarInfo('extra' if extra else 'RESULT.json');m.size=3
    if kind=='link':m.type=tarfile.LNKTYPE;m.linkname='other'
    t.addfile(m,io.BytesIO(b'124' if wrong else b'123'))
   if kind=='regular' and not wrong and not extra:
    D.unpack_verified(archive,p/'dest',mapped);self.assertEqual((p/'dest'/mapped[0]['local_path']).read_bytes(),b'123')
   else:
    with self.assertRaises(RuntimeError):D.unpack_verified(archive,p/'dest',mapped)
 def test_real_regular_tar_verified(self):self._tar()
 def test_tar_hardlink_rejected(self):self._tar(kind='link')
 def test_tar_bad_SHA_rejected(self):self._tar(wrong=True)
 def test_tar_unexpected_member_rejected(self):self._tar(extra=True)
 def test_existing_claim_no_second_delivery(self):
  # Check production guard is before any transport or destination writes.
  source=__import__('inspect').getsource(D._deliver_locked);self.assertLess(source.index('CLAIM, FAILURE, RESULT, DEST'),source.index('observer._query_remote'))
 def test_download_regularizes_hardlinks(self):self.assertIn('dereference=True',__import__('inspect').getsource(D.download))

class Readiness(unittest.TestCase):
 def snap(self):return dict(status='TERMINAL_SEAL_READY_FOR_INDEPENDENT_AUDIT',terminal_ready_observed=True,all_original_processes_exited_observed=True,source_freeze_sha256='a',permit_sha256='b',nvml=dict(status='PASS_READONLY_NVML',exact_owned_apps=[]),scorers=[],offline_chain_wait_observed=dict(status='PASS_CPU_OBSERVED_DIRECT_CHILD_SHELL_WAIT_STATUSES',finisher_shell_wait_status=0,controller_shell_wait_status=0,source_freeze_sha256='a',permit_sha256='b'))
 def test_ready(self):D.validate_ready(self.snap(),{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_live_parent_rejected(self):
  s=self.snap();s['all_original_processes_exited_observed']=False
  with self.assertRaises(RuntimeError):D.validate_ready(s,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_own_NVML_rejected(self):
  s=self.snap();s['nvml']['exact_owned_apps']=[{'pid':1}]
  with self.assertRaises(RuntimeError):D.validate_ready(s,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_foreign_NVML_does_not_block_release(self):
  s=self.snap();s['nvml']['foreign_apps_on_authorized_UUIDs']=[{'pid':1}];D.validate_ready(s,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_scorer_nonzero_rejected(self):
  s=self.snap();s['scorers']=[dict(actual_returncode=1,process=dict(owned_alive=False))]
  with self.assertRaises(RuntimeError):D.validate_ready(s,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_source_mismatch_rejected(self):
  with self.assertRaises(RuntimeError):D.validate_ready(self.snap(),{'SOURCE_FREEZE.json':'wrong','GPU_PERMIT.json':'b'})
 def test_missing_owned_chainwait_rejected(self):
  snap=self.snap();snap['offline_chain_wait_observed']=None
  with self.assertRaises(RuntimeError):D.validate_ready(snap,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_finisher_wait_nonzero_rejected(self):
  snap=self.snap();snap['offline_chain_wait_observed']['finisher_shell_wait_status']=139
  with self.assertRaises(RuntimeError):D.validate_ready(snap,{'SOURCE_FREEZE.json':'a','GPU_PERMIT.json':'b'})
 def test_remote_reader_compiles_with_exactpins(self):
  code='EXPECTED_SOURCE_SHA="a"\nEXPECTED_PERMIT_SHA="b"\n'+O.REMOTE_CODE.replace('REMOTE_ROOT',repr(O.REMOTE)).replace('EXPECTED_UUIDS',repr(O.UUIDS));compile(code,'blindread','exec')
 def test_exittransition_not_terminal_gone(self):
  self.assertIn("gone_statuses=('ORIGINAL_PROCESS_ABSENT','PID_REUSED_FOREIGN')",O.REMOTE_CODE)
  self.assertIn("all(x['status'] in gone_statuses for x in (controller,finisher,shell))",O.REMOTE_CODE)
  self.assertIn('not owned)',O.REMOTE_CODE)
 def test_closed_chain_missing_is_actionable(self):
  self.assertIn("'TRUE_TERMINAL_OWNED_CHAIN_WAIT_MISSING_OR_FAILED' if all_original_exited and seal is not None and not chain_ready else",O.REMOTE_CODE)
 def test_remote_auditor_compiles(self):compile(D.audit_code({}),'terminalread','exec')
 def test_strict_individual_gate_AST_preserved(self):
  production=ast.parse((ROOT/'src/control_qualification.py').read_text());expected=copy.deepcopy(next(n for n in production.body if isinstance(n,ast.FunctionDef) and n.name=='inspect_unit'));expected.name='strict_unit'
  actual=ast.parse(__import__('inspect').getsource(D.strict_unit)).body[0];self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))
 def test_no_terminal_reprepare_when_notfrozen(self):
  # This task's real final permit is absent during preparation; no network call.
  if not (ROOT/'GPU_PERMIT.json').exists():
   with self.assertRaises((FileNotFoundError,RuntimeError)):D.local_pins()

class CPUTerminalLifecycle(unittest.TestCase):
 def _fixture(self):
  temp=tempfile.TemporaryDirectory();root=pathlib.Path(temp.name).resolve();unit=next(u for u in json.loads((ROOT/'GPU_EXECUTION_PLAN.json').read_text())['qualification_units'] if u['id']==D.FRESH_IDS[0]);target=root/'execution'/unit['id'];target.mkdir(parents=True);(root/'analysis').mkdir();pins={'SOURCE_FREEZE.json':'s'*64,'GPU_PERMIT.json':'p'*64};identity={'pid':100,'starttick':200,'cmdline':['worker']};prof={'actual_CUDA_kernel_calls':16,'quantizer_CUDA_milliseconds':1.0};digest=hashlib.sha256(D.canonical(prof)).hexdigest()
  binding={'status':'PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS','files':[{'path':'/existing/libcupti.so.12','sha256':'e'*64}],'header':{'path':'/existing/cupti_activity.h','sha256':'h'*64}};(root/'analysis/CUPTI_LIFECYCLE_BINDING.json').write_text(json.dumps(binding));msha=D.stable_sha(root/'analysis/CUPTI_LIFECYCLE_BINDING.json')
  common={'identity':identity,'source_freeze_sha256':pins['SOURCE_FREEZE.json'],'permit_sha256':pins['GPU_PERMIT.json']};bound=dict(common,path='/existing/libcupti.so.12',sha256='e'*64,manifest_sha256=msha)
  pre=dict(bound,status='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL',CUPTI_API_calls=0,CUDA_Driver_API_calls=0,already_held=False);done=dict(common,status='PASS_ACTUAL_PROFILE_COLLECTION_COMPLETED_BEFORE_DETACH',profile_summary=prof,profile_summary_sha256=digest);detach=dict(bound,status='DOCUMENTED_CUPTI_FLUSH_AND_DETACH_RETURNED_SUCCESS',flush_returncode=0,finalize_returncode=0,profile_summary_sha256=digest,CUPTI_API_calls=2,profiler_collection_already_completed=True,CUDA_or_CUPTI_APIs_not_disabled=True,signal_or_exit_suppression=False,actual_process_exit_still_required=True)
  for n,o in [('RESULT.json',dict(identity=identity,result=dict(profiler=prof,profiler_native_lifecycle=detach))),('CUPTI_PRELOAD_ADMISSION.json',pre),('PROFILE_COLLECTION_COMPLETED.json',done),('CUPTI_PROFILE_DETACH.json',detach)]: (target/n).write_text(json.dumps(o))
  phases=['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN'];events=[dict(phase=p,monotonic=float(i),returncode=0) for i,p in enumerate(phases)];(target/'CUPTI_PROFILE_DETACH_EVENTS.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
  return temp,root,target,unit,pins
 def test_lifecycle_positive_still_needs_external_exit(self):
  temp,r,t,u,p=self._fixture()
  with temp:self.assertTrue(D.verify_profile_lifecycle(r,u,p)['actualexit_still_independently_required'])
 def _bad(self,filename,change):
  temp,r,t,u,p=self._fixture()
  with temp:
   f=t/filename;o=json.loads(f.read_text());change(o);f.write_text(json.dumps(o))
   with self.assertRaises(RuntimeError):D.verify_profile_lifecycle(r,u,p)
 def test_detach_nonzero_rejected(self):self._bad('CUPTI_PROFILE_DETACH.json',lambda x:x.update(finalize_returncode=7))
 def test_flush_nonzero_rejected(self):self._bad('CUPTI_PROFILE_DETACH.json',lambda x:x.update(flush_returncode=7))
 def test_lifecycle_identity_rejected(self):self._bad('CUPTI_PROFILE_DETACH.json',lambda x:x['identity'].update(starttick=999))
 def test_different_handle_path_rejected(self):self._bad('CUPTI_PROFILE_DETACH.json',lambda x:x.update(path='/foreign/libcupti.so.12'))
 def test_missing_actualprofile_digest_rejected(self):self._bad('PROFILE_COLLECTION_COMPLETED.json',lambda x:x.update(profile_summary_sha256='x'*64))
 def test_lifecycle_source_rejected(self):self._bad('CUPTI_PRELOAD_ADMISSION.json',lambda x:x.update(source_freeze_sha256='x'*64))
 def test_lifecycle_journal_order_rejected(self):
  temp,r,t,u,p=self._fixture()
  with temp:
   f=t/'CUPTI_PROFILE_DETACH_EVENTS.jsonl';rows=f.read_text().splitlines();rows[1],rows[4]=rows[4],rows[1];f.write_text('\n'.join(rows)+'\n')
   with self.assertRaises(RuntimeError):D.verify_profile_lifecycle(r,u,p)
 def test_wrong_qualification_scope_rejected(self):
  temp,r,t,u,p=self._fixture()
  with temp:
   u['id']='GPU1_W_FIELDS'
   with self.assertRaises(RuntimeError):D.verify_profile_lifecycle(r,u,p)
 def test_newroot_exact_CUPTI_scope(self):
  self.assertEqual(D.REMOTE,'/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025');self.assertEqual(D.REVISION,'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_CUPTI_REPAIR_20261025')
 def test_prior368_exact_failedorigin_kept(self):
  self.assertEqual(D.PRIOR368_SOURCE,'0ef900e88640bd255bdf29f6da8efe5b3d2e05b00c1c32fb74027e27a3c33d50');self.assertIn('actualexit_minus11_preserved=True',__import__('inspect').getsource(D.audit_prior368_origin))

class LateClosedFileEvidence(unittest.TestCase):
 def test_lateclosed_file_code_and_supplement_compile_without_network(self):
  import base64
  pins={'SOURCE_FREEZE.json':'s'*64,'GPU_PERMIT.json':'p'*64};identity={'pid':10,'starttick':100,'cmdline':['child']};finisher={'pid':20,'starttick':200,'cmdline':['finisher']};chain=dict(status='PASS_CPU_OBSERVED_DIRECT_CHILD_SHELL_WAIT_STATUSES',controller={'identity':identity},finisher={'identity':finisher},controller_shell_wait_status=1,finisher_shell_wait_status=0,shell_parent_PID=30,source_freeze_sha256=pins['SOURCE_FREEZE.json'],permit_sha256=pins['GPU_PERMIT.json'])
  class FakeTransport:
   def __init__(self):self.calls=0
   def run(self,code,**kwargs):
    compile(code,'actual_production_late_supplement_code','exec');self.calls+=1
    if self.calls==1:
     files=[]
     for n,b in [('OFFLINE_CHAIN_OBSERVED_EXIT.json',json.dumps(chain).encode()),('finisher.log',b'Closed\n')]:files.append(dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
     return dict(status='PASS_READONLY_LATE_CLOSED_FILE_BYTES',files=files)
    tree=ast.parse(code);values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body[:3]};self.assertion=values
    return dict(status='PASS_INDEPENDENT_SUPPLEMENT_SHA_BYTES_RENAMEAT2_NOREPLACE',root=values['FINAL'],sha256=values['DIGEST'],bytes=len(values['PAYLOAD']))
  transport=FakeTransport();after=dict(controller={'expected':identity},finisher={'expected':finisher},offline_shell={'expected':{'pid':30}})
  with tempfile.TemporaryDirectory() as temp:
   old=D.ROOT;D.ROOT=pathlib.Path(temp).resolve();(D.ROOT/'operations').mkdir()
   try:
    result=D.publish_supplement(transport,after,after,pins,{'counts':{'full':False},'manifest_sha256':'m'*64},{},[]);self.assertEqual(transport.calls,2);self.assertTrue((D.ROOT/'operations/REUSE48_INDEPENDENT_LATE_SUPPLEMENT.json').is_file());self.assertEqual(result['sha256'],hashlib.sha256(transport.assertion['PAYLOAD']).hexdigest())
   finally:D.ROOT=old

if __name__=='__main__':unittest.main(verbosity=2)
