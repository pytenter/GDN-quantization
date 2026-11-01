"""Actual authorization function, isolated CPU fixtures; zero GPU credit."""
from pathlib import Path
import copy,importlib.util,json,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import execution_control as C

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,sort_keys=True)+'\n')

class Authorization(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.exact=C.EXECUTION_ROOT;self.original_grant=C.HUMAN_CUPTI_AUTHORIZATION_SHA256;C.EXECUTION_ROOT=str(self.root)
        for n in ['operations/HUMAN_TASK_AUTHORIZATION.json','operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json','operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json','CALIBRATION_REUSE_ADMISSION.json','GPU_EXECUTION_PLAN.json','CALIBRATION_DATA_FREEZE.json','TRANSPORTED_QUERY_OMEGA.json','CALIBRATION_PARAMETER_SELECTION.json']:
            d=json.loads((ROOT/n).read_text())
            if n.endswith('HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'):d['new_root']=str(self.root)
            write(self.root/n,d)
        C.HUMAN_CUPTI_AUTHORIZATION_SHA256=C.sha(self.root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json')
        # Explicit CPU-only stand-ins for the additional ELF/header and finite
        # engineering2 proof; production validates exact pinned files normally.
        elf=self.root/'external/libcupti.so.12';header=self.root/'external/cupti_activity.h';elf.parent.mkdir();elf.write_bytes(b'CPU_MOCK_ELF_NOT_LOADED');header.write_bytes(b'CPU_MOCK_HEADER_NOT_API')
        def row(path):
            t=path.stat();return dict(path=str(path),bytes=t.st_size,sha256=C.sha(path),inode=t.st_ino,device=t.st_dev,mtime_ns=t.st_mtime_ns)
        write(self.root/'analysis/CUPTI_LIFECYCLE_BINDING.json',dict(status='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS',execution_root=str(self.root),files=[row(elf)],header=row(header),CPU_MOCK_ONLY=True))
        for name in ['src/native_profiler_lifecycle.py','src/profiler_detach_lifecycle.py','operations/cupti_lifecycle_preflight_once.py']:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU_MOCK_ONLY_HELPER_NOT_EXECUTED\n')
        helper_names=['src/native_profiler_lifecycle.py','src/profiler_detach_lifecycle.py','analysis/CUPTI_LIFECYCLE_BINDING.json','operations/cupti_lifecycle_preflight_once.py']
        write(self.root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json',dict(replicates=['R0','R1'],UUID=C.UUIDS[1],files=[dict(path=n,bytes=(self.root/n).stat().st_size,sha256=C.sha(self.root/n)) for n in helper_names],CPU_MOCK_ONLY=True))
        write(self.root/'engineering/cupti_lifecycle_preflight/RESULT.json',dict(status='PASS_TWO_FRESH_ACTUAL_CUDA_PROFILE_DETACH_OS_EXIT_RELEASE',freeze_sha256=C.sha(self.root/'ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json'),qualification_credit=0,AIME_generations=0,model_forward_calls=0,QDQ_calls=0,results=[dict(replicate=n,actual_returncode=0,released=True,UUID=C.UUIDS[1],identity=dict(pid=987654321,starttick=123,cmdline=['CPU_MOCK_ONLY'])) for n in ['R0','R1']],CPU_MOCK_ONLY=True))
        for name in ['SOURCE_FREEZE.json','GPU_PERMIT.json','operations/FINAL_TRUE_STOP_INDEPENDENT_REVIEW.json']:
            source=ROOT/'provenance/PRIOR_REUSE48_TRUE_STOP'/name;target=self.root/'provenance/PRIOR_REUSE48_TRUE_STOP'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        write(self.root/'SOURCE_FREEZE.json',{'files':[],'CPU_fixture_only':True})
        write(self.root/'QUALIFICATION_REUSE_ADMISSION.json',{'CPU_fixture_only':True})
        p=json.loads((ROOT/'provenance/BYTEVIEW_TRUE_STOP_METADATA/GPU_PERMIT.json').read_text())
        p.update(execution_root=str(self.root),execution_revision=C.EXECUTION_REVISION,scope='ONE_NEW_INDEPENDENT_CUPTI_REPAIR_REUSE48_PLUS4_THEN90_ARCHIVE29_ONCE',old_qualification_credit=48,fresh_qualification_count=4,old_formal_credit=0)
        for key,n in [('source_freeze_sha256','SOURCE_FREEZE.json'),('plan_sha256','GPU_EXECUTION_PLAN.json'),('human_grant_sha256','operations/HUMAN_TASK_AUTHORIZATION.json'),('byteview_repair_authorization_sha256','operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json'),('reduction_repair_authorization_sha256','operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json'),('calibration_reuse_admission_sha256','CALIBRATION_REUSE_ADMISSION.json'),('reuse48_authorization_sha256','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'),('qualification_reuse_admission_sha256','QUALIFICATION_REUSE_ADMISSION.json'),('cupti_repair_authorization_sha256','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'),('cupti_lifecycle_binding_sha256','analysis/CUPTI_LIFECYCLE_BINDING.json'),('cupti_engineering_freeze_sha256','ENGINEERING_CUPTI_LIFECYCLE_PREFLIGHT_FREEZE.json'),('cupti_engineering_result_sha256','engineering/cupti_lifecycle_preflight/RESULT.json'),('calibration_freeze_sha256','CALIBRATION_DATA_FREEZE.json'),('omega_manifest_sha256','TRANSPORTED_QUERY_OMEGA.json'),('parameter_selection_sha256','CALIBRATION_PARAMETER_SELECTION.json')]:p[key]=C.sha(self.root/n)
        self.permit=p;write(self.root/'GPU_PERMIT.json',p)
    def tearDown(self):C.EXECUTION_ROOT=self.exact;C.HUMAN_CUPTI_AUTHORIZATION_SHA256=self.original_grant;self.tmp.cleanup()
    def admit(self):return C.authorize(self.root,self.root/'GPU_PERMIT.json')
    def reject_permit(self,key,value):
        self.permit[key]=value;write(self.root/'GPU_PERMIT.json',self.permit)
        with self.assertRaises(RuntimeError):self.admit()
    def test_positive_actual_authorize(self):self.assertEqual(self.admit()['fresh_qualification_count'],4)
    def test_GPU0_forbidden(self):self.reject_permit('physical_indices',[0,2,3])
    def test_server29_UUID_forbidden(self):self.reject_permit('devices',['GPU-de53f243-b271-521c-d875-d96611ed8574',C.UUIDS[2],C.UUIDS[3]])
    def test_wrong_root(self):self.reject_permit('execution_root','/tmp/old')
    def test_wrong_revision(self):self.reject_permit('execution_revision',C.BYTEVIEW_ORIGIN_REVISION)
    def test_wrong_scope(self):self.reject_permit('scope','RETRY_FOREVER')
    def test_reuse49_refused(self):self.reject_permit('old_qualification_credit',49)
    def test_fresh52_refused(self):self.reject_permit('fresh_qualification_count',52)
    def test_formal_old_credit_refused(self):self.reject_permit('old_formal_credit',1)
    def test_wrong_source_pin(self):self.reject_permit('source_freeze_sha256','0'*64)
    def test_wrong_admission_pin(self):self.reject_permit('qualification_reuse_admission_sha256','0'*64)
    def test_wrong_human_pin(self):self.reject_permit('reuse48_authorization_sha256','0'*64)
    def test_wrong_seed(self):self.reject_permit('seed',2)
    def test_shortened_cap(self):self.reject_permit('max_new_tokens',4096)
    def test_timecap_refused(self):self.reject_permit('unit_elapsed_cap',3600)
    def test_extra_CUPTI_pin_refused(self):self.reject_permit('cupti_lifecycle_binding_sha256','0'*64)
    def test_engineering_result_pin_refused(self):self.reject_permit('cupti_engineering_result_sha256','0'*64)
    def test_engineering_freeze_pin_refused(self):self.reject_permit('cupti_engineering_freeze_sha256','0'*64)
    def test_extra_CUPTI_changed_bytes_refused(self):
        (self.root/'external/libcupti.so.12').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):self.admit()
    def test_header_changed_bytes_refused(self):
        (self.root/'external/cupti_activity.h').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):self.admit()
    def test_prior368_failed_stop_not_reclassified(self):
        write(self.root/'provenance/PRIOR_REUSE48_TRUE_STOP/operations/FINAL_TRUE_STOP_INDEPENDENT_REVIEW.json',dict(status='FORGED_PASS'))
        with self.assertRaises(RuntimeError):self.admit()
    def test_engineering_helper_changed_after_probe_refused(self):
        (self.root/'src/native_profiler_lifecycle.py').write_text('changed after CPU mock probe')
        with self.assertRaises(RuntimeError):self.admit()
    def test_unknown_origin_refused(self):
        n='operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json';d=json.loads((self.root/n).read_text());d['prior_stopped_revision']='FOREIGN';write(self.root/n,d);self.permit['reuse48_authorization_sha256']=C.sha(self.root/n);write(self.root/'GPU_PERMIT.json',self.permit)
        with self.assertRaises(RuntimeError):self.admit()
    def test_infinite_retry_refused(self):
        n='operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json';d=json.loads((self.root/n).read_text());d['single_new_independent_execution_not_restart_of_old_root']=False;write(self.root/n,d);self.permit['reuse48_authorization_sha256']=C.sha(self.root/n);write(self.root/'GPU_PERMIT.json',self.permit)
        with self.assertRaises(RuntimeError):self.admit()

if __name__=='__main__':unittest.main()
