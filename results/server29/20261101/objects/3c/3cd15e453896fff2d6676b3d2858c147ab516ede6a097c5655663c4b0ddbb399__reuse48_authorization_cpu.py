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
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.exact=C.EXECUTION_ROOT;C.EXECUTION_ROOT=str(self.root)
        for n in ['operations/HUMAN_TASK_AUTHORIZATION.json','operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json','operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json','CALIBRATION_REUSE_ADMISSION.json','GPU_EXECUTION_PLAN.json','CALIBRATION_DATA_FREEZE.json','TRANSPORTED_QUERY_OMEGA.json','CALIBRATION_PARAMETER_SELECTION.json']:
            d=json.loads((ROOT/n).read_text())
            if n.endswith('HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'):d['new_remote_root']=str(self.root)
            write(self.root/n,d)
        write(self.root/'SOURCE_FREEZE.json',{'files':[],'CPU_fixture_only':True})
        write(self.root/'QUALIFICATION_REUSE_ADMISSION.json',{'CPU_fixture_only':True})
        p=json.loads((ROOT/'provenance/BYTEVIEW_TRUE_STOP_METADATA/GPU_PERMIT.json').read_text())
        p.update(execution_root=str(self.root),execution_revision=C.EXECUTION_REVISION,scope='ONE_NEW_INDEPENDENT_REUSE48_PLUS4_THEN90_ARCHIVE29_ONCE',old_qualification_credit=48,fresh_qualification_count=4,old_formal_credit=0)
        for key,n in [('source_freeze_sha256','SOURCE_FREEZE.json'),('plan_sha256','GPU_EXECUTION_PLAN.json'),('human_grant_sha256','operations/HUMAN_TASK_AUTHORIZATION.json'),('byteview_repair_authorization_sha256','operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json'),('reduction_repair_authorization_sha256','operations/HUMAN_REDUCTION_REPAIR_AUTHORIZATION.json'),('calibration_reuse_admission_sha256','CALIBRATION_REUSE_ADMISSION.json'),('reuse48_authorization_sha256','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'),('qualification_reuse_admission_sha256','QUALIFICATION_REUSE_ADMISSION.json'),('calibration_freeze_sha256','CALIBRATION_DATA_FREEZE.json'),('omega_manifest_sha256','TRANSPORTED_QUERY_OMEGA.json'),('parameter_selection_sha256','CALIBRATION_PARAMETER_SELECTION.json')]:p[key]=C.sha(self.root/n)
        self.permit=p;write(self.root/'GPU_PERMIT.json',p)
    def tearDown(self):C.EXECUTION_ROOT=self.exact;self.tmp.cleanup()
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
    def test_unknown_origin_refused(self):
        n='operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json';d=json.loads((self.root/n).read_text());d['prior_true_STOP_root']='/tmp/foreign';write(self.root/n,d);self.permit['reuse48_authorization_sha256']=C.sha(self.root/n);write(self.root/'GPU_PERMIT.json',self.permit)
        with self.assertRaises(RuntimeError):self.admit()
    def test_infinite_retry_refused(self):
        n='operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json';d=json.loads((self.root/n).read_text());d['one_new_execution_only']=False;write(self.root/n,d);self.permit['reuse48_authorization_sha256']=C.sha(self.root/n);write(self.root/'GPU_PERMIT.json',self.permit)
        with self.assertRaises(RuntimeError):self.admit()

if __name__=='__main__':unittest.main()
