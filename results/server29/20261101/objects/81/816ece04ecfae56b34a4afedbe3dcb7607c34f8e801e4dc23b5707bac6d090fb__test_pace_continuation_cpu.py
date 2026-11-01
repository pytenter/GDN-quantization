"""Finite stdlib-only tests: real frozen old receipts, no GPU/answers/scoring."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
import pace_continuation as K
spec=importlib.util.spec_from_file_location('pace_repair_controller',ROOT/'scripts/pace_controller.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
OLD=(ROOT/K.SNAPSHOT if (ROOT/K.SNAPSHOT).is_dir() else
     ROOT.parent/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1/independent_delivery/primary_archive_small_files')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')

def row(root,path):
    p=root/path
    return dict(path=path,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.tmp.name).resolve()/'base';cls.base.mkdir()
        shutil.copytree(OLD,cls.base/K.SNAPSHOT)
        for directory in ('src','protocol','vendor','analysis'):
            shutil.copytree(ROOT/directory,cls.base/directory)
        for name in ('AIME26_DATASET_FREEZE.json','FORMAL_JOB_MANIFEST.json'):
            shutil.copyfile(ROOT/name,cls.base/name)
        eq=K.verify_equivalence(cls.base,cls.base/K.SNAPSHOT)
        review=dict(status='PASS_FRESH_DUAL_HOST_PACE_REUSE3_SOURCE_SCIENCE_MODEL',formal_count=3,
                    head_mask_sha256=K.MASK_SHA,hosts={h:dict(status='PASS',source_count=133,
                    source_sha256=K.OLD_SOURCE_SHA,science_freeze_sha256=K.OLD_SCIENCE_SHA,
                    full_model17_SHA_PASS=True) for h in ('55','29')},
                    accepted_domain_equivalence=dict(status='PASS',old_quantizer_sha256=eq['old_quantizer_sha256'],
                        new_quantizer_sha256=eq['new_quantizer_sha256']))
        write(cls.base/K.FRESH_REVIEW,review)
        files=[row(cls.base,p.relative_to(cls.base).as_posix()) for p in (cls.base/K.SNAPSHOT).rglob('*') if p.is_file()]
        files.append(row(cls.base,K.FRESH_REVIEW))
        write(cls.base/C.SOURCE,dict(task_id=C.TASK,count=len(files),files=files))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name).resolve()/'new'
        shutil.copytree(self.base,self.root)
    def tearDown(self):self.t.cleanup()
    def test_real_old3_and_qualifications_have_strict_credit(self):
        value=K.reuse_admission(self.root)
        self.assertEqual(value['reused_formal_ids'],list(K.REUSED_IDS))
        self.assertEqual(len(value['proofs']),6)
        self.assertEqual(value['old_failed_partial_credit'],0)
        self.assertEqual(len(K.remaining_jobs(self.root)),57)
        self.assertEqual(K.remaining_jobs(self.root)[0]['id'],K.FAILED_ID)
    def test_completed_rows_only_returns_original3(self):
        result=K.completed_rows(self.root)
        self.assertEqual([x[0]['id'] for x in result],list(K.REUSED_IDS))
        self.assertTrue(all(str(t).startswith(str(self.root/K.SNAPSHOT)) for j,w,t in result))
    def test_original_costs_are_not_new_intervals(self):
        result=K.continuation_summary(self.root)
        self.assertEqual(result['reused_formal_count'],3)
        self.assertEqual(result['current_formal_count'],0)
        self.assertTrue(result['current_intervals_only'])
        self.assertEqual(result['historical_formal_completed'],3)
        self.assertFalse(result['runtime_reuse_admitted'])
        self.assertEqual(result['admitted_reused_formal_count'],0)
        self.assertGreater(len(result['historical_costs']['intervals']),0)
    def test_foreign_origin_snapshot_symlink_rejected(self):
        path=self.root/K.SNAPSHOT;shutil.rmtree(path);path.symlink_to(OLD,target_is_directory=True)
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_wrong_source_pin_rejected(self):
        (self.root/K.SNAPSHOT/C.SOURCE).write_text('{}')
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_wrong_permit_pin_rejected(self):
        (self.root/K.SNAPSHOT/'GPU_PERMIT.json').write_text('{}')
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_wrong_mask_rejected(self):
        (self.root/K.SNAPSHOT/'calibration/head_mask.json').write_text('{}')
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_wrong_UUID_receipt_rejected(self):
        p=C.unit_target(self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])/'DISPATCH.json'
        value=C.read(p);value['gpu_uuid']='GPU-foreign';write(p,value)
        with self.assertRaises(RuntimeError):K.verify_old_unit(self.root,self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])
    def test_wrong_origin_dispatch_rejected(self):
        p=C.unit_target(self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])/'DISPATCH.json'
        value=C.read(p);value['execution_root']='/foreign/root';write(p,value)
        with self.assertRaises(RuntimeError):K.verify_old_unit(self.root,self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])
    def test_positive_RESULT_nonzero_exit_never_credit(self):
        p=C.unit_target(self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])/'RESOURCE_INTERVAL.json'
        value=C.read(p);value['returncode']=-11;write(p,value)
        with self.assertRaises(RuntimeError):K.verify_old_unit(self.root,self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])
    def test_release_false_rejected(self):
        p=C.unit_target(self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])/'RESOURCE_INTERVAL.json'
        value=C.read(p);value['released']=False;write(p,value)
        with self.assertRaises(RuntimeError):K.verify_old_unit(self.root,self.root/K.SNAPSHOT,'formal',K.REUSED_IDS[0])
    def test_duplicate_old_CURRENT_artifact_rejected(self):
        C.unit_target(self.root,'formal',K.REUSED_IDS[0]).mkdir(parents=True)
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_no_old_qual_or_oldformal_dispatch(self):
        for args in [('math_gpu_gates','GATES_ABC'),('calibration','CALIBRATION_ONCE'),
                     ('gate_d','FULL_MODEL_D'),('formal',K.REUSED_IDS[0])]:
            with self.assertRaises(RuntimeError):K.dispatch_allowed(self.root,*args)
        K.dispatch_allowed(self.root,K.GATE_STAGE,K.GATE_UNIT)
        K.dispatch_allowed(self.root,'formal',K.FAILED_ID)
    def test_snapshot_manifest_source_bound(self):
        value=C.read(self.root/C.SOURCE);value['files']=value['files'][:-2];write(self.root/C.SOURCE,value)
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_fresh_modelreview_required(self):
        value=C.read(self.root/K.FRESH_REVIEW);value['hosts']['55']['full_model17_SHA_PASS']=False
        write(self.root/K.FRESH_REVIEW,value)
        with self.assertRaises(RuntimeError):K.reuse_admission(self.root)
    def test_buffer_math_change_rejects_reuse(self):
        p=self.root/'src/pace_buffer.py';p.write_text(p.read_text()+'\n# altered bytes\n')
        with self.assertRaises(RuntimeError):K.verify_equivalence(self.root,self.root/K.SNAPSHOT)
    def test_other_quantizer_mapping_change_rejected(self):
        p=self.root/'src/pace_quantizer.py';p.write_text(p.read_text().replace('q.clamp(0, qmax)','q.clamp(1, qmax)'))
        with self.assertRaises(RuntimeError):K.verify_equivalence(self.root,self.root/K.SNAPSHOT)
    def test_preserved_postcast_guard_cannot_be_weakened(self):
        p=self.root/'src/pace_quantizer.py';p.write_text(p.read_text().replace('effective <= 0','effective < 0'))
        with self.assertRaises(RuntimeError):K.verify_equivalence(self.root,self.root/K.SNAPSHOT)
    def test_record_admission_is_exclusive_no_restart(self):
        K.record_admission(self.root)
        with self.assertRaises(RuntimeError):K.record_admission(self.root)
    def test_new_gate_failure_no_calibration_or_formal_admission(self):
        K.record_admission(self.root)
        p=C.unit_target(self.root,K.GATE_STAGE,K.GATE_UNIT);p.mkdir(parents=True)
        write(p/'RESOURCE_INTERVAL.json',dict(returncode=1,released=True,failure={'actual':True}))
        with self.assertRaises(RuntimeError):K.admit_reused_science(self.root)
        self.assertFalse((self.root/'calibration').exists())
    def _make_current_gate(self):
        for name in ('method_spec.json',):shutil.copyfile(ROOT/name,self.root/name)
        write(self.root/'GPU_PERMIT.json',{'CPUfixture':'only'})
        identity=dict(pid=77,starttick=1001,cmdline=['/data/ydai/miniconda3/envs/bitdecode/bin/python',
            C.REMOTE+'/scripts/pace_worker.py','--root',C.REMOTE,'--stage',K.GATE_STAGE,'--unit',K.GATE_UNIT])
        controller=dict(pid=42,starttick=1000,cmdline=['CPUfixture'])
        p=C.unit_target(self.root,K.GATE_STAGE,K.GATE_UNIT);p.mkdir(parents=True)
        session=dict(task_id=C.TASK,identity=controller,source_manifest_sha256=C.sha(self.root/C.SOURCE),
                     permit_sha256=C.sha(self.root/'GPU_PERMIT.json'))
        write(self.root/'SESSION_STARTED.json',session)
        dispatch=dict(task_id=C.TASK,stage=K.GATE_STAGE,unit=K.GATE_UNIT,execution_root=C.REMOTE,
            controller=controller,gpu=0,gpu_uuid=C.UUID,source_manifest_sha256=session['source_manifest_sha256'],
            permit_sha256=session['permit_sha256'])
        write(p/'DISPATCH.json',dispatch);write(p/'ACTUAL_CHILD_IDENTITY.json',identity)
        write(p/'STARTED.json',dict(task_id=C.TASK,stage=K.GATE_STAGE,unit=K.GATE_UNIT,identity=identity,
              dispatch_sha256=C.sha(p/'DISPATCH.json')))
        write(p/'DRIVER_IDENTITY.json',dict(status='PASS',actual_uuid=C.UUID,CUDA_VISIBLE_DEVICES=C.UUID,
                                           logical_ordinal=0,visible_device_count=1))
        write(p/'RESOURCE_INTERVAL.json',dict(task_id=C.TASK,stage=K.GATE_STAGE,unit=K.GATE_UNIT,
            pid=identity['pid'],identity=identity,returncode=0,released=True,failure=None,
            accounting_final=True,lower_bound_only=False,NVML={'apps':[]}))
        value=dict(status='PASS',actual_cuda=True,no_AIME_or_calibration_or_formal_credit=True,
            original_gates=dict(status='PASS',actual_cuda=True,gates={g:dict(status='PASS',actual_cuda=True) for g in 'ABC'}),
            scale_boundary_tests=dict(status='PASS',actual_cuda=True))
        wrapper=dict(status='PASS',task_id=C.TASK,stage=K.GATE_STAGE,unit=K.GATE_UNIT,gpu=0,gpu_uuid=C.UUID,
            identity=identity,source_manifest_sha256=session['source_manifest_sha256'],
            permit_sha256=session['permit_sha256'],science_freeze_sha256=None,result=value,
            postflight_NVML={'apps':[dict(pid=77,uuid=C.UUID)]})
        wrapper['payload_sha256']=C.objsha(wrapper);write(p/'RESULT.json',wrapper)
        return p
    def test_new_gate_success_admits_exact_old_calibration_without_rerun(self):
        K.record_admission(self.root);self._make_current_gate()
        K.admit_reused_science(self.root)
        self.assertEqual(C.sha(self.root/'calibration/head_mask.json'),K.MASK_SHA)
        self.assertEqual(C.read(self.root/'REUSED_CALIBRATION_ADMISSION.json')['calibration_GPU_reruns'],0)
        self.assertEqual(C.read(self.root/'SCIENCE_FREEZE.json')['status'],
                         'SCIENCE_FROZEN_AFTER_REPAIR_GATE_AND_ONCE_CALIBRATION_REUSE')
        self.assertFalse((self.root/'execution/CALIBRATION_ONCE').exists())
    def test_new_gate_claimed_CPU_is_rejected(self):
        K.record_admission(self.root);p=self._make_current_gate();wrapper=C.read(p/'RESULT.json')
        wrapper['result']['scale_boundary_tests']['actual_cuda']=False
        wrapper.pop('payload_sha256');wrapper['payload_sha256']=C.objsha(wrapper);write(p/'RESULT.json',wrapper)
        with self.assertRaises(RuntimeError):K.admit_reused_science(self.root)
    def test_new_gate_wrong_child_identity_is_rejected(self):
        K.record_admission(self.root);p=self._make_current_gate();identity=C.read(p/'ACTUAL_CHILD_IDENTITY.json')
        identity['starttick']+=1;write(p/'ACTUAL_CHILD_IDENTITY.json',identity)
        with self.assertRaises(RuntimeError):K.admit_reused_science(self.root)
    def test_current_failed_raw_interval_recorded_no_credit(self):
        p=C.unit_target(self.root,'formal',K.FAILED_ID);p.mkdir(parents=True)
        write(p/'RESOURCE_INTERVAL.json',dict(returncode=1,released=True,failure={'true':True}))
        invalid=[];result=K.completed_rows(self.root,invalid)
        self.assertEqual(len(result),3);self.assertEqual(len(invalid),1)
        self.assertEqual(invalid[0]['partial_completion_credit'],0)
    def test_ledger_cannot_claim_reused_as_new(self):
        write(self.root/'RESOURCE_LEDGER.json',dict(completed_units=[dict(stage='formal',unit=K.REUSED_IDS[0])]))
        with self.assertRaises(RuntimeError):K.continuation_summary(self.root)

class FiniteController(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name).resolve()
        with mock.patch.object(C,'process_identity',return_value=dict(pid=42,starttick=100,cmdline=['CPUfixture'])):
            self.c=M.Controller(self.root)
    def tearDown(self):self.t.cleanup()
    def _patches(self):
        return mock.patch.multiple(C,authorize=mock.DEFAULT,device=mock.DEFAULT,storage_guard=mock.DEFAULT,
            model_guard=mock.DEFAULT,verify_external=mock.DEFAULT,validate_jobs=mock.DEFAULT,
            verify_head_mask=mock.DEFAULT,verify_science=mock.DEFAULT,sha=mock.DEFAULT)
    def test_mock_finite_gate_then57_only(self):
        jobs=[dict(id=f'x{i}') for i in range(57)]
        with self._patches() as f, mock.patch.object(K,'record_admission',return_value={'proofs':[1,2,3]}),\
             mock.patch.object(K,'admit_reused_science') as cal,mock.patch.object(K,'remaining_jobs',return_value=jobs),\
             mock.patch.object(K,'completed_rows',return_value=[None]*60),mock.patch.object(self.c,'execute_unit') as unit:
            f['sha'].return_value='0'*64
            f['verify_external'].return_value={'status':'CPU_MOCK_ONLY'}
            f['verify_head_mask'].return_value={'status':'CPU_MOCK_ONLY'}
            f['verify_science'].return_value='0'*64
            def execute(stage,name):
                self.c.completed.append(dict(stage=stage,unit=name));return {}
            unit.side_effect=execute
            self.assertEqual(self.c.run(),0)
            self.assertEqual(unit.call_args_list[0].args,(K.GATE_STAGE,K.GATE_UNIT))
            self.assertEqual(len(unit.call_args_list),58);cal.assert_called_once()
            result=C.read(self.root/'SESSION_RESULT.json')
            self.assertEqual(result['status'],'GPU_COMPLETE_PACE48_REUSE3_PLUS57_FORMAL')
            self.assertEqual(result['formal_total_unique'],60)
    def test_mock_gatefailure_never_formal_or_cal(self):
        with self._patches() as f,mock.patch.object(K,'record_admission',return_value={'proofs':[1,2,3]}),\
             mock.patch.object(K,'admit_reused_science') as cal,mock.patch.object(self.c,'execute_unit',side_effect=RuntimeError('real gate failure')) as unit:
            f['sha'].return_value='0'*64
            f['verify_external'].return_value={'status':'CPU_MOCK_ONLY'}
            self.assertEqual(self.c.run(),1);self.assertEqual(unit.call_count,1);cal.assert_not_called()
            self.assertEqual(C.read(self.root/'SESSION_RESULT.json')['current_formal_count'],0)
    def test_existing_session_rejects_restart_before_device(self):
        write(self.root/'SESSION_STARTED.json',{})
        with mock.patch.object(C,'authorize'),mock.patch.object(C,'device') as device:
            with self.assertRaises(RuntimeError):self.c.run()
            device.assert_not_called()

class ExactRepairAuthorization(unittest.TestCase):
    def setUp(self):
        self.grant=C.read(ROOT/'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json')
        self.metadata=dict(task_id=C.TASK,execution_revision=C.REVISION,remote_root=C.REMOTE,archive_root=C.ARCHIVE)
        self.permit=dict(task_id=C.TASK,ALLOW_GPU='YES',execution_revision=C.REVISION,execution_root=C.REMOTE,
            physical_indices=[0],devices=[C.UUID],source_manifest_sha256='1'*64,method_spec_sha256='2'*64,
            human_authorization_sha256=C.SCALE_REPAIR_AUTHORIZATION_SHA,formal_new_samples=57,
            reused_formal_samples=3,formal_total_unique=60,seed=1,max_new_tokens=81920,
            watchdog_seconds=1800,cleanup_seconds=120,GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None)
    def check(self,grant_sha=None):
        return C.validate_permit_values(self.metadata,self.permit,self.grant,'1'*64,'2'*64,
                                       grant_sha or C.SCALE_REPAIR_AUTHORIZATION_SHA)
    def test_exact_one_finite_repair_permit(self):self.check()
    def test_oldroot_forbidden(self):
        self.metadata['remote_root']=K.OLD_REMOTE
        with self.assertRaises(RuntimeError):self.check()
    def test_oldrevision_forbidden(self):
        self.permit['execution_revision']=C.TASK
        with self.assertRaises(RuntimeError):self.check()
    def test_GPU1_borrow_forbidden(self):
        self.permit['physical_indices']=[1]
        with self.assertRaises(RuntimeError):self.check()
    def test_foreign_UUID_forbidden(self):
        self.permit['devices']=['GPU-foreign']
        with self.assertRaises(RuntimeError):self.check()
    def test_full60_newdispatch_forbidden(self):
        self.permit['formal_new_samples']=60
        with self.assertRaises(RuntimeError):self.check()
    def test_fourth_old_partial_credit_forbidden(self):
        self.permit['reused_formal_samples']=4
        with self.assertRaises(RuntimeError):self.check()
    def test_seed_or_length_change_forbidden(self):
        for field,value in [('seed',2),('max_new_tokens',8192)]:
            old=self.permit[field];self.permit[field]=value
            with self.assertRaises(RuntimeError):self.check()
            self.permit[field]=old
    def test_unbounded_retry_grant_forbidden(self):
        self.grant['one_new_independent_finite_execution_no_unbounded_GPU_retry']=False
        with self.assertRaises(RuntimeError):self.check()
    def test_old_human_authorization_forbidden(self):
        with self.assertRaises(RuntimeError):self.check('a'*64)
    def test_GPUh_and_elapsedcap_forbidden(self):
        self.permit['GPUh_hard_cap']=1
        with self.assertRaises(RuntimeError):self.check()

if __name__=='__main__':unittest.main(verbosity=2)
