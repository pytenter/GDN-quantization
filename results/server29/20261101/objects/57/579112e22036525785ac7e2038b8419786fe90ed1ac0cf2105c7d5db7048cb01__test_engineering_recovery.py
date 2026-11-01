"""Synthetic fault injection; never imports torch or starts GPU computations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from gpu_admission import classify,classify_window,choose,UUID
from recovery_state import atomic_json,lock,complete,recovery_action,validate_positions,sha

def idle(uuid=UUID,index='6'):
    used=15*2**20;reserved=450*2**20;total=24576*2**20
    return dict(uuid=uuid,physical=dict(returncode=0),compute=dict(returncode=0,stdout=''),fuser=dict(returncode=1),
        target=[dict(index=index,uuid=uuid,name='NVIDIA GeForce RTX 3090',total_MiB='24576',used_MiB='15',GPU_util='0')],
        direct_NVML=dict(status='PASS',memory=dict(total=total,used=used+reserved,free=total-used-reserved),
            memory_v2=dict(total=total,used=used,reserved=reserved,free=total-used-reserved),utilization=dict(gpu=0)))

class AdmissionTests(unittest.TestCase):
    def test_idle(self):self.assertEqual(classify_window([idle()]*13),'AVAILABLE_VERIFIED')
    def test_visible_busy(self):
        x=idle();x['compute']['stdout']=UUID+', 12, 1000';self.assertEqual(classify(x),'BUSY_VISIBLE_PROCESS')
    def test_unattributed_busy(self):
        x=idle();x['target'][0]['GPU_util']='100';x['direct_NVML']['utilization']['gpu']=100
        self.assertEqual(classify_window([x]*13),'BUSY_UNATTRIBUTED')
    def test_inconsistent(self):
        x=idle();x['target'][0]['GPU_util']='100';self.assertEqual(classify(x),'TELEMETRY_INCONSISTENT')
    def test_nvml_error(self):
        x=idle();x['compute']['returncode']=1;self.assertEqual(classify(x),'DRIVER_OR_NVML_ERROR')
    def test_insufficient(self):
        x=idle();x['target'][0]['total_MiB']='12288';x['direct_NVML']['memory_v2']['total']=12288*2**20;x['direct_NVML']['memory_v2']['free']-=12288*2**20
        self.assertEqual(classify(x),'INSUFFICIENT_MEMORY')
    def test_idle_to_busy(self):
        x=idle();x['compute']['stdout']='visible';self.assertEqual(classify_window([idle()]*12+[x]),'BUSY_VISIBLE_PROCESS')
    def test_transient_to_verified_idle(self):
        x=idle();x['target'][0]['GPU_util']='100';x['direct_NVML']['utilization']['gpu']=100
        self.assertEqual(classify_window([x]*10+[idle()]*3),'AVAILABLE_VERIFIED')
    def test_uuid_index_change(self):self.assertEqual(classify(idle(index='2')),'AVAILABLE_VERIFIED')
    def test_unknown_not_idle(self):
        x=idle();x['direct_NVML']['memory_v2']={'status':'UNKNOWN'};self.assertEqual(classify(x),'DRIVER_OR_NVML_ERROR')
    def test_no_qualified(self):self.assertIsNone(choose({UUID:'BUSY_UNATTRIBUTED'},{UUID:True}))
    def test_alternative_numeric_fail(self):
        self.assertIsNone(choose({UUID:'BUSY_UNATTRIBUTED','GPU-other':'AVAILABLE_VERIFIED'},{UUID:True,'GPU-other':False}))
    def test_alternative_numeric_pass(self):
        self.assertEqual(choose({UUID:'BUSY_UNATTRIBUTED','GPU-other':'AVAILABLE_VERIFIED'},{UUID:True,'GPU-other':True}),'GPU-other')

class RecoveryTests(unittest.TestCase):
    def test_atomic_commit_duplicate_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'output.json';atomic_json(p,{'a':1});h=sha(p);atomic_json(p,{'a':1});self.assertEqual(sha(p),h)
            with self.assertRaises(AssertionError):atomic_json(p,{'a':2})
    def test_partial_write_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'output.json';(Path(d)/'old.partial').write_text('corrupt unfinished')
            atomic_json(p,{'a':1});self.assertEqual(json.loads(p.read_text()),{'a':1})
    def test_duplicate_controller_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock';a=lock(p)
            with self.assertRaises(BlockingIOError):lock(p)
            a.close();b=lock(p);b.close()
    def test_success_reuse_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';e=Path(d)/'e.json';atomic_json(p,{'status':'PASS'})
            atomic_json(e,{'natural_exit':True,'result_sha256':sha(p),'unit_identity':{'document':'same'}})
            complete(p,e,{'document':'same'});p.write_text('{"status":"PASS","changed":true}')
            with self.assertRaises(AssertionError):complete(p,e,{'document':'same'})
    def test_checkpoint_and_interrupt_decisions(self):
        self.assertEqual(recovery_action(True,True,False,False),'VERIFY_AND_REUSE')
        self.assertEqual(recovery_action(False,False,True,False),'ATTACH_EXISTING_RUNNER')
        self.assertEqual(recovery_action(False,False,False,True),'RESTORE_ONLY_UNDER_FROZEN_8192_CONTRACT')
        self.assertEqual(recovery_action(False,False,False,False),'STOP_INCOMPLETE_NO_QUALIFIED_CHECKPOINT')
        self.assertEqual(recovery_action(False,False,False,True,True),'STOP_FIRST_SCIENTIFIC_FAILURE')
    def test_384_positions_no_fill(self):
        x=[dict(position=8192+64*k,kl=k/10000) for k in range(384)]
        self.assertEqual(validate_positions(x),sum(r['kl'] for r in x)/384)
        with self.assertRaises(AssertionError):validate_positions(x[:-1])
        x[2]['position']=8192
        with self.assertRaises(AssertionError):validate_positions(x)
    @unittest.skipUnless(Path('/proc/self/stat').exists(),'Linux process identity boundary; mandatory on server')
    def test_runner_receipt_survives_detached_controller(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);base=d/'unit';stub=d/'cpu_fixture.py'
            stub.write_text('import json,time,pathlib\ntime.sleep(.1)\npathlib.Path('+repr(str(base)+'.json')+').write_text(json.dumps({"status":"PASS","synthetic_fixture":True}))\n')
            intent=d/'intent.json';atomic_json(intent,dict(base=str(base),command=[sys.executable,str(stub)],unit_identity={'synthetic':True},source_commit='FIXTURE_ONLY',tag='unit',stage='MOCK'))
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('unit_runner.py')),str(intent)],start_new_session=True)
            self.assertEqual(proc.wait(timeout=10),0);complete(Path(str(base)+'.json'),Path(str(base)+'.NATURAL_EXIT.json'),{'synthetic':True})
            h=sha(Path(str(base)+'.json'));again=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('unit_runner.py')),str(intent)],capture_output=True)
            self.assertNotEqual(again.returncode,0);self.assertEqual(sha(Path(str(base)+'.json')),h)

class PipelineTests(unittest.TestCase):
    def test_controller_completed_FP_is_reused_without_launch(self):
        import recovery_controller as c
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'formal').mkdir();(d/'preregistration').mkdir()
            atomic_json(d/'SOURCE_CODE_FREEZE.json',{'source_commit':'FIXTURE','config_sha256':'CONFIG'})
            protocol=d/'preregistration/DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json';atomic_json(protocol,{'fixture':True})
            ident=dict(document='fixture',rotation='FP',stage='formal',source_commit='FIXTURE',config_sha256='CONFIG',token_sha256='TOKEN',contract_sha256=sha(protocol),execution=str(d),target_gpu_uuid='FIXTURE_UUID')
            r=d/'formal/fixture__FP__formal.json';e=d/'formal/fixture__FP__formal.NATURAL_EXIT.json'
            atomic_json(r,{'status':'PASS','synthetic_fixture':True});atomic_json(e,{'natural_exit':True,'result_sha256':sha(r),'unit_identity':ident})
            with patch.object(c,'D',d),patch.object(c.p,'verify',lambda:{'selected_gpu_uuid':'FIXTURE_UUID'}),patch.object(c,'entry_for',lambda doc:{'token_file_sha256':'TOKEN'}),patch.object(c,'admit',side_effect=AssertionError('DUPLICATE_GPU_LAUNCH')):
                c.run('fixture','FP','formal');c.run('fixture','FP','formal')
            self.assertEqual(len(list((d/'formal').glob('*.json'))),2)
            r.write_text('{"status":"PASS","changed":true}')
            with patch.object(c,'D',d),patch.object(c.p,'verify',lambda:{'selected_gpu_uuid':'FIXTURE_UUID'}),patch.object(c,'entry_for',lambda doc:{'token_file_sha256':'TOKEN'}):
                with self.assertRaises(AssertionError):c.run('fixture','FP','formal')

    def test_freeze_restart_partial_commit_and_pair_mutation(self):
        import recovery_controller as c
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'screening').mkdir();(d/'runs').mkdir();(d/'formal').mkdir()
            docs=[dict(document=f'fixture{i}',scan_index=i+1,eligible_pairs=[{'fixture':True}]) for i in range(4)]
            for row in docs:atomic_json(d/'screening'/f"{row['document']}.SELECTION_DECISION.json",row)
            freeze=d/'FROZEN_4DOC_SELECTION.json';atomic_json(freeze,dict(documents=docs))
            with patch.object(c,'D',d):
                c.frozen_selection();h=sha(freeze);c.frozen_selection();self.assertEqual(sha(freeze),h)
                freeze.write_text(json.dumps({'documents':docs[:-1]}))
                with self.assertRaises(AssertionError):c.frozen_selection()

    def test_real_selection_resumes_three_and_freezes_four(self):
        import prospective_pipeline as p
        original=Path(os.environ.get('LING_TEST_ORIGINAL_DELIVERY',str(Path(__file__).parent.parent/'delivery')))
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'screening').mkdir();(d/'prereg').mkdir()
            audit=json.loads((original/'preregistration/PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json').read_text())
            cfg=json.loads((original/'CALIBRATION_CONFIG.json').read_text())
            from unittest.mock import patch
            for f in (original/'screening').glob('*.SELECTION_DECISION.json'):(d/'screening'/f.name).write_bytes(f.read_bytes())
            # All original scientific identities remain in fixture copies; the fake fourth stays in this temp directory only.
            for n in ['INPUT_ELIGIBILITY_REPRODUCTION.json','SOURCE_CODE_FREEZE.json']:(d/n).write_bytes((original/n).read_bytes())
            for n in ['PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json','DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json','NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md']:(d/'prereg'/n).write_bytes((original/'preregistration'/n).read_bytes())
            fourth=[e for e in audit['inputs'] if e['input_identity_eligible']][3]['document'];calls=[]
            fixture=json.loads((original/'screening/pg19-test-7987.json').read_text());fixture['synthetic_fixture']=True
            def fake_run(doc):
                calls.append(doc);atomic_json(d/'screening'/f'{doc}.json',fixture);return fixture
            def tx(path,s):Path(path).write_text(s)
            with patch.object(p,'OUT',d),patch.object(p,'OLD',d/'prereg'),patch.object(p,'WT',original),patch.object(p,'verify',lambda:cfg),patch.object(p,'run',fake_run),patch.object(p,'publish',atomic_json),patch.object(p,'publish_text',tx):
                s=p.select()
            self.assertEqual(calls,[fourth]);self.assertEqual(len(s['documents']),4)
            self.assertFalse(list(d.glob('runs/*.STARTED.json')));self.assertTrue((d/'SELECTION_FREEZE_RECEIPT.json').exists())
            for f in (original/'screening').glob('*.SELECTION_DECISION.json'):self.assertEqual(sha(d/'screening'/f.name),sha(f))

    def test_full_phase_order_and_qualification_failure_barrier(self):
        import prospective_pipeline as p
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'runs').mkdir();calls=[]
            units=[dict(document=f'fixture{i}',rotation=c) for i in range(4) for c in ['FP','P1','L7']]
            sel=dict(experiment='SYNTHETIC_FIXTURE_ONLY',execution_order=[dict(document=f'fixture{i}',conditions=['FP','P1','L7']) for i in range(4)])
            atomic_json(d/'FROZEN_4DOC_SELECTION.json',sel)
            def run(doc,c,stage):
                calls.append((doc,c,stage));r=dict(status='PASS',synthetic_fixture=True,uninterrupted_suffix_full_logit_sha256=doc+c,final_full_native_cache_sha256=doc+c,audit_state_fingerprint=doc+c,checkpoint_sha256=doc+c,integrity={'passed':True},cache_inventory={'511':{'cache_bytes':512*61440},'8255':{'cache_bytes':8256*61440}})
                if stage!='formal':atomic_json(d/'runs'/f'{doc}__{c}__{stage}.json',r)
                return r
            with patch.object(p,'OUT',d),patch.object(p,'select',lambda:sel),patch.object(p,'make_long_spec',lambda s:{'new_units':units}),patch.object(p,'run',run),patch.object(p,'publish',atomic_json):p.main()
            self.assertEqual(len(calls),36);self.assertEqual([x[2] for x in calls[:24]],['parent','resumed_parity']*12)
            self.assertEqual([x[2] for x in calls[24:]],['formal']*12)
            self.assertEqual(calls[24:],[ (u['document'],u['rotation'],'formal') for u in units])
            self.assertTrue((d/'PROSPECTIVE_COMPLETE.json').exists())
            fail=d/'failure_case';fail.mkdir();(fail/'runs').mkdir();atomic_json(fail/'FROZEN_4DOC_SELECTION.json',sel);calls.clear()
            def bad_run(doc,c,stage):
                calls.append((doc,c,stage));r=dict(status='PASS',synthetic_fixture=True,uninterrupted_suffix_full_logit_sha256=doc+c,final_full_native_cache_sha256=doc+c,audit_state_fingerprint=doc+c,checkpoint_sha256=doc+c,integrity={'passed':True},cache_inventory={'511':{'cache_bytes':512*61440},'8255':{'cache_bytes':8256*61440}})
                atomic_json(fail/'runs'/f'{doc}__{c}__{stage}.json',r)
                if stage=='resumed_parity':r['final_full_native_cache_sha256']='WRONG_FIXTURE_CACHE'
                return r
            with patch.object(p,'OUT',fail),patch.object(p,'select',lambda:sel),patch.object(p,'make_long_spec',lambda s:{'new_units':units}),patch.object(p,'run',bad_run),patch.object(p,'publish',atomic_json):
                with self.assertRaises(AssertionError):p.main()
            self.assertFalse(any(x[2]=='formal' for x in calls));self.assertFalse((fail/'PROSPECTIVE_COMPLETE.json').exists())

if __name__=='__main__':
    os.environ.setdefault('LING_PROSPECTIVE_RESULTS',tempfile.gettempdir()+'/ling-engineering-fixtures-only')
    unittest.main(verbosity=2)
