"""Execute actual parent entry in isolated mocked Popen lifecycles; CPU only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types


def run(source):
    checks=[]
    original_modules={k:sys.modules.get(k) for k in ('execution_control','collect_fp_once')}
    try:
        for mode in ('exit0','exit0_identity_race','query_error_cleanup','foreign_owner_cleanup','nonzero_exit'):
            with tempfile.TemporaryDirectory(prefix='WSCALE_CPU_PARENT_ONLY_') as directory:
                root=Path(directory)/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1';(root/'calibration').mkdir(parents=True)
                receipts={};events=[];calls={'poll':0,'query':0}
                identity=dict(pid=123456789,starttick=111,cmdline=['CPU_FIXTURE_COLLECTOR'])
                class Child:
                    pid=identity['pid']
                    terminated=False
                    alive=mode in ('query_error_cleanup','foreign_owner_cleanup')
                    def poll(self):
                        calls['poll']+=1
                        if mode=='exit0_identity_race' and calls['poll']==1:return None
                        if self.alive:return None
                        return -15 if self.terminated else 2 if mode=='nonzero_exit' else 0
                    def wait(self,timeout=None):
                        self.alive=False;events.append(('wait',self.pid,timeout))
                        return self.poll()
                    def terminate(self):
                        self.terminated=True;self.alive=False;events.append(('terminate',self.pid))
                child=Child();control=types.ModuleType('execution_control')
                control.UUIDS=('CPU_ONLY_0','CPU_ONLY_1','CPU_ONLY_2','CPU_ONLY_3')
                control.device=lambda gpu:events.append(('CPU_mock_device',gpu))
                control.process_identity=lambda pid:dict(identity) if pid==child.pid else dict(pid=pid,starttick=222,cmdline=['CPU_FIXTURE_PARENT'])
                control.same_process=lambda ident:child.alive if ident==identity else True
                control.sha=lambda p:'CPU_SHA_'+str(p)
                def new(p,obj):
                    name=Path(p).name
                    if name in receipts:raise RuntimeError('duplicate immutable CPU fixture receipt')
                    receipts[name]=obj
                control.atomic_new=new
                control.read=lambda p:dict(identity=identity) if Path(p).name=='FP_COLLECTION_MANIFEST.json' else None
                control.worker_environment=lambda gpu:dict(CUDA_VISIBLE_DEVICES=control.UUIDS[gpu],CPU_FIXTURE_ONLY='YES')
                def query():
                    calls['query']+=1
                    if mode=='query_error_cleanup' and calls['query']==1:raise RuntimeError('CPU_FIXTURE_QUERY_ERROR')
                    if mode=='foreign_owner_cleanup':return dict(apps=[dict(uuid=control.UUIDS[1],pid=987654321)])
                    return dict(apps=[])
                control.query_gpu=query
                collector=types.ModuleType('collect_fp_once');collector.validate_calibration_permit=lambda r,g,c:events.append(('mock_CPU_permit_only',str(r),g))
                sys.modules['execution_control']=control;sys.modules['collect_fp_once']=collector
                ns={'__name__':'parent_cpu_fixture','__file__':'CPU_FIXTURE_PARENT_SOURCE'}
                exec(compile(source,'calibration_parent_once.py','exec'),ns)
                clock=types.SimpleNamespace(monotonic=lambda:1.,sleep=lambda seconds:None)
                ns['time']=clock;ns['subprocess']=types.SimpleNamespace(Popen=lambda *a,**k:child,STDOUT=-2)
                ns['signal']=types.SimpleNamespace(SIGTERM=15,SIGINT=2,signal=lambda *a:None)
                error=None
                try:ns['main'](root,1)
                except BaseException as exc:error=exc
                if mode in ('exit0','exit0_identity_race'):
                    assert error is None,(mode,error)
                    assert receipts['FP_COLLECTION_EXIT_RELEASE.json']['worker_identity']==identity
                    assert receipts['PARENT_RESULT.json']['exit_returncode']==0
                    assert not any(e[0]=='terminate' for e in events)
                    checks.append(mode+'_actual_exit0_receipt_and_no_signal')
                else:
                    assert error is not None and 'PARENT_FAILURE.json' in receipts
                    assert 'FP_COLLECTION_EXIT_RELEASE.json' not in receipts
                    failure=receipts['PARENT_FAILURE.json'];assert failure['cleanup']['release']=='PASS'
                    if mode=='query_error_cleanup':
                        assert 'CPU_FIXTURE_QUERY_ERROR' in failure['error']
                        assert child.terminated and failure['cleanup']['actual_exit']==-15
                        checks.append('query_error_preserves_first_cause_then_owned_exit_release')
                    elif mode=='foreign_owner_cleanup':
                        assert child.terminated and receipts['PARENT_RESULT.json']['exit_returncode']==-15
                        assert all(e[1]==child.pid for e in events if e[0]=='terminate')
                        assert 'Foreign process' in receipts['PARENT_RESULT.json']['stop_reason']
                        checks.append('foreign_owner_never_signalled_owned_Popen_only')
                    else:
                        assert receipts['PARENT_RESULT.json']['exit_returncode']==2
                        assert failure['cleanup']['actual_exit']==2 and not child.terminated
                        checks.append('nonzero_exit_preserved_without_fake_PASS')
                assert receipts['PARENT_DISPATCH.json']['identity']==identity
                assert receipts['PARENT_DISPATCH.json']['parent_identity']['pid']==os.getpid()
                checks.append(mode+'_exact_parent_and_child_dispatch_identity')
    finally:
        for key,old in original_modules.items():
            if old is None:sys.modules.pop(key,None)
            else:sys.modules[key]=old
    return dict(status='PASS_ACTUAL_PARENT_ENTRY_CPU_MOCKED_LIFECYCLE',test_count=len(checks),checks=checks,fixtures='CPU_ONLY_MOCKED_POPEN_IDENTITY_NVML_NO_GPU_PROOF',model_loads=0,GPU_calls=0,real_process_signals=0)
