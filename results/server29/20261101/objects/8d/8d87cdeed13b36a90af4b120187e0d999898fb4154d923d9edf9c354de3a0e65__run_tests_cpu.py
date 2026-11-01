import contextlib
import io
import json
import os
from pathlib import Path
import time
import unittest
import test_engineering_recovery as tests
from recovery_state import sha,atomic_json

class Results(unittest.TextTestResult):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.items=[]
    def addSuccess(self,test):super().addSuccess(test);self.items.append(dict(test=test.id(),status='PASS'))
    def addError(self,test,error):super().addError(test,error);self.items.append(dict(test=test.id(),status='ERROR'))
    def addFailure(self,test,error):super().addFailure(test,error);self.items.append(dict(test=test.id(),status='FAIL'))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.items.append(dict(test=test.id(),status='SKIP',reason=reason))

if __name__=='__main__':
    d=Path(os.environ['LING_PROSPECTIVE_RESULTS']);log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    with contextlib.redirect_stdout(log):r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Results).run(suite)
    (d/'ENGINEERING_TESTS.log').write_text(log.getvalue())
    for kind,name in [('AdmissionTests','GPU_ADMISSION_TEST_REPORT.json'),('RecoveryTests','CONTROLLER_RESUME_TEST_REPORT.json')]:
        rows=[x for x in r.items if kind in x['test'] or kind=='RecoveryTests' and 'PipelineTests' in x['test']]
        atomic_json(d/name,dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',tests=rows,CPU_synthetic_only=True,
            test_source_sha256=sha(Path(tests.__file__)),new_GPU_scientific_units=0,epoch=time.time()))
    print(json.dumps(dict(tests=r.testsRun,success=r.wasSuccessful(),skips=len(r.skipped),GPU_scientific_units=0)))
    assert r.wasSuccessful() and not r.skipped
