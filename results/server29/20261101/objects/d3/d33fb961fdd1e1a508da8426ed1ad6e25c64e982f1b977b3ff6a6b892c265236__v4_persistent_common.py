"""Prospectively qualified serial mapping; unchanged V2 numerical main."""
import v4_common as namespace
from v4_common import *
FREEZE=OUT/'V4_PERSISTENT_CODE_FREEZE.json'

def verify():
    namespace.FREEZE=FREEZE
    cfg,sel,freeze=namespace.verify()
    prior=checked(OUT/'V4_PORTABILITY_CODE_FREEZE.json')
    for p,h in prior['code_sha256'].items(): assert digest(p)==h,('PORTABILITY_CODE_CHANGED',p)
    assert checked(OUT/'P1_L7_PORTABILITY_GATE.json')['status']=='PASS'
    amendment=checked(OUT/'HARDWARE_MAPPING_AMENDMENT.json')
    assert amendment['status']=='APPROVED_PROSPECTIVE_V4_ONLY'
    assert amendment['V4_map']=={'pg19-test-860':6,'pg19-test-5956':6}
    assert [e['document'] for e in sel['documents']]==amendment['document_order']
    return cfg,sel,freeze

def complete(mode,doc,gpu=6,rep=0):
    v,p=namespace.complete(mode,doc,gpu,rep)
    cfg,sel,freeze=verify()
    assert v['freeze_sha256']==digest(FREEZE) and v['engine_sha256']==ENGINE_SHA
    assert v['runner_sha256']==digest(CODE/'v2_worker.py')
    assert v['pair_selection_sha256']==digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json')
    assert checked(RUNS/(tag(mode,doc,gpu,rep)+'.PREFIX_FINGERPRINTS.json'))['status']=='PASS'
    return v,p
