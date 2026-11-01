"""Fully gated serial qualification -> micro -> two formal docs, no retry."""
import os
import time
import traceback
from v4_persistent_common import *
from v4_persistent_controller import launch,wait,qualification_gate,qualification_document_gate
from v4_persistent_stages import physical_micro_gate,freeze_formal
from v4_terminal_report import terminal_report

def status(stage):
    emit(OUT/f'V4_STAGE_STATUS_{time.time_ns()}.json',dict(stage=stage,epoch=time.time(),supervisor_pid=os.getpid()))
    emit(OUT/f'RESOURCE_USAGE_LEDGER_SNAPSHOT_{time.time_ns()}.json',ledger())
    print('V4_STAGE',stage,flush=True)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'V4_PERSISTENT_SUPERVISOR_STARTED.json').exists(),'NO_DUPLICATE_SUPERVISOR'
    cfg,sel,freeze=verify(); order=[e['document'] for e in sel['documents']]
    emit(OUT/'V4_PERSISTENT_SUPERVISOR_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),code_commit=freeze['code_commit'],order=order))
    try:
        for doc in order:
            for rep in (0,1):
                status(f'{doc}_QUALIFICATION_FRESH{rep}'); wait(launch('qualification',doc,rep))
            qualification_document_gate(doc)
        qualification_gate(); status('BOTH_DOCUMENT_512X2_PASS')
        for doc in order:
            status(f'{doc}_MICRO_8448_STARTED'); wait(launch('micro',doc))
        physical_micro_gate(); status('MICRO_AND_PHYSICAL_GATE_PASS'); freeze_formal()
        # Launch order is frozen. wait() accesses mechanical status only, never
        # reads student scientific values or ranks to decide the second launch.
        for doc in order:
            status(f'{doc}_FORMAL_32K_STARTED'); wait(launch('formal',doc))
        status('ALL_FORMAL_WORKERS_ENDED_BEFORE_SCIENCE_ANALYSIS'); terminal_report()
    except Exception as exc:
        stop(repr(exc),'PERSISTENT_SUPERVISION',traceback=traceback.format_exc())
        for _ in range(12):
            active=[p for p in RUNS.glob('*.PROCESS_IDENTITY.json') if alive(p.name.removesuffix('.PROCESS_IDENTITY.json'))]
            if not active: break
            time.sleep(5)
        terminal_report(repr(exc)); raise

if __name__=='__main__': main()
