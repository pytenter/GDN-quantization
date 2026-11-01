"""CPU supervisor: serial qualification, qualified micro, prefrozen full runs."""
import traceback
from v3_common import *
from v3_controller import launch,wait,aggregate_qualification,alive
from v3_stages import physical_micro_gate,freeze_formal
from v3_report import terminal_report

def status(stage):
    emit(OUT/f'V3_STAGE_STATUS_{time.time_ns()}.json',dict(stage=stage,epoch=time.time(),supervisor_pid=os.getpid()))
    emit(OUT/f'RESOURCE_USAGE_LEDGER_SNAPSHOT_{time.time_ns()}.json',ledger())
    print('V3_STAGE',stage,flush=True)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'SUPERVISOR_STARTED.json').exists(),'NO_DUPLICATE_SUPERVISOR'
    emit(OUT/'SUPERVISOR_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),code_commit=verify()[2]['code_commit']))
    try:
        for gpu in (6,3):
            status(f'GPU{gpu}_SERIAL_TINY_PROBE'); wait(launch(gpu,'guardprobe',0))
            for rep in (0,1):
                status(f'GPU{gpu}_QUALIFICATION_FRESH{rep}'); wait(launch(gpu,'qualification',rep))
        aggregate_qualification(); status('BOTH_GPU_512X2_PASS')
        # Optional optimization: scheduling unavailability before any probe
        # receipt falls back to serial. A real unknown ownership/query/runtime
        # failure is still fail-closed, not an excuse to suppress a hard gate.
        unavailable=any(subprocess.run(['tmux','has-session','-t',f'ling_p1l7_v3_guardprobe_gpu{g}_fresh1'],
            capture_output=True).returncode==0 for g in (6,3))
        parallel_tags=[]; parallel=False
        if not unavailable:
            parallel_tags=[launch(g,'guardprobe',1) for g in (6,3)]
            for t in parallel_tags: wait(t)
            parallel=True
        emit(OUT/'GPU_GUARD_SCHEDULING.json',dict(status='PASS' if parallel else 'SERIAL_FALLBACK',
            qualification_order=[6,3],qualification_serial=True,subsequent_document_parallel=parallel,
            dual_tiny_guard_probe_tags=parallel_tags,parallel_is_optional_not_science_gate=True,
            serial_fallback_reason='optional scheduling session unavailable before probe launch' if unavailable else None,
            no_peer_compute_allowlist=True,science_inputs_and_order_unchanged=True))
        status('MICRO_8448_STARTED')
        tags=[launch(g,'micro') for g in (6,3)] if parallel else []
        if parallel:
            for t in tags: wait(t)
        else:
            for g in (6,3): wait(launch(g,'micro'))
        physical_micro_gate(parallel); status('MICRO_AND_PHYSICAL_GATE_PASS'); freeze_formal(parallel)
        status('FORMAL_32K_STARTED')
        tags=[launch(g,'formal') for g in (6,3)] if parallel else []
        if parallel:
            for t in tags: wait(t)
        else:
            for g in (6,3): wait(launch(g,'formal'))
        status('ALL_FORMAL_WORKERS_COMPLETED_BEFORE_SCIENTIFIC_ANALYSIS'); terminal_report()
    except Exception as e:
        stop(repr(e),'supervision',traceback=traceback.format_exc())
        # No signals. Peers see this latch at the next safe token boundary.
        for _ in range(12):
            if not any(alive(p.name.removesuffix('.PROCESS_IDENTITY.json')) for p in RUNS.glob('*.PROCESS_IDENTITY.json')): break
            time.sleep(5)
        terminal_report(repr(e)); raise

if __name__=='__main__': main()
