"""Real CPU owning exits plus report/CSV completeness; fixtures are not results."""
import copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import dp32_parent as P
import dp32_reports as B
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def main():
    temp=Path(tempfile.mkdtemp(prefix='CPU_FIXTURES_ONLY_',dir=ROOT/'cpu_tests'));(temp/'runtime').mkdir()
    pin=temp/'NONEMPTY_SOURCE.txt';pin.write_text('CPU TEST ONLY, NO GPU CREDIT\n')
    put(temp/'SOURCE_MANIFEST.json',dict(files=[dict(path=pin.name,bytes=pin.stat().st_size,sha256=P.C.sha(pin))]))
    expected=P.C.sha(temp/'SOURCE_MANIFEST.json');P.ROOT=temp;codes=[]
    for code in [0,7,143]:
        rc=P.owned('CPU_EXIT_'+str(code),[sys.executable,'-B','-c','import time,sys;print("CPU_PHASE",flush=True);time.sleep(.2);sys.exit('+str(code)+')'],False,expected)
        assert rc==code;codes.append(rc)
    reports=[]
    for kind in ['STOP','COMPLETE','OVERLAP']:
        t=temp/('REPORT_'+kind);(t/'runtime').mkdir(parents=True);(t/'validation').mkdir();(t/'measurements').mkdir()
        put(t/'runtime/COMPUTE_TERMINAL.json',dict(status='STOP_NO_IDLE_GPU' if kind=='STOP' else 'BENCHMARKS_COMPLETE_PENDING_CPU_REPORT',source_sha256='CPU_FIXTURE_NOT_SCIENTIFIC'))
        if kind!='STOP':
            for mode in ['short','long']:
                for repeat in range(3):
                    for position,period in enumerate([8,16,32]):
                        speed=(5+position) if kind=='COMPLETE' else 5
                        layer=dict(base_nbytes=0,buffer_nbytes=0,peak_decode_buffer_nbytes=0,decode_buffer_capacity_nbytes=0,vector_scratch_peak_payload_bytes=0,decode_dense_materializations=0,packed_kernel_launches=0,pending_tokens=0,inventory=[])
                        x=dict(CPU_FIXTURE_ONLY=True,mode=mode,repeat=repeat,position=position,decode_period=period,prompt_tokens=16,warmup_decode_forwards=32,measured_decode_forwards=128,decode_TPS=speed,mean_decode_token_ms=1000/speed,decode_wall_s=128/speed,prefill_latency_s=1,peak_allocated_bytes=0,peak_reserved_bytes=0,quantizer_audit=dict(storage_by_layer={'0':layer}),intervals=[dict(start_decode_forward=lo,end_decode_forward=hi,TPS=speed) for lo,hi in [(2048,8192),(8192,16384)]])
                        put(t/'measurements'/(mode+str(repeat)+'_'+str(period)+'.json'),x)
        B.ROOT=t;B.main();result=json.loads((t/'RESULT.json').read_text())
        want={'STOP':'STOP_NO_IDLE_GPU','COMPLETE':'COMPLETE','OVERLAP':'COMPLETE_NO_CLEAR_SPEEDUP'}[kind];assert result['status']==want
        if kind=='STOP':assert not (t/'BENCHMARK_RAW.csv').exists()
        else:assert len((t/'BENCHMARK_RAW.csv').read_text().splitlines())==19
        reports.append(dict(fixture=kind,expected=want,actual=result['status']))
    result=dict(status='PASS_REAL_CPU_OWNING_AND_REPORT_BRANCHES',real_exit_codes=codes,report_fixtures=reports,fixtures_root=str(temp),all_fixtures_are_not_measurements=True,GPU_credit=0)
    put(ROOT/'CPU_CONTROL_VALIDATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    (ROOT/'cpu_tests').mkdir(exist_ok=True);main()
