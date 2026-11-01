"""Observe unchanged compiler calls, including disk-cache loads; no flags changed."""
import atexit,json,time
def install(path):
    import triton.compiler as compiler
    original=compiler.compile;records=[]
    def observed(*args,**kwargs):
        start=time.perf_counter();out=original(*args,**kwargs);elapsed=time.perf_counter()-start
        src=args[0];constants={str(k):repr(v) for k,v in getattr(src,'constants',{}).items()}
        records.append((out,dict(compile_or_disk_cache_load_wall_s=elapsed,constants=constants,compiler_flags_unchanged=True)))
        return out
    compiler.compile=observed
    def finish():
        rows=[]
        for obj,row in records:
            row=dict(row);meta=getattr(obj,'metadata',None)
            row.update(kernel_name=getattr(meta,'name',None),num_warps=getattr(meta,'num_warps',None),shared_bytes=getattr(meta,'shared',None),registers_per_thread=getattr(obj,'n_regs',None),spill_count_reported_by_loader=getattr(obj,'n_spills',None))
            rows.append(row)
        with path.open('x') as f:json.dump(dict(scope='Actual unchanged Triton compiler API call wall, includes disk-cache load if hit; not solely device compile time. Registers/spills after real loads when supplied; SM occupancy remains unmeasured.',rows=rows),f,indent=2)
    atexit.register(finish)
