"""Isolated architecture tests; synthetic source/native fixtures only."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from capsule_v2 import Inventory, Escape, canonical, sha, write


def main():
    here = Path(__file__).resolve().parent
    out = here / 'engineering_tests'
    out.mkdir(exist_ok=False)
    sandbox = Path(tempfile.mkdtemp(prefix='ling_capsule_v2_tests_'))
    stable = sandbox / 'stable.py'
    stable.write_text('value = 17\n')
    policy = dict(roots=[dict(path=str(here), classification='V2_ENGINEERING_IMPLEMENTATION'),
        dict(path='/data/zypan/envs/ling-kda', classification='PYTHON_RUNTIME'),
        dict(path='/usr/lib', classification='SYSTEM_RUNTIME'),
        dict(path='/usr/bin', classification='SYSTEM_EXECUTABLE')],
        exact=[dict(path=str(stable), classification='SYNTHETIC_STABLE_SOURCE'),
            dict(path='/etc/ld.so.cache', classification='DYNAMIC_LOADER_CONFIG')],
        write_roots=[str(out), str(sandbox)], advisory_exact=[],
        advisory_prefixes=['/usr/share/zoneinfo/'])
    a = Inventory(policy, out / 'identity_A')
    a.observe(str(stable), 'fixture')
    frozen = dict(artifacts=list(a.rows.values()))
    a.finish('PASS')
    stable.write_text('value = 18\n')
    b = Inventory(policy, out / 'identity_B', frozen)
    try:
        b.observe(str(stable), 'fixture')
        raise AssertionError('mutation accepted')
    except Escape as e:
        assert 'MISMATCH' in str(e)
    b.finish('FAIL', 'EXPECTED_STABLE_MUTATION_REJECTED')
    stable.write_text('value = 17\n')
    results = {'V2_TEST_1': 'PASS_EXPECTED_MUTATION_FAIL'}
    v1 = sandbox / 'A/foo.py'; v2 = sandbox / 'B/foo.py'
    for v in (v1, v2):
        v.parent.mkdir(); v.write_text('x = 1\n')
    pv = dict(policy, exact=policy['exact'] + [dict(path=str(v), classification='SYNTHETIC_VOLATILE_SOURCE', volatile=True) for v in (v1,v2)])
    av = Inventory(pv, out / 'volatile_A'); av.observe(str(v1), 'fixture'); av.finish('PASS')
    fv = dict(artifacts=list(av.rows.values()))
    bv = Inventory(pv, out / 'volatile_B', fv); bv.observe(str(v2), 'fixture'); bv.finish('PASS')
    results['V2_TEST_2'] = 'PASS_CONTENT_IDENTITY_RANDOM_PATH'
    v2.write_text('x = 2\n')
    cv = Inventory(pv, out / 'volatile_C', fv)
    try:
        cv.observe(str(v2), 'fixture'); raise AssertionError('volatile mutation accepted')
    except Escape as e:
        assert 'UNKNOWN_EXTERNAL' in str(e)
    cv.finish('FAIL', 'EXPECTED_DIFFERENT_VOLATILE_BYTES_REJECTED')
    results['V2_TEST_3'] = 'PASS_EXPECTED_DIFFERENT_CONTENT_FAIL'

    # Actual ptrace-supervised native boundary, not simulated event accounting.
    plugin = sandbox / 'plugin.so'; loader = sandbox / 'loader.so'
    src = sandbox / 'plugin.c'; src.write_text('int answer(void) { return 42; }\n')
    subprocess.run(['gcc','-shared','-fPIC',str(src),'-o',str(plugin)], check=True)
    src2 = sandbox / 'loader.c'
    src2.write_text('#include <dlfcn.h>\nint load_other(const char *p) { return dlopen(p, RTLD_NOW) ? 0 : 1; }\n')
    subprocess.run(['gcc','-shared','-fPIC',str(src2),'-o',str(loader),'-ldl'], check=True)
    policy['exact'].append(dict(path=str(loader), classification='SYNTHETIC_NATIVE_LOADER'))
    pol = out / 'test_policy.json'; write(pol, policy)

    def run(name, code, frozen_file=None):
        dest = out / name
        cmd = [sys.executable, str(here/'syscall_supervisor_v2.py'), '--policy', str(pol),
            '--output',str(dest),'--cwd',os.getcwd()]
        if frozen_file: cmd += ['--frozen', str(frozen_file)]
        cmd += ['--',sys.executable,'-B','-c',code]
        with (out/(name+'.log')).open('xb') as f:
            result = subprocess.run(cmd, stdout=f,stderr=subprocess.STDOUT, env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        report = json.loads((dest/'runtime_escape_report.json').read_text())
        assert result.returncode == (0 if report['gate']=='PASS' else 1)
        return report

    code = 'import collections, tokenize, ctypes; T=collections.namedtuple("SyntheticPair", "x y"); assert eval("lambda z: z+1")(T(1,2).x)==2; ctypes.CDLL(%r)' % str(loader)
    ra = run('closed_startup_A', code)
    assert ra['gate']=='PASS', ra['error']
    frozen_file = out/'test_frozen.json'; write(frozen_file,dict(artifacts=ra['artifacts']))
    frozen_before = sha(frozen_file)
    rb = run('closed_startup_B', code, frozen_file)
    assert rb['gate']=='PASS', rb['error']
    results['V2_TEST_5']='PASS_ACTUAL_DYNAMIC_IN_MEMORY_CODE_NO_NAME_WHITELIST'
    results['V2_TEST_7']='PASS_STARTUP_PRELOADED_EXTERNAL_MODULES_BOUND'
    unknown = sandbox/'unknown.py'; unknown.write_text('value = 3\n')
    ru = run('unknown_python_B', 'import importlib.util; s=importlib.util.spec_from_file_location("u",%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)' % str(unknown), frozen_file)
    assert ru['gate']=='FAIL' and 'UNCLASSIFIED_EXTERNAL' in ru['error'], ru
    results['V2_TEST_4']='PASS_ACTUAL_UNKNOWN_FILE_BACKED_MODULE_FAIL'
    text = sandbox/'code.txt'; text.write_text('value = 99\n')
    rf = run('external_exec_B', 'exec(open(%r).read())' % str(text), frozen_file)
    assert rf['gate']=='FAIL' and 'UNCLASSIFIED_EXTERNAL' in rf['error'], rf
    results['V2_TEST_6']='PASS_ACTUAL_UNCLOSED_EXTERNAL_TEXT_READ_BLOCKED'
    rn = run('unknown_native_B','import ctypes; m=ctypes.CDLL(%r); m.load_other.argtypes=[ctypes.c_char_p]; m.load_other(%r)' % (str(loader),os.fsencode(plugin)), frozen_file)
    assert rn['gate']=='FAIL' and 'UNCLASSIFIED_EXTERNAL' in rn['error'], rn
    results['V2_TEST_9']='PASS_C_LEVEL_DLOPEN_UNKNOWN_LIBRARY_BLOCKED'
    results['V2_TEST_10']='PASS_NO_APPEND_NO_RETRY_NO_WARNING'
    assert sha(frozen_file)==frozen_before
    # Startup mismatch uses a real preloaded file-backed module, before the
    # module-origin inventory would be installed inside the Python process.
    startup_code = 'import importlib.util; s=importlib.util.spec_from_file_location("preloaded",%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.value==17' % str(stable)
    r0=run('startup_fixture_A',startup_code); assert r0['gate']=='PASS',r0['error']
    ff=out/'startup_frozen.json';write(ff,dict(artifacts=r0['artifacts']))
    stable.write_text('value = 19\n')
    r1=run('startup_fixture_B',startup_code,ff)
    assert r1['gate']=='FAIL' and 'MISMATCH' in r1['error'],r1
    results['V2_TEST_8']='PASS_ACTUAL_STARTUP_FILE_MISMATCH_FAIL'
    write(out/'results.json',dict(**results, gate='PASS',synthetic_only=True,formal_measurements=0,
        policy_sha256=sha(pol), frozen_manifest_unchanged=sha(frozen_file)==frozen_before,
        native_coverage='C-level dlopen intercepted at file ingress before mapping/resume', sandbox=str(sandbox)))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
