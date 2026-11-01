"""Synthetic architecture tests only. No model/Torch/scientific measurements."""
import json
import os
from pathlib import Path
import subprocess
import sys
from capsule_v2 import Inventory, Escape, sha, write

P = Path(__file__).resolve().parent


def main():
    out=P/'semantic_tests'; out.mkdir(exist_ok=False)
    fixtures=out/'fixtures'; fixtures.mkdir()
    metadata=fixtures/'arbitrary_support_data.txt'
    metadata.write_text('# synthetic alias data\nenglish en_US.ISO-8859-1\n')
    pol=dict(roots=[dict(path=str(P),classification='SYNTHETIC_ENGINEERING_CODE')],
        exact=[],write_roots=[str(out)],advisory_exact=[],advisory_prefixes=[])
    # Do not authorize fixture roots. Only the closed engineering files root;
    # use unrelated external fixture directory via a preserved test tempdir.
    import tempfile
    external=Path(tempfile.mkdtemp(prefix='ling_semantic_fixture_'))
    meta=external/'support.txt'; meta.write_bytes(metadata.read_bytes())
    ctx=dict(synthetic_only=True,consumer_role='SYNTHETIC_LOCALE_RESOLUTION',read_only=True)
    results={}
    inv=Inventory(pol,out/'ambient')
    inv.observe(str(meta),'external_read',context=ctx)
    report=inv.finish('PASS')
    r=inv.ambient[str(meta)]
    assert report['gate']=='PASS' and not inv.rows
    assert r['sha256']==sha(meta) and r['read_only_ingress'] and r['reason']
    assert r['classification']=='C_AMBIENT_OS_METADATA_CONFIG'
    results['SEMANTIC_TEST_1']='PASS_AMBIENT_RECORDED_NONBLOCKING_NO_PATH_EXCEPTION'

    def must_fail(name, callback, needle):
        try:
            callback(); raise AssertionError('accepted '+name)
        except Escape as exc:
            assert needle in str(exc), (name,str(exc))

    inv2=Inventory(pol,out/'escalation')
    inv2.observe(str(meta),'external_read',context=ctx)
    must_fail('import escalation',lambda:inv2.observe(str(meta),'python_import'), 'ESCALATION')
    assert inv2.transitions[-1]['after']=='A_EXECUTABLE_CODE_BEARING'
    inv2.finish('FAIL','EXPECTED_ESCALATION')
    results['SEMANTIC_TEST_2']='PASS_SAME_FILE_CODE_USE_UPGRADES_AND_BLOCKS'
    for i,(name,phase,code,scientific) in enumerate([
        ('unknown.py','python_import',True,False),
        ('unknown.so','executable_mapping',True,False),
        ('unknown_model.bin','model_checkpoint_read',False,True),
        ('unknown_document.txt','document_read',False,True)],3):
        p=external/name;p.write_bytes(b'unknown fixture\n')
        x=Inventory(pol,out/('unknown_'+str(i)))
        must_fail(name,lambda:x.observe(str(p),phase,executable=code,scientific=scientific),'UNCLASSIFIED_EXTERNAL')
        assert x.rejections[-1]['classification']==('A_EXECUTABLE_CODE_BEARING' if code else 'B_SCIENTIFIC_NUMERICAL_INPUT')
        x.finish('FAIL','EXPECTED_UNKNOWN')
        results['SEMANTIC_TEST_'+str(i)]='PASS_UNKNOWN_CODE_OR_SCIENTIFIC_INPUT_BLOCKED'
    # Real execvp PATH probes + real successful PATH executable, independent
    # ptrace. Positive A capture then immutable B enforcement, no auto append.
    path_pol=dict(pol,roots=pol['roots']+[
        dict(path='/data/zypan/envs/ling-kda',classification='PYTHON_RUNTIME'),
        dict(path='/usr/lib',classification='NATIVE_RUNTIME'),
        dict(path='/usr/bin',classification='EXECUTABLE_RUNTIME')],
        exact=[dict(path='/etc/ld.so.cache',classification='DYNAMIC_LOADER_CONFIG')])
    policy=out/'path_policy.json';write(policy,path_pol)
    code='import os; os.execvp("true",["true"])'
    def run(name,frozen=None):
        dest=out/name
        args=[sys.executable,str(P/'syscall_supervisor_v2.py'),'--policy',str(policy),
            '--output',str(dest),'--cwd',os.getcwd()]
        if frozen:args+=['--frozen',str(frozen)]
        args+=['--',sys.executable,'-B','-c',code]
        env=dict(os.environ,PATH='/nonexistent/semantic_probe:/usr/bin',PYTHONDONTWRITEBYTECODE='1')
        with (out/(name+'.log')).open('xb') as f:
            r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
        report=json.loads((dest/'runtime_escape_report.json').read_text())
        assert r.returncode==0 and report['gate']=='PASS',report['error']
        return report
    a=run('PATH_A'); frozen=out/'path_frozen.json';write(frozen,dict(artifacts=a['artifacts']))
    before=sha(frozen); b=run('PATH_B',frozen)
    assert any(r['path']=='/usr/bin/true' for r in b['artifacts'])
    assert sha(frozen)==before
    results['SEMANTIC_TEST_7']='PASS_REAL_PATH_EXECUTABLE_FROZEN_CLOSED'
    assert not any('/nonexistent/' in r['path'] for r in a['artifacts'])
    results['SEMANTIC_TEST_8']='PASS_ABSENT_EXECVP_PROBE_NO_SUCCESSFUL_INGRESS'
    p=external/'ordinary.txt';p.write_text('plain external text without ambient semantics\n')
    x=Inventory(pol,out/'unknown_text')
    must_fail('unknown text',lambda:x.observe(str(p),'external_read',context=ctx),'UNCLASSIFIED_EXTERNAL')
    x.finish('FAIL','EXPECTED_UNKNOWN_TEXT')
    results['SEMANTIC_TEST_9']='PASS_UNKNOWN_REGULAR_TEXT_BLOCKED'
    x=Inventory(pol,out/'ambient_scientific')
    x.observe(str(meta),'external_read',context=ctx)
    must_fail('scientific escalation',lambda:x.observe(str(meta),'model_read',scientific=True),'ESCALATION')
    x.finish('FAIL','EXPECTED_SCIENTIFIC_ESCALATION')
    results['SEMANTIC_TEST_10']='PASS_AMBIENT_SCIENTIFIC_UPGRADE_BLOCKED'
    x=Inventory(pol,out/'ambient_mutation');x.observe(str(meta),'external_read',context=ctx)
    must_fail('ambient mutation',lambda:x.check_write(str(meta)),'MUTATION')
    x.finish('FAIL','EXPECTED_MUTATION')
    results['SEMANTIC_TEST_11']='PASS_AMBIENT_WRITE_FORBIDDEN'
    # Exercise the independent Python-origin channel, including direct compile
    # of ambient file bytes with a synthetic <string> filename.
    compile_metadata=external/'compile_support.txt'
    compile_metadata.write_text('# synthetic OS alias fixture\nassert en_US.ISO88591\n')
    audit_policy=out/'audit_policy.json';write(audit_policy,dict(path_pol,synthetic_ambient_test_only=True))
    audit_fixture=out/'audit_fixture.py'
    audit_fixture.write_text('import importlib.util\nfrom pathlib import Path\nimport sys\n'
        'p=Path(__file__).parents[1]/"python_origin_audit.py"\n'
        's=importlib.util.spec_from_file_location("audit",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);ch=m.install()\n'
        'if sys.argv[1]=="ok":\n'
        ' import collections\n'
        ' T=collections.namedtuple("Synthetic","x");assert eval("lambda z:z+1")(T(1).x)==2\n'
        'else:\n'
        ' raw=open(%r,"rb").read();compile(raw,"<string>","exec")\n' % str(compile_metadata))
    for mode,expected in [('ok','PASS'),('ambient_compile','FAIL')]:
        dest=out/('audit_'+mode)
        args=[sys.executable,str(P/'syscall_supervisor_v2.py'),'--policy',str(audit_policy),
            '--output',str(dest),'--cwd',os.getcwd(),'--audit-channel','--',sys.executable,'-B',str(audit_fixture),mode]
        with (out/('audit_'+mode+'.log')).open('xb') as f:
            result=subprocess.run(args,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
        report=json.loads((dest/'runtime_escape_report.json').read_text())
        assert report['gate']==expected,report['error']
        if mode=='ambient_compile':
            assert 'AMBIENT_ROLE_ESCALATION' in report['error'],report['error']
            sem=json.loads((dest/'semantic_ingress_report.json').read_text())
            assert sem['classification_transitions'][-1]['after']=='A_EXECUTABLE_CODE_BEARING'
        else:
            assert result.returncode==0
    results['SEMANTIC_TEST_12']='PASS_REAL_PYTHON_AUDIT_ORIGIN_CHANNEL'
    results['SEMANTIC_TEST_13']='PASS_REAL_AMBIENT_BYTES_COMPILE_UPGRADE_BLOCKED'
    write(out/'results.json',dict(gate='PASS',tests=results,synthetic_only=True,
        external_fixture_directory=str(external),formal_measurements=0))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
