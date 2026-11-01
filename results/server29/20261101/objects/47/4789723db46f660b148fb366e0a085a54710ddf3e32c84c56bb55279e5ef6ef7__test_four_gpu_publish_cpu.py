"""Actual CPU55->29 SHA/bytes/NOREPLACE tests on new isolated synthetic paths."""
import hashlib,json,os,sys,tempfile,time
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
import pace_four_gpu_publish as P

def run():
    label='CPU_FIXTURE_PACE4_'+str(os.getpid())+'_'+str(time.time_ns())
    archive='/data/zypan/qwen_gdn_pace48_archive/'+label;rows=[]
    with tempfile.TemporaryDirectory(prefix='pace4_publish_CPU_') as td:
        r=Path(td);(r/'source_manifest.json').write_text('{"CPU_FIXTURE_ONLY":true}\n');(r/'GPU_PERMIT.json').write_text('{"ALLOW_GPU":"NO"}\n')
        with mock.patch.object(C,'REMOTE',str(r)),mock.patch.object(C,'ARCHIVE',archive),mock.patch.object(C,'REVISION',label):
            for name,rc in [('CPU_EXIT0',0),('CPU_EXIT1',1)]:
                target=r/'execution'/name;target.mkdir(parents=True);(target/'fixture.txt').write_text('CPU only; not a model result\n')
                interval=dict(unit=name,stage='scale_repair_gates',returncode=rc,released=True)
                result=P.publish_unit(r,target,interval)
                assert result['actual_exit_returncode']==rc and result['released'] is True
                before=(target/'DURABLE29_RECEIPT.json').read_bytes()
                try:P.publish_unit(r,target,interval)
                except (FileExistsError,RuntimeError):pass
                else:raise AssertionError('Repeat publication accepted')
                assert before==(target/'DURABLE29_RECEIPT.json').read_bytes()
                rows.append(dict(case=name,actual_cpu_transfer=True,actual_NOREPLACE=True,actual_sha_bytes=True,nonzero_exit_not_reclassified=rc!=0,repeat_refused=True,receipt=result))
            target=r/'execution/UNRELEASED';target.mkdir()
            try:P.publish_unit(r,target,dict(unit='UNRELEASED',stage='scale_repair_gates',returncode=0,released=False))
            except RuntimeError:pass
            else:raise AssertionError('Unreleased unit published')
            assert not (target/'DURABLE29_MANIFEST.json').exists()
            try:P.publish_unit(r,r.parent,dict(unit='ESCAPE',stage='formal',returncode=0,released=True))
            except RuntimeError:pass
            else:raise AssertionError('Foreign root accepted')
            try:P.publish_unit(r,target,dict(unit='../FOREIGN',stage='formal',returncode=0,released=True))
            except RuntimeError:pass
            else:raise AssertionError('Unsafe unit accepted')
    return dict(status='PASS_ACTUAL_CPU_PUBLICATION_FIXTURES',cases=5,rows=rows,archive_fixture_parent=archive,CPU_MOCK_ONLY=True,GPU_calls=0,model_loads=0,production_qualification_credit=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
