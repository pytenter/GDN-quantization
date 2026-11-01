#!/usr/bin/env python3
"""Synthetic bounded archives and fake-host transfers only. No actual network."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'scripts'))
spec=importlib.util.spec_from_file_location('download_fixture',RUN/'scripts/download_verify_small_delivery.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def encoded(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def row(name,data):return dict(path=name,bytes=len(data),sha256=D.digest(data))


def fixture(destination,bad_review=False,bad_inherited=False,canonical_pair_same=False):
    """407 synthetic sources, canonical194 and two equal-byte projection pairs."""
    destination=Path(destination);destination.mkdir(parents=True)
    base={f'candidate/dependency_by_path_sha256/{i:064x}/source.py':f'# canonical synthetic {i}\n'.encode() for i in range(194)}
    if canonical_pair_same:base[f'candidate/dependency_by_path_sha256/{1:064x}/source.py']=base[f'candidate/dependency_by_path_sha256/{0:064x}/source.py']
    base.update({f'candidate/source_tree/stub{i:03}.txt':f'synthetic projection {i}\n'.encode() for i in range(205)})
    base.update({'candidate/source_tree/MANIFEST.json':b'{}\n','candidate/source_tree/manifest.json':b'{}\n','candidate/source_tree/DOCS.json':b'[]\n','candidate/source_tree/docs.json':b'[]\n'})
    assert len(base)==403
    old=dict(base,**{n:b'old synthetic source\n' for n in D.REPLACED});new=dict(base,**{n:b'new observation/resource source\n' for n in D.REPLACED})
    if bad_inherited:new['candidate/source_tree/stub000.txt']=b'not inherited\n'
    parent=encoded(dict(files=[row(n,b) for n,b in sorted(old.items())]));parent_sha=D.digest(parent)
    freeze=encoded(dict(task_id=D.TASK,files=[row(n,b) for n,b in sorted(new.items())]));source_sha=D.digest(freeze)
    data={'payload/'+n:b for n,b in new.items()};data.update({'payload/SOURCE_FREEZE.json':freeze,'payload/provenance/PARENT_EFFECTIVE_SOURCE_FREEZE.json':parent})
    data.update({'FINAL_SCIENTIFIC_REPORT.md':'纯CPU合成报告，不是科学运行。\n'.encode(),'RUNTIME_RESTORATION_DECISION.json':encoded(dict(task_id=D.TASK,status='FAIL_CLOSED_STOP')),'RESOURCE_LEDGER.json':encoded(dict(scope='SYNTHETIC_ONLY')),'analysis/FINAL_SOURCE_DEPENDENCY_VERIFICATION.json':encoded(dict(status='PASS_SYNTHETIC')),'analysis/POST_GPU_RELEASE_ADMISSION.json':encoded(dict(task_id=D.TASK,schema='QWEN_RECOVERY_RELEASE_ADMISSION_V1',status='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED',all_owned_processes_exited=True,all_owned_GPU_resources_released=True)),'archive/ARCHIVE_VERIFICATION.json':encoded(dict(status='PASS_SYNTHETIC_NO_RAW')),'archive/ASSET_CATALOG.json':encoded(dict(files=[]))})
    review_names=['FINAL_SCIENTIFIC_REPORT.md','RUNTIME_RESTORATION_DECISION.json','RESOURCE_LEDGER.json','analysis/FINAL_SOURCE_DEPENDENCY_VERIFICATION.json','analysis/POST_GPU_RELEASE_ADMISSION.json','archive/ARCHIVE_VERIFICATION.json']
    review=dict(status='PASS_SYNTHETIC_REVIEW',findings=[],checks_total=6,checks_passed=6,receipts=[row(n,data[n]) for n in review_names])
    if bad_review:review['receipts'][0]['sha256']='0'*64
    data['analysis/FINAL_INDEPENDENT_RESULT_REVIEW.json']=encoded(review)
    indexed=[row(n,b) for n,b in sorted(data.items())]
    manifest=encoded(dict(task_id=D.TASK,files_count=len(indexed),total_bytes=sum(x['bytes'] for x in indexed),files=indexed));mh=D.digest(manifest)
    sums=(''.join(x['sha256']+'  '+x['path']+'\n' for x in indexed)+mh+'  SHA256_MANIFEST.json\n').encode()
    seal=encoded(dict(task_id=D.TASK,manifest_sha256=mh,SHA256SUMS_sha256=D.digest(sums),source_freeze_sha256=source_sha,source_freeze=dict(path='payload/SOURCE_FREEZE.json',bytes=len(freeze),sha256=source_sha,files_verified=407),FINAL_SCIENTIFIC_REPORT_sha256=D.digest(data['FINAL_SCIENTIFIC_REPORT.md']),report_review_sha256=D.digest(data['analysis/FINAL_INDEPENDENT_RESULT_REVIEW.json']),release_proof_sha256=D.digest(data['analysis/POST_GPU_RELEASE_ADMISSION.json']),archive_verification_sha256=D.digest(data['archive/ARCHIVE_VERIFICATION.json'])))
    data.update({'SHA256_MANIFEST.json':manifest,'SHA256SUMS.txt':sums,'FINAL_SEAL_RECEIPT.json':seal})
    settings=dict(indexed_files=len(indexed),indexed_bytes=sum(x['bytes'] for x in indexed),manifest_sha256=mh,SHA256SUMS_sha256=D.digest(sums),seal_sha256=D.digest(seal),payload_roster_sha256=D.digest(D.canonical([row(n,b) for n,b in sorted(data.items())])))
    for name,b in sorted(data.items()):
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==b
        else:p.write_bytes(b)
    return dict(data=data,settings=settings,source_sha=source_sha,parent_sha=parent_sha)


def bundle(path,data):
    with tarfile.open(path,'w:gz') as tf:
        for name,b in sorted(data.items()):
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600;tf.addfile(info,io.BytesIO(b))


class PayloadTests(unittest.TestCase):
    def test_full407_logical_and_canonical194(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d)
            with patch.object(D,'PARENT_SHA',f['parent_sha']):out=D.verify_delivery(d,f['settings'],f['source_sha'])
            self.assertEqual(out['source407_verified'],407);self.assertEqual(out['canonical_source194_independent_physical_files'],194);self.assertEqual(len(out['case_projection_aliases']),2);self.assertTrue(out['logical_manifest_bytes_equality_only_not_physical_roster_equality'])

    def test_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);(d/'FINAL_SCIENTIFIC_REPORT.md').write_text('changed')
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaises(RuntimeError):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);(d/'RESOURCE_LEDGER.json').unlink()
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaises(RuntimeError):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);(d/'unlisted.txt').write_text('extra')
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaises(RuntimeError):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_unlisted_same_bytes_projection_distinct_inode_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);extra='UNLISTED_CASE_VARIANT.txt';(d/extra).write_bytes(f['data']['FINAL_SCIENTIFIC_REPORT.md']);original=D.projection_key
            def alias(n):return original('FINAL_SCIENTIFIC_REPORT.md') if n==extra else original(n)
            with patch.object(D,'PARENT_SHA',f['parent_sha']),patch.object(D,'projection_key',alias),self.assertRaisesRegex(RuntimeError,'Unindexed distinct physical case variant'):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_review_binding_negative(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d,bad_review=True)
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaisesRegex(RuntimeError,'Final review exact SHA/bytes'):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_inherited403_change_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d,bad_inherited=True)
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaisesRegex(RuntimeError,'Source403\\+4 lineage'):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_canonical_hardlink_collapse_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d,canonical_pair_same=True);names=[n for n in f['data'] if n.startswith('payload/candidate/dependency_by_path_sha256/')]
            (d/names[1]).unlink();os.link(d/names[0],d/names[1])
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaisesRegex(RuntimeError,'Canonical194'):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_non_mac_unindexed_same_inode_projection_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);extra='UNINDEXED_CASE_HARDLINK.txt';os.link(d/'FINAL_SCIENTIFIC_REPORT.md',d/extra);original=D.projection_key
            def alias(n):return original('FINAL_SCIENTIFIC_REPORT.md') if n==extra else original(n)
            with patch.object(D,'PARENT_SHA',f['parent_sha']),patch.object(D,'projection_key',alias),patch.object(D.sys,'platform','linux'),self.assertRaisesRegex(RuntimeError,'Non-Mac unindexed physical'):D.verify_delivery(d,f['settings'],f['source_sha'])

    def test_manifest_anchor_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'delivery';f=fixture(d);(d/'SHA256SUMS.txt').write_text('0  x\n')
            with patch.object(D,'PARENT_SHA',f['parent_sha']),self.assertRaisesRegex(RuntimeError,'Anchor hash'):D.verify_delivery(d,f['settings'],f['source_sha'])


class TarTests(unittest.TestCase):
    def bad_tar(self,root,name,kind=tarfile.REGTYPE,size=1,link=''):
        p=root/'bad.tar.gz'
        with tarfile.open(p,'w:gz') as tf:
            item=tarfile.TarInfo(name);item.type=kind;item.size=size;item.linkname=link
            tf.addfile(item,io.BytesIO(b'x'*size) if size<1024 else None)
        return p

    def test_safe_extract_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);f=fixture(r/'source');b=r/'bundle.tar.gz';bundle(b,f['data'])
            with patch.object(D,'PARENT_SHA',f['parent_sha']):proof=D.extract_safe(b,r/'incoming',f['settings'],f['source_sha'])
            self.assertEqual(proof['source407_verified'],407);D.publish_local_no_clobber(r/'incoming',r/'delivery');self.assertTrue((r/'delivery').is_dir())
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'delivery',f['settings'],f['source_sha'])

    def test_parent_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'../escape.txt')
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'out',{},'')
            self.assertFalse((r/'escape.txt').exists())

    def test_absolute_path(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'/escape.txt')
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'out',{},'')

    def test_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'alias',tarfile.SYMTYPE,0,'elsewhere')
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'out',{},'')

    def test_hardlink(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'alias',tarfile.LNKTYPE,0,'elsewhere')
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'out',{},'')

    def test_directory_member(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'directory',tarfile.DIRTYPE,0)
            with self.assertRaises(RuntimeError):D.extract_safe(b,r/'out',{},'')

    def test_raw_tensor_suffix(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'state.bin')
            with self.assertRaisesRegex(RuntimeError,'Model/raw/native'):D.extract_safe(b,r/'out',{},'')

    def test_oversized_member_header(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'too_large.json',size=D.MAX_FILE+1)
            with self.assertRaisesRegex(RuntimeError,'Tar duplicate/count/size'):D.extract_safe(b,r/'out',{},'')

    def test_compressed_bound_before_extract(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=self.bad_tar(r,'small.txt')
            with patch.object(D,'MAX_BUNDLE',1),self.assertRaisesRegex(RuntimeError,'Compressed bundle'):D.extract_safe(b,r/'out',{},'')
            self.assertFalse((r/'out').exists())

    def test_total_size_bound(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=r/'t.tar.gz';bundle(b,{'a.txt':b'1234567890','b.txt':b'1234567890'})
            with patch.object(D,'MAX_TOTAL',15),self.assertRaisesRegex(RuntimeError,'Tar logical total'):D.extract_safe(b,r/'out',{},'')

    def test_duplicate_members(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=r/'t.tar.gz'
            with tarfile.open(b,'w:gz') as tf:
                for _ in range(2):
                    x=tarfile.TarInfo('same.txt');x.size=1;tf.addfile(x,io.BytesIO(b'x'))
            with self.assertRaisesRegex(RuntimeError,'Tar duplicate'):D.extract_safe(b,r/'out',{},'')

    def test_differing_case_alias_collision(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);b=r/'t.tar.gz';bundle(b,{'A.txt':b'a','a.txt':b'b'})
            with self.assertRaisesRegex(RuntimeError,'Differing-byte projection collision'):D.extract_safe(b,r/'out',{},'')


class FakeTransport:
    calls=[]
    def __init__(self,state,record):self.root=state;self.record=record
    def call55(self,code,args=(),data=None):
        self.calls.append(('55',code))
        if code==D.META_READ:
            path=Path(json.loads(args[0])['path']);b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=D.digest(b),document=json.loads(b))
        if code==D.P.REMOTE_UPLOAD:
            row_=json.loads(args[0]);p=Path(row_['path']);assert len(data)==row_['bytes'] and D.digest(data)==row_['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as out:out.write(data)
            return dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',**row_)
        settings=json.loads(data)
        if code==D.P.REMOTE_RECEIPT_CHECK:
            p=Path(settings['destination'])/D.EXTERNAL;assert D.digest(p.read_bytes())==settings['sha256'];return dict(status='PASS_EXTERNAL_RECEIPT_SHA_BYTES_AND_FINAL_ROSTER',sha256=settings['sha256'],bytes=p.stat().st_size)
        assert code==D.remote_check_program();return D.verify_delivery(Path(settings['destination']),settings['settings'],settings['source_sha256'],settings.get('with_receipt',False))
    def download55(self,description,destination):
        self.calls.append(('download',description['path']));b=Path(description['path']).read_bytes();assert len(b)==description['bytes'] and D.digest(b)==description['sha256'];Path(destination).write_bytes(b);return dict(status='PASS_FAKE_FIXED_BUNDLE_SHA_BYTES',bytes=len(b),sha256=D.digest(b))
    def forward_bundle29(self,path55,path29,description):
        self.calls.append(('forward',path29));p=Path(path29);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as out:out.write(Path(path55).read_bytes())
        return dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',path=path29,bytes=p.stat().st_size,sha256=D.digest(p.read_bytes()))
    def call29(self,code,settings,data=b''):
        self.calls.append(('29',code))
        if code==D.REMOTE_VERIFY64:
            incoming=Path(settings['destination']+'.incoming');D.extract_safe(Path(settings['bundle']),incoming,settings, self.root['source_freeze_sha256']);D.publish_local_no_clobber(incoming,Path(settings['destination']));return dict(status='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES',atomic_no_clobber_publication=True,payload_roster_sha256=settings['payload_roster_sha256'])
        if code==D.P.REMOTE_UPLOAD:
            p=Path(settings['path']);assert len(data)==settings['bytes'] and D.digest(data)==settings['sha256']
            with p.open('xb') as out:out.write(data)
            return dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',**settings)
        if code==D.P.REMOTE_RECEIPT_CHECK:
            p=Path(settings['destination'])/D.EXTERNAL;assert D.digest(p.read_bytes())==settings['sha256'];return dict(status='PASS_EXTERNAL_RECEIPT_SHA_BYTES_AND_FINAL_ROSTER',sha256=settings['sha256'],bytes=p.stat().st_size)
        assert code==D.remote_check_program();return D.verify_delivery(Path(settings['destination']),settings['settings'],settings['source_sha256'],settings.get('with_receipt',False))


class FlowTests(unittest.TestCase):
    def setup_flow(self,t):
        r=Path(t).resolve();local=r/'MacG';remote=r/'remote55G';archive=r/'remote29';local.mkdir();f=fixture(remote/'final_delivery');(remote/'operations').mkdir();b=remote/'SMALL_DELIVERY.tar.gz';bundle(b,f['data'])
        state=dict(task_id=D.TASK,local_root=str(local),remote_root=str(remote),source_freeze_sha256=f['source_sha'],remote_host='202.38.247.55',remote_user='zypan',archive_host='202.38.247.29',archive_root=str(archive),remote_python='/data/ydai/miniconda3/envs/bitdecode/bin/python')
        (local/'ROOT.json').write_bytes(encoded(state));meta=dict(task_id=D.TASK,status='PASS_POST_RELEASE_SMALL_DELIVERY_SEALED_BUNDLE',bundle=dict(path=str(b),bytes=b.stat().st_size,sha256=D.digest(b.read_bytes())),settings=f['settings'],destination=str(remote/'final_delivery'),scientific_status='FAIL_CLOSED_STOP')
        (remote/'operations/SMALL_DELIVERY_BUNDLE.json').write_bytes(encoded(meta));(remote/'operations/SMALL_DELIVERY_LOCAL_VERIFICATION.json').write_bytes(encoded(dict(status='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES',settings=f['settings'])))
        pin=D.digest((remote/'operations/SMALL_DELIVERY_BUNDLE.json').read_bytes());return local,remote,archive,f,pin

    def test_fake_three_host_flow_and_replication(self):
        with tempfile.TemporaryDirectory() as t:
            local,remote,archive,f,pin=self.setup_flow(t);FakeTransport.calls=[]
            with patch.object(D,'PARENT_SHA',f['parent_sha']):result=D.run(local,pin,FakeTransport)
            self.assertEqual(result['status'],'PASS_VERIFIED_POST_RELEASE_SCIENTIFIC_SMALL_DELIVERY_THREE_HOSTS')
            b=(local/'delivery'/D.EXTERNAL).read_bytes();self.assertEqual(b,(remote/'final_delivery'/D.EXTERNAL).read_bytes());self.assertEqual(b,(archive/'SMALL_DELIVERY'/D.EXTERNAL).read_bytes());self.assertTrue((local/'operations/EXTERNAL_RECEIPT_REPLICATION.json').is_file())
            proof=json.loads(b);self.assertTrue(proof['all_three_hosts_logical_manifest_SHA_bytes_equal']);self.assertTrue(proof['physical_roster_identical_not_claimed']);self.assertGreater(len(FakeTransport.calls),5)

    def test_pin_mismatch_retains_failure_never_retries(self):
        with tempfile.TemporaryDirectory() as t:
            local,remote,archive,f,pin=self.setup_flow(t)
            with self.assertRaisesRegex(RuntimeError,'sealed descriptor SHA'):D.run(local,'0'*64,FakeTransport)
            self.assertFalse((local/'delivery').exists());self.assertIn('FAILED_RETAIN_EVIDENCE_NO_AUTOMATIC_RETRY',(local/'operations/SMALL_DELIVERY_DOWNLOAD_STATE.jsonl').read_text())
            with self.assertRaisesRegex(RuntimeError,'One-shot download state'):D.run(local,pin,FakeTransport)

    def test_fixed_bundle_path_gate(self):
        state=dict(remote_root='/G');doc=dict(task_id=D.TASK,status='PASS_POST_RELEASE_SMALL_DELIVERY_SEALED_BUNDLE',destination='/G/final_delivery',bundle=dict(path='/other/SMALL_DELIVERY.tar.gz',bytes=1,sha256='a'*64),settings={})
        with self.assertRaisesRegex(RuntimeError,'Fixed bundle path'):D.descriptor_admission(state,dict(sha256='a'*64,document=doc),dict(document={}), 'a'*64)

    def test_not_sealed_gate(self):
        with self.assertRaisesRegex(RuntimeError,'sealed bundle status'):D.descriptor_admission(dict(remote_root='/G'),dict(sha256='a'*64,document=dict(status='DRAFT')),dict(document={}), 'a'*64)

    def test_remote_code_limits_and_explicit_python(self):
        self.assertIn('member.size<=64*1024**2',D.REMOTE_VERIFY64);self.assertIn('total<=256*1024**2',D.REMOTE_VERIFY64);self.assertEqual(D.digest(Path(D.P.__file__).read_bytes()),D.PRIMITIVES_SHA)
        self.assertIn('/usr/bin/python3',D.inspect.getsource(D.Transport.call29));self.assertIn('/usr/bin/python3',D.inspect.getsource(D.Transport.forward_bundle29))
        compile(D.remote_check_program(),'<synthetic remote verifier>','exec')

    def test_default_preview_no_network_or_writes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);result=subprocess.run([sys.executable,'-B',str(RUN/'scripts/download_verify_small_delivery.py'),'--root',str(root)],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['status'],'PREVIEW_ONLY_NO_NETWORK_NO_WRITES_NO_GPU');self.assertEqual(list(root.iterdir()),[])


class StreamDownloadTests(unittest.TestCase):
    def stream(self,body):
        read_fd,write_fd=os.pipe();os.write(write_fd,body);os.close(write_fd)
        class FakeProcess:
            stdout=os.fdopen(read_fd,'rb');stderr=io.BytesIO(b'')
            def wait(self,timeout=None):return 0
            def poll(self):return 0
            def kill(self):raise AssertionError('Already exited fake child')
        return FakeProcess()

    def transport(self):
        state=dict(ssh_control_path='/NO_NETWORK.sock',remote_user='fixture',remote_host='invalid.example',remote_python='/NO_EXECUTABLE')
        return D.Transport(state,lambda *a,**k:None)

    def test_exact_stream_sha_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            body=b'bounded fake compressed bytes';p=self.stream(body);dest=Path(t)/'bundle.tar.gz';desc=dict(path='/fixed/SMALL_DELIVERY.tar.gz',bytes=len(body),sha256=D.digest(body))
            try:
                with patch.object(D.subprocess,'Popen',return_value=p) as create:receipt=self.transport().download55(desc,dest)
                self.assertEqual(dest.read_bytes(),body);self.assertEqual(receipt['sha256'],desc['sha256']);self.assertEqual(create.call_count,1)
            finally:p.stdout.close()

    def test_wrong_stream_sha_retains_incoming(self):
        with tempfile.TemporaryDirectory() as t:
            body=b'bounded fake bytes';p=self.stream(body);dest=Path(t)/'bundle.tar.gz';desc=dict(path='/fixed/SMALL_DELIVERY.tar.gz',bytes=len(body),sha256='0'*64)
            try:
                with patch.object(D.subprocess,'Popen',return_value=p),self.assertRaisesRegex(RuntimeError,'exact SHA/bytes'):self.transport().download55(desc,dest)
                self.assertFalse(dest.exists());self.assertTrue(Path(str(dest)+'.incoming').exists())
            finally:p.stdout.close()

    def test_stream_overshoot_bounded(self):
        with tempfile.TemporaryDirectory() as t:
            p=self.stream(b'too many bytes');dest=Path(t)/'bundle.tar.gz';desc=dict(path='/fixed/SMALL_DELIVERY.tar.gz',bytes=3,sha256='0'*64)
            try:
                with patch.object(D.subprocess,'Popen',return_value=p),self.assertRaisesRegex(RuntimeError,'exceeded exact compressed bound'):self.transport().download55(desc,dest)
                self.assertFalse(dest.exists())
            finally:p.stdout.close()

    def test_existing_download_no_dispatch(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'bundle.tar.gz';dest.write_bytes(b'existing')
            with patch.object(D.subprocess,'Popen') as create,self.assertRaisesRegex(RuntimeError,'Download path already exists'):self.transport().download55(dict(bytes=1,path='/fixed',sha256='0'*64),dest)
            create.assert_not_called();self.assertEqual(dest.read_bytes(),b'existing')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);result=unittest.TextTestRunner(verbosity=2).run(suite);source=RUN/'scripts/download_verify_small_delivery.py'
    receipt=dict(status='PASS_CPU_FAKE_SMALL_DELIVERY_TESTS' if result.wasSuccessful() else 'FAIL_CPU_FAKE_SMALL_DELIVERY_FIXTURE',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),source_sha256=D.digest(source.read_bytes()),test_sha256=D.digest(Path(__file__).read_bytes()),copied_primitive_sha256=D.digest(Path(D.P.__file__).read_bytes()),scope='Synthetic407/403+4/canonical194, bounded safe extraction, fake local three-host flow and receipt replication only. No actual network, remote publication, model/Torch, Driver/NVML/GPU or old-seal mutation.',actual_network_calls=0,GPU_queries=0,GPU_dispatches=0)
    target=RUN/'tests/SMALL_DELIVERY_DOWNLOADER_CPU_TEST_RESULTS.json'
    if target.exists():
        with (target.parent/'SMALL_DELIVERY_DOWNLOADER_CPU_TEST_HISTORY.jsonl').open('a') as stream:stream.write(target.read_text().strip()+'\n')
    target.write_text(json.dumps(receipt,indent=2)+'\n');sys.exit(0 if result.wasSuccessful() else 1)
