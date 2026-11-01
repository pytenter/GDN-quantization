#!/usr/bin/env python3
"""Actual guard body, isolated CPU filesystem/mapping fixtures; no CUDA/model.

All negative files/symlinks and synthetic ELF-like bytes stay in a temporary
directory outside the scientific root. These tests prove guard mechanics, not
ELF provenance or real GPU execution; real two-fresh CPU/ELF closure probes and
the authorized one-time load-only engineering probe are separate evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);source=args.root.resolve()
    spec=importlib.util.spec_from_file_location('_actual_native_closure_guard_cpu',source/'src/qualification_guards.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,OSError,ValueError):checks.append(name)
        else:raise AssertionError('Expected guard rejection: '+name)
    def put(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()+b'\n')
    def row(path,provenance=False):
        path=path.resolve();s=path.stat()
        value=dict(path=str(path),bytes=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,sha256=guard.sha(path))
        if provenance:value['provenance']=dict(CPU_SYNTHETIC_FIXTURE_NOT_REAL_ELF_PACKAGE=True)
        return value
    def mapped(path,*,inode=None,device=None,deleted=False,permissions='r-xp',escaped=False):
        s=path.stat();name=str(path)
        if escaped:name=name.replace(' ','\\040')
        return '100000-101000 %s 00000000 %s %d %s%s\n'%(permissions,device or '%x:%x'%(os.major(s.st_dev),os.minor(s.st_dev)),s.st_ino if inode is None else inode,name,' (deleted)'if deleted else '')
    with tempfile.TemporaryDirectory(prefix='resident_native_guard_CPU_ONLY_')as temporary:
        parent=Path(temporary).resolve();root=parent/'independent_science_fixture';root.mkdir()
        libraries=parent/'native providers';libraries.mkdir()
        base=libraries/'base.so';base.write_bytes(b'\x7fELF'+b'BASE_TEST_ONLY'*10)
        lazy=libraries/'lazy normal provider.so';lazy.write_bytes(b'\x7fELF'+b'LAZY_TEST_ONLY'*10)
        unknown=libraries/'another_normal_name.so';unknown.write_bytes(b'\x7fELF'+b'UNKNOWN_TEST_ONLY'*10)
        baseline=[{key:value for key,value in row(base).items()if key in('path','bytes','sha256')}]
        for index in range(60):
            p=parent/('baseline_%02d.txt'%index);p.write_bytes(('ORIGINAL61_CPU_FIXTURE_%d'%index).encode())
            baseline.append(dict(path=str(p),bytes=p.stat().st_size,sha256=guard.sha(p)))
        put(root/'analysis/EXTERNAL_BINDINGS.json',dict(files=baseline));base_contract=(root/'analysis/EXTERNAL_BINDINGS.json').read_bytes()
        model=[]
        for index in range(17):
            p=parent/('model_stat_fixture_%02d'%index);p.write_bytes(b'CPU_STAT_ONLY_NO_MODEL')
            model.append(row(p))
        put(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json',dict(files=model))
        probes=[]
        for index in range(2):
            p=root/('analysis/CPU_NATIVE_PROBE_%d.json'%index)
            put(p,dict(status='CPU_TEST_ONLY',pid=100+index,CUDA_initialized=False))
            probes.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=guard.sha(p)))
        manifest=dict(schema_version=1,status=guard.NATIVE_CLOSURE_STATUS,base_external_bindings_sha256=guard.sha(root/'analysis/EXTERNAL_BINDINGS.json'),
                      files=[row(base,True),row(lazy,True)],aliases=[],cpu_probe_receipts=probes)
        put(root/guard.NATIVE_CLOSURE,manifest)
        loaded=guard.load_native_closure(root)
        check('exact_separate_two_probe_manifest_accepted',loaded==manifest)
        check('original61_contract_not_rewritten',base_contract==(root/'analysis/EXTERNAL_BINDINGS.json').read_bytes())
        closure=guard.verify_native_closure(root,fresh=True)
        check('all_closure_files_fresh_stat_SHA_verified',closure['count']==2 and closure['full_fresh_SHA']is True)
        original_sha=guard.sha;calls=[]
        def count_sha(path):calls.append(str(Path(path).resolve()));return original_sha(path)
        with patch.object(guard,'sha',side_effect=count_sha):
            returned=guard.verify_model_and_external(root,fresh_external=True)
        check('original61_return_shape_retained',returned==baseline)
        check('dedup_original_native_hash_in_extra_closure',calls.count(str(base.resolve()))==1 and calls.count(str(lazy.resolve()))==1)
        maps=mapped(base)+mapped(lazy,escaped=True)+'200000-201000 r-xp 00000000 00:00 0 [vdso]\n'
        actual=guard.assess_native_maps(root,maps)
        check('base_plus_extended_native_exact_union_pass',actual['status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED' and len(actual['mapped_native'])==2)
        check('space_escaped_fullpathname_parsed',any(item['path']==str(lazy)for item in actual['mapped_native']))
        check('anonymous_kernel_executable_scope_recorded',len(actual['anonymous_or_kernel_executable_mappings'])==1)
        observed=guard.capture_native_map_evidence(mapped(unknown))
        check('load_only_observation_never_admission_PASS',observed['status']=='OBSERVATION_ONLY_NO_NATIVE_ADMISSION'and observed['exact_allowlist_enforced']is False)
        check('load_only_observation_keeps_actual_unknown_SHA',observed['mapped_native'][0]['sha256']==guard.sha(unknown))
        bad=guard.assess_native_maps(root,mapped(unknown))
        check('same_provider_directory_never_blanket_allow',bad['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        check('unknown_actual_fullpath_lstat_inode_SHA_preserved',bad['mapped_native'][0]['path']==str(unknown)and bad['mapped_native'][0]['lstat']['inode']==unknown.stat().st_ino and bad['mapped_native'][0]['sha256']==guard.sha(unknown))
        quiet=libraries/'arbitrary_name.bin';quiet.write_bytes(b'\x7fELFNOT_NAMED_CUDA_OR_TORCH')
        check('unknown_executable_without_keyword_rejected',guard.assess_native_maps(root,mapped(quiet))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        alias=libraries/'alias.so';alias.symlink_to(lazy)
        check('exact_known_inode_canonical_alias_passes',guard.assess_native_maps(root,mapped(alias))['status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED')
        alias.unlink();alias.symlink_to(unknown)
        check('alias_to_foreign_canonical_file_rejected',guard.assess_native_maps(root,mapped(alias))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        check('mapped_inode_change_rejected',guard.assess_native_maps(root,mapped(base,inode=base.stat().st_ino+1))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        check('mapped_device_change_rejected',guard.assess_native_maps(root,mapped(base,device='ff:ff'))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        check('deleted_mapping_rejected_even_if_path_recreated',guard.assess_native_maps(root,mapped(base,deleted=True))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        check('any_bad_segment_rejects_duplicate_file_mapping',guard.assess_native_maps(root,mapped(base)+mapped(base,inode=base.stat().st_ino+1))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        missing=parent/'missing.so'
        missing_text='100000-101000 r-xp 00000000 00:00 123 '+str(missing)+' (deleted)\n'
        gone=guard.assess_native_maps(root,missing_text)
        check('missing_deleted_fullpath_error_retained',gone['status']=='FAIL_NATIVE_MAPPING_IDENTITY'and gone['mapped_native'][0]['path']==str(missing)and 'error'in gone['mapped_native'][0])
        data=parent/'model_tensor.bin';data.write_bytes(b'READ_ONLY_DATA_NOT_A_NATIVE_LIBRARY')
        data_maps=mapped(data,permissions='r--p')+'300000-301000 rw-s 00000000 00:00 1 /dev/nvidia0\n'
        check('nonexecutable_model_and_GPU_device_mappings_not_read_as_libraries',not guard.assess_native_maps(root,data_maps)['mapped_native'])
        text=libraries/'text.so';text.write_bytes(b'NOT_ELF')
        check('executable_or_so_nonELF_is_not_legitimized',guard.assess_native_maps(root,mapped(text))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        original=lazy.read_bytes();s=lazy.stat();changed=b'\x7fELF'+b'X'*(len(original)-4);lazy.write_bytes(changed);os.utime(lazy,ns=(s.st_atime_ns,s.st_mtime_ns))
        reject('closure_fresh_SHA_rejects_equal_stat_byte_tamper',lambda:guard.verify_native_closure(root,fresh=True))
        check('mapped_fresh_SHA_rejects_equal_stat_byte_tamper',guard.assess_native_maps(root,mapped(lazy))['status']=='FAIL_NATIVE_MAPPING_IDENTITY')
        lazy.write_bytes(original);os.utime(lazy,ns=(s.st_atime_ns,s.st_mtime_ns))
        def manifest_negative(name,mutation):
            value=copy.deepcopy(manifest);mutation(value);put(root/guard.NATIVE_CLOSURE,value)
            try:reject(name,lambda:guard.load_native_closure(root))
            finally:put(root/guard.NATIVE_CLOSURE,manifest)
        manifest_negative('wrong_base61_binding_rejected',lambda v:v.__setitem__('base_external_bindings_sha256','0'*64))
        manifest_negative('single_CPU_probe_rejected',lambda v:v.__setitem__('cpu_probe_receipts',probes[:1]))
        manifest_negative('duplicate_CPU_probe_rejected',lambda v:v.__setitem__('cpu_probe_receipts',[probes[0],probes[0]]))
        manifest_negative('manifest_template_not_PASS_rejected',lambda v:v.__setitem__('status','TEMPLATE'))
        manifest_negative('duplicate_native_path_rejected',lambda v:v['files'].append(v['files'][0]))
        manifest_negative('omitted_original61_native_ELF_rejected',lambda v:v['files'].pop(0))
        manifest_negative('closure_cannot_redefine_original61_native_SHA',lambda v:v['files'][0].__setitem__('sha256','0'*64))
        manifest_negative('missing_native_provenance_rejected',lambda v:v['files'][0].__setitem__('provenance',{}))
        manifest_negative('noncanonical_alias_in_manifest_rejected',lambda v:v['files'][0].__setitem__('path',str(alias)))
        manifest_negative('relative_native_path_rejected',lambda v:v['files'][0].__setitem__('path','relative.so'))
        manifest_negative('CPU_probe_parent_escape_rejected',lambda v:v['cpu_probe_receipts'][0].__setitem__('path','../outside.json'))
        receipt=root/probes[0]['path'];saved=receipt.read_bytes();receipt.write_bytes(saved+b' ')
        reject('bound_CPU_receipt_byte_change_rejected',lambda:guard.load_native_closure(root));receipt.write_bytes(saved)
        evidence=guard.persist_native_audit(root,bad);persisted=json.loads((root/evidence['path']).read_text())
        check('durable_forensic_JSON_original_bytes_SHA',all(persisted[key]==value for key,value in bad.items())
              and persisted['writer_identity']['pid']==os.getpid() and type(persisted['writer_monotonic_ns'])is int
              and evidence['sha256']==guard.sha(root/evidence['path']))
        # Execute the actual public post-model guard without a model or GPU.
        # Only actual /proc text and NVML observation are explicit test mocks.
        real_read_text=Path.read_text
        def saved_maps(path,*a,**k):
            return mapped(unknown) if str(path)=='/proc/self/maps' else real_read_text(path,*a,**k)
        before=set((root/'operations/native_library_audit').glob('*.json'))
        with patch.object(Path,'read_text',saved_maps),patch.object(guard,'_own_gpu',return_value=dict(CPU_MOCK_NVML=True)):
            reject('actual_postmodel_guard_fails_only_after_durable_unknown_evidence',lambda:guard.check_after_model(root,1,os.getpid()))
        created=set((root/'operations/native_library_audit').glob('*.json'))-before
        check('actual_guard_wrote_unique_forensic_record_before_raise',len(created)==1 and json.loads(next(iter(created)).read_text())['guard_status']=='FAIL_CLOSED_STOP')
        failed=json.loads(next(iter(created)).read_text())
        check('actual_guard_failure_contains_unknown_path_stat_and_SHA',failed['mapped_native'][0]['path']==str(unknown)and failed['mapped_native'][0]['sha256']==guard.sha(unknown))
        def good_maps(path,*a,**k):
            return maps if str(path)=='/proc/self/maps' else real_read_text(path,*a,**k)
        with patch.object(Path,'read_text',good_maps),patch.object(guard,'_own_gpu',return_value=dict(CPU_MOCK_NVML=True)):
            passed=guard.check_after_model(root,1,os.getpid())
        check('actual_postmodel_guard_positive_full_union',passed['status']=='PASS_OWN_UUID_MODEL_STAT_AND_MAPPED_LIBRARY_IDENTITY'and passed['exact_native_closure']is True)
        check('positive_guard_also_preserves_fresh_evidence',(root/passed['native_audit']['path']).is_file())
        outside=parent/'outside_keep';outside.mkdir();(outside/'KEEP').write_bytes(b'UNCHANGED')
        # Replace the isolated fixture audit directory only, never science paths.
        for p in (root/'operations/native_library_audit').iterdir():p.unlink()
        (root/'operations/native_library_audit').rmdir();(root/'operations/native_library_audit').symlink_to(outside,target_is_directory=True)
        reject('forensic_output_symlink_directory_not_followed',lambda:guard.persist_native_audit(root,bad))
        check('external_symlink_target_untouched',(outside/'KEEP').read_bytes()==b'UNCHANGED'and len(list(outside.iterdir()))==1)
        check('original61_still_byte_identical_at_end',base_contract==(root/'analysis/EXTERNAL_BINDINGS.json').read_bytes())
        check('no_Torch_or_CUDA_import','torch'not in sys.modules)
        check('all_negative_symlinks_external_to_science_root',not any(p.is_symlink()for p in source.rglob('*')))
    result=dict(status='PASS_ACTUAL_RUNTIME_NATIVE_CLOSURE_GUARD_CPU_FIXTURES',check_count=len(checks),checks=checks,
                guard_sha256=guard.sha(source/'src/qualification_guards.py'),test_script_sha256=guard.sha(Path(__file__).resolve()),
                source_root=str(source),CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                identity=dict(pid=os.getpid(),starttick=None,argv=sys.argv),GPU_dispatches=0,model_loads=0,Torch_imported=False,
                scope='Isolated synthetic ELF-like files and maps; public post-model guard uses real SHA/stat/fsync filesystem behavior and mocked NVML/proc maps. No real ELF provenance/GPU proof.',
                external_negative_fixtures_destroyed_with_temporary_directory=True)
    try:
        raw=Path('/proc/self/stat').read_text();result['identity']['starttick']=int(raw[raw.rfind(')')+2:].split()[19])
        result['identity']['cmdline']=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    except FileNotFoundError:
        pass
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],check_count=result['check_count'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
