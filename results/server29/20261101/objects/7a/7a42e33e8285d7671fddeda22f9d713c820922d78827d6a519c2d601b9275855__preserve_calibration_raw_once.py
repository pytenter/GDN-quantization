"""CPU-only exact-byte archival input clone after actual FP collector release.

Original shared-memory scientific files are preserved. This creates ordinary
files under the new root so the terminal archive includes every actual raw file.
Failure leaves its exclusive claim and partial copies; never auto-retry.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import traceback


def main(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU raw preservation requires empty CUDA visibility')
    sys.path.insert(0,str(root/'src'))
    import execution_control as C
    release=C.read(root/'calibration/FP_COLLECTION_EXIT_RELEASE.json')
    manifest=C.read(root/'calibration/FP_COLLECTION_MANIFEST.json')
    C.require(release['status']=='PASS_ACTUAL_EXIT0_AND_RELEASE' and release['worker_identity']==manifest['identity'],
              'Actual exact collector exit/release required')
    C.require(not C.same_process(manifest['identity']),'Collector still alive')
    C.atomic_new(root/'calibration/RAW_CLONE_CLAIM.json',dict(identity=C.process_identity(os.getpid()),
        manifest_sha256=C.sha(root/'calibration/FP_COLLECTION_MANIFEST.json'),GPU_calls=0))
    try:
        source=Path(manifest['raw_root']);dest=root/'calibration/raw';dest.mkdir(exist_ok=False)
        C.require(not source.is_symlink() and not dest.is_symlink(),'No raw path symlinks')
        total=sum(x['bytes'] for x in manifest['files']);s=os.statvfs(root)
        C.require(s.f_bavail*s.f_frsize-total>=4*1024**3,'Insufficient independent output disk for raw clone')
        rows=[]
        # RAW_IDENTITY supplements the actual frozen science roster.
        files=manifest['files']+[dict(relative='RAW_IDENTITY.json',path=str(source/'RAW_IDENTITY.json'),
            bytes=(source/'RAW_IDENTITY.json').stat().st_size,sha256=C.sha(source/'RAW_IDENTITY.json'))]
        for row in files:
            rel=Path(row['relative']);p=source/rel;q=dest/rel
            C.require(not rel.is_absolute() and '..' not in rel.parts and str(p)==row['path'] and not p.is_symlink(),
                      'Unsafe actual raw source path')
            before=p.stat();q.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
            with p.open('rb') as inp,q.open('xb') as out:
                for chunk in iter(lambda:inp.read(4*1024**2),b''):
                    out.write(chunk);h.update(chunk);count+=len(chunk)
                out.flush();os.fsync(out.fileno())
            after=p.stat()
            C.require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                      (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Raw source changed during CPU copy')
            C.require(count==row['bytes'] and q.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256']==C.sha(q),
                      'Independent raw copy SHA/bytes mismatch')
            rows.append(dict(source=str(p),path=q.relative_to(root).as_posix(),bytes=count,sha256=h.hexdigest()))
        C.atomic_new(root/'calibration/RAW_CLONE_VERIFICATION.json',dict(status='PASS_ALL_ACTUAL_NONAIME_RAW_SHA_BYTES',
            files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows),original_SHM_files_preserved=True,
            model_weight_copies=0,Mac_tensor_downloads=0,GPU_calls=0,
            collection_manifest_sha256=C.sha(root/'calibration/FP_COLLECTION_MANIFEST.json')))
    except BaseException as error:
        C.atomic_new(root/'calibration/RAW_CLONE_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),
            preserve_claim_and_partial=True,no_automatic_retry=True,GPU_calls=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
