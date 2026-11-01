#!/usr/bin/env python3
"""Read-only, finite small-file transport. Stdout is TAR; no GPU or model loads."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--ready-sha256', required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    ready_bytes = (root / 'DELIVERY_READY.json').read_bytes()
    assert hashlib.sha256(ready_bytes).hexdigest() == a.ready_sha256
    ready = json.loads(ready_bytes)
    manifest_bytes = (root / 'SHA256_MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == ready['manifest']['sha256']
    manifest = json.loads(manifest_bytes)
    rows = [dict(x, source=x.get('remote_path', x['path']))
            for x in manifest['all_current_task_files'] + manifest['frozen_small_external_dependencies'] if x['download']]
    for key in ('manifest', 'checksums', 'remote_verification'):
        d = ready[key]
        rows.append(dict(d, source=d['path'], delivery_path=Path(d['path']).relative_to(root).as_posix()))
    rows.append(dict(source=str(root / 'DELIVERY_READY.json'), delivery_path='DELIVERY_READY.json',
                     bytes=len(ready_bytes), sha256=a.ready_sha256))
    paths = [r['delivery_path'] for r in rows]
    assert len(paths) == len(set(paths))
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for row in rows:
            name = PurePosixPath(row['delivery_path'])
            assert not name.is_absolute() and '..' not in name.parts
            p = Path(row['source'])
            assert row['bytes'] <= 16 * 1024**2 and not p.name.endswith('.trace.json')
            assert p.suffix not in {'.pt', '.pth', '.bin', '.safetensors', '.npz', '.npy'}
            data = p.read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], str(p)
            info = tarfile.TarInfo(str(name))
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))

if __name__ == '__main__':
    main()
