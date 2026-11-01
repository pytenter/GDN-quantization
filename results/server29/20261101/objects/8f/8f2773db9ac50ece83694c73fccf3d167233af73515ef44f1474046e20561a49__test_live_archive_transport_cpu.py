#!/usr/bin/env python3
"""Finite 4MiB engineering roundtrip on .55->.29. No CUDA/model imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from archive_io import SSHArchiveTransport, load_archive_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--root', required=True, type=Path);a=p.parse_args()
    config=load_archive_config(a.root);t=SSHArchiveTransport(config)
    preflight=t.preflight()
    raw=bytes(range(256))*16384;sha=hashlib.sha256(raw).hexdigest()
    name=config['archive_root']+'/engineering_transport_probe/roundtrip_4MiB.bin'
    written=t.put_immutable(name, [raw], len(raw), sha)
    with t.open_verified(name,len(raw),sha) as f:
        actual=f.read(len(raw));assert actual==raw and not f.read(1)
    # Reusing an already published object must be refused, never overwritten.
    refused=False
    try:t.put_immutable(name,[raw],len(raw),sha)
    except (RuntimeError,BrokenPipeError):refused=True
    assert refused
    with t.open_verified(name,len(raw),sha) as f:
        assert f.read(len(raw))==raw and not f.read(1)
    result={'status':'PASS_CPU_STORAGE_TRANSPORT_ONLY','GPU_imports':0,'GPU_dispatches':0,
            'bytes':len(raw),'sha256':sha,'remote_receipt':written,'preflight':preflight,
            'same_name_overwrite_refused':True,'original_bytes_still_verified':True}
    path=a.root/'qualification/LIVE_ARCHIVE_TRANSPORT_CPU.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(result))

if __name__=='__main__':main()
