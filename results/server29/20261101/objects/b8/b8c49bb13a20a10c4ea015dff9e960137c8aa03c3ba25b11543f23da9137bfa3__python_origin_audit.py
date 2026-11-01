"""Origin-only Python audit, not call tracing or code-object whitelisting."""
import hashlib
import json
import os
import socket
import sys


def install():
    fd = os.environ.get('LING_PROVENANCE_AUDIT_FD')
    if fd is None:
        raise RuntimeError('INDEPENDENT_PYTHON_ORIGIN_CHANNEL_REQUIRED')
    channel = socket.socket(fileno=int(fd))
    busy = False
    channel.send(json.dumps(dict(event='observer_installed',pid=os.getpid())).encode('utf-8'))
    if channel.recv(16)!=b'PASS':raise RuntimeError('AUDIT_PHASE_MARKER_NOT_ACKNOWLEDGED')

    def audit(event, args):
        nonlocal busy
        if busy or event not in ('compile', 'exec', 'import'):
            return
        path = None
        digest = None
        if event == 'compile':
            source, path = args[:2]
            if isinstance(source, str): source = source.encode('utf-8')
            if isinstance(source, bytes): digest = hashlib.sha256(source).hexdigest()
        elif event == 'exec':
            path = args[0].co_filename
        elif len(args) > 1:
            path = args[1]
        if not (isinstance(path, str) and path.startswith('/')):
            path = None
        if path is None and digest is None:
            return
        busy = True
        try:
            channel.send(json.dumps(dict(event=event, path=path, source_sha256=digest,
                pid=os.getpid())).encode('utf-8'))
            if channel.recv(16) != b'PASS':
                raise RuntimeError('ORIGIN_AUDIT_NOT_ACKNOWLEDGED')
        finally:
            busy = False
    sys.addaudithook(audit)
    return channel
