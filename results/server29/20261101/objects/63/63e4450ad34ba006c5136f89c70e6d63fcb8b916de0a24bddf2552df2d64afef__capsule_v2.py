"""Artifact identities, not code-object/event identities. No scientific code."""
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())


class Escape(RuntimeError):
    pass


class Inventory:
    def __init__(self, policy, output, frozen=None):
        self.policy = policy
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.frozen = frozen
        self.rows = {}
        self.opened_for_write = set()
        self.cached = {}
        self.events = []
        self.network = 0
        self.mismatch = 0
        self.unknown = 0
        self.runtime_escape = 0
        self.observed_native = set()
        self.expected = {} if frozen is None else {r['identity']: r for r in frozen['artifacts']}

    def classification(self, path, executable=False):
        p = os.path.realpath(path)
        # Explicit synthetic fixture classes are part of the test policy only.
        for row in self.policy.get('exact', []):
            if p == row['path']:
                return row['classification'], row.get('volatile', False)
        for row in self.policy['roots']:
            root = row['path'].rstrip('/')
            if p == root or p.startswith(root + '/'):
                return row['classification'], row.get('volatile', False)
        # Generated files are external file-backed artifacts, NOT in-memory
        # derived code. Discovery retains bytes; enforcement checks content ID.
        if p in self.opened_for_write and (p.startswith('/tmp/') or '/.triton/' in p):
            return 'PROCESS_GENERATED_FILE_BACKED_ARTIFACT', True
        raise Escape('UNCLASSIFIED_EXTERNAL_ARTIFACT: ' + p)

    def observe(self, path, phase, executable=False, fd_path=None):
        diagnostic = path.removesuffix(' (deleted)')
        p = os.path.realpath(diagnostic)
        try:
            classification, volatile = self.classification(p, executable)
        except Escape:
            self.unknown += 1
            self.runtime_escape += 1
            raise
        source = fd_path or p
        st = os.stat(source)
        stamp = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        cached = self.cached.get(p)
        if cached is None or cached[0] != stamp:
            content = sha(source)
            self.cached[p] = (stamp, content)
        else:
            content = cached[1]
        identity = ('content:' + classification + ':' + content) if volatile else 'path:' + p
        if self.frozen is not None:
            old = self.expected.get(identity)
            if old is None:
                self.unknown += 1
                self.runtime_escape += 1
                raise Escape('UNKNOWN_EXTERNAL_ARTIFACT: ' + identity)
            if old['sha256'] != content or old['size'] != st.st_size or old['classification'] != classification:
                self.mismatch += 1
                raise Escape('EXTERNAL_CONTENT_HASH_MISMATCH: ' + p)
        old = self.rows.get(identity)
        if old and old['sha256'] != content:
            self.mismatch += 1
            raise Escape('EXTERNAL_ARTIFACT_MUTATED_DURING_RUN: ' + p)
        if volatile:
            dest = self.output / 'retained_blobs' / content
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(source, 'rb') as src, dest.open('xb') as dst:
                    for raw in iter(lambda: src.read(8 * 1024 * 1024), b''):
                        dst.write(raw)
                if sha(dest) != content:
                    self.mismatch += 1
                    raise Escape('VOLATILE_CAPTURE_BYTE_MISMATCH')
        if identity not in self.rows:
            self.rows[identity] = dict(identity=identity, path=p, classification=classification,
                volatile=volatile, sha256=content, size=st.st_size, phase=phase)
        if executable or p.endswith(('.so', '.pyd', '.dylib')) or '.so.' in p:
            self.observed_native.add(identity)
        return identity

    def check_write(self, path):
        p = os.path.realpath(path.removesuffix(' (deleted)'))
        allowed = any(p.startswith(x.rstrip('/') + '/') for x in self.policy['write_roots'])
        if not allowed:
            self.runtime_escape += 1
            raise Escape('WRITE_OUTSIDE_ENGINEERING_OUTPUT_OR_EXISTING_JIT_CACHE: ' + p)
        # A stable bound artifact cannot be changed even in an allowed output root.
        if self.frozen and 'path:' + p in self.expected:
            self.runtime_escape += 1
            raise Escape('FROZEN_EXTERNAL_ARTIFACT_WRITE: ' + p)
        self.opened_for_write.add(p)

    def finish(self, status, error=None):
        # Stable files are checked once more. Volatile raw blobs survive exit.
        if status == 'PASS':
            for row in list(self.rows.values()):
                if row['volatile']:
                    assert sha(self.output / 'retained_blobs' / row['sha256']) == row['sha256']
                else:
                    if sha(row['path']) != row['sha256']:
                        self.mismatch += 1
                        status, error = 'FAIL', 'POST_RUN_STABLE_ARTIFACT_MISMATCH: ' + row['path']
                        break
        report = dict(gate=status, error=error, artifacts=sorted(self.rows.values(), key=lambda x: x['identity']),
            OBSERVED_EXTERNAL_ARTIFACT_COUNT=len(self.rows), UNKNOWN_EXTERNAL_ARTIFACT_COUNT=self.unknown,
            CONTENT_HASH_MISMATCH_COUNT=self.mismatch, RUNTIME_ESCAPE_COUNT=self.runtime_escape,
            NETWORK_ESCAPE_COUNT=self.network, NATIVE_EXTERNAL_ARTIFACTS=len(self.observed_native),
            VOLATILE_EXTERNAL_ARTIFACTS=sum(r['volatile'] for r in self.rows.values()),
            enforcement_manifest_expanded=False, execution_source_events_traced=False)
        write(self.output / 'runtime_escape_report.json', report)
        return report
