"""Append-only semantic ingress correction; old V2 implementation untouched.

This is a cooperative runtime provenance observer, not a malicious-code sandbox.
Unknown data and unsupported semantic roles fail closed. Ambient status never
authorizes compilation, imports, executable mappings, scientific use or writes.
"""
import importlib.util
import os
from pathlib import Path
import re
import stat

_old = Path(__file__).resolve().parent.parent / 'capsule_v2.py'
_spec = importlib.util.spec_from_file_location('preserved_v2_inventory', _old)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
Escape, canonical, sha, write = _base.Escape, _base.canonical, _base.sha, _base.write


def locale_alias_schema(raw):
    """A data grammar, not a filename exception. No code/eval interpretation."""
    try:
        lines = raw.decode('ascii').splitlines()
    except UnicodeError:
        return False
    rows = []
    for line in lines:
        text = line.split('#', 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 2:
            return False
        if not re.fullmatch(r'[A-Za-z0-9_.@+-]+', fields[0]):
            return False
        if not re.fullmatch(r'[a-z]{2,3}_[A-Z]{2}\.[A-Za-z0-9_-]+(?:@[A-Za-z0-9_-]+)?', fields[1]):
            return False
        rows.append(fields)
    return bool(rows)


def semantic_role(path, raw, context):
    """Require recognized support-data format, consumer and usage together."""
    p = Path(path)
    consumer = context.get('consumer_role')
    readonly = context.get('read_only', False)
    if not readonly:
        return None
    if consumer == 'FROZEN_GIT_IDENTITY_QUERY':
        if (p.is_relative_to('/etc') or p.is_relative_to('/usr/share/locale') or
                p.is_relative_to('/usr/lib/X11/locale')) and locale_alias_schema(raw):
            return 'LOCALE_RESOLUTION_ALIAS_DATA', 'Two-field locale alias grammar read only by frozen git identity/status query; not numerical configuration'
        if p.is_relative_to('/usr/share/locale') and raw[:4] in (b'\xde\x12\x04\x95', b'\x95\x04\x12\xde'):
            return 'GETTEXT_MESSAGE_CATALOG', 'GNU gettext data catalog for git identity/status messages, not imported or numerical input'
    if p.is_relative_to('/usr/share/zoneinfo') and raw[:4] == b'TZif' and consumer in ('FROZEN_RUNTIME_SUPPORT_READ', 'FROZEN_GIT_IDENTITY_QUERY'):
        return 'TIMEZONE_RESOLUTION_DATA', 'TZif OS timezone resolution bytes; no scientific data role, execution or mutation'
    # Synthetic consumers exist only in isolated tests, never in qualification.
    if context.get('synthetic_only') and consumer == 'SYNTHETIC_LOCALE_RESOLUTION' and locale_alias_schema(raw):
        return 'LOCALE_RESOLUTION_ALIAS_DATA', 'Isolated synthetic locale parser read, same grammar as production; no path exception'
    return None


class Inventory(_base.Inventory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ambient = {}
        self.transitions = []
        self.rejections = []
        self.consumer_context = {}

    def reject(self, path, category, reason):
        self.unknown += 1
        self.rejections.append(dict(path=path, classification=category, reason=reason))
        raise Escape(reason + ': ' + path)

    def ambient_record(self, p, phase, source, context, role, reason):
        if p in self.opened_for_write:
            self.reject(p, 'D_UNKNOWN', 'AMBIENT_ALREADY_MUTATED')
        try:
            s = os.stat(source)
            digest = sha(source) if stat.S_ISREG(s.st_mode) else None
            size = s.st_size if stat.S_ISREG(s.st_mode) else None
        except OSError as exc:
            self.reject(p, 'D_UNKNOWN', 'AMBIENT_BYTES_UNAVAILABLE_' + type(exc).__name__)
        prior = self.ambient.get(p)
        if prior and prior['sha256'] is not None and digest != prior['sha256']:
            self.mismatch += 1
            raise Escape('AMBIENT_CONTENT_CHANGED_DURING_QUALIFICATION: ' + p)
        row = dict(path=p, classification='C_AMBIENT_OS_METADATA_CONFIG', semantic_role=role,
                   read_only_ingress=True, sha256=digest, size=size, reason=reason,
                   first_phase=phase, consumer=context,
                   imported_as_code=False, executed=False, executable_mapping=False,
                   scientific_input=False, experiment_mutated=False)
        self.ambient[p] = row
        self.events.append(dict(path=p, phase=phase, classification=row['classification'], read_only_ingress=True))
        return 'ambient:' + p

    def observe(self, path, phase, executable=False, fd_path=None, context=None, scientific=False):
        p = os.path.realpath(path.removesuffix(' (deleted)'))
        source = fd_path or p
        ctx = context or self.consumer_context
        code = executable or phase in ('python_import', 'python_compile', 'python_exec', 'observer_implementation')
        previous = self.ambient.get(p)
        if previous and (code or scientific):
            category = 'A_EXECUTABLE_CODE_BEARING' if code else 'B_SCIENTIFIC_NUMERICAL_INPUT'
            self.transitions.append(dict(path=p, before=previous['classification'], after=category, phase=phase))
            self.reject(p, category, 'AMBIENT_ROLE_ESCALATION_REQUIRES_FROZEN_CLOSURE')
        # An OS-support grammar never grants executable or scientific access.
        role = None
        if not code and not scientific and Path(source).is_file():
            with open(source, 'rb') as f:
                raw = f.read(1024 * 1024 + 1)
            if len(raw) <= 1024 * 1024:
                role = semantic_role(p, raw, ctx)
        if role:
            return self.ambient_record(p, phase, source, ctx, *role)
        try:
            identity = super().observe(p, phase, executable=code, fd_path=source)
        except Escape as exc:
            self.rejections.append(dict(path=p, classification=(
                'A_EXECUTABLE_CODE_BEARING' if code else
                'B_SCIENTIFIC_NUMERICAL_INPUT' if scientific else 'D_UNKNOWN'), reason=str(exc)))
            raise
        row = self.rows[identity]
        old_class = row['classification']
        looks_code = p.endswith(('.py', '.pyc', '.so', '.pyd', '.dylib')) or '.so.' in p
        category = 'A_EXECUTABLE_CODE_BEARING' if code or looks_code else 'B_SCIENTIFIC_NUMERICAL_INPUT'
        # Frozen membership still checks original exact identity AND bytes.
        row['semantic_classification'] = category
        row['closure_classification'] = old_class
        row['closure_required'] = True
        return identity

    def check_write(self, path):
        p = os.path.realpath(path.removesuffix(' (deleted)'))
        if p in self.ambient:
            self.transitions.append(dict(path=p, before='C_AMBIENT_OS_METADATA_CONFIG', after='D_UNKNOWN', phase='write'))
            self.reject(p, 'D_UNKNOWN', 'AMBIENT_MUTATION_FORBIDDEN')
        return super().check_write(path)

    def finish(self, status, error=None):
        if status == 'PASS':
            for row in self.ambient.values():
                if row['sha256'] is not None and sha(row['path']) != row['sha256']:
                    status, error = 'FAIL', 'AMBIENT_BYTES_CHANGED_AFTER_READ: ' + row['path']
                    self.mismatch += 1
                    break
        report = super().finish(status, error)
        write(self.output / 'semantic_ingress_report.json', dict(
            gate=report['gate'], error=report['error'],
            closure_artifacts=report['artifacts'], ambient_metadata=list(self.ambient.values()),
            classification_transitions=self.transitions, rejections=self.rejections,
            ambient_observations=self.events, unknown_is_blocking=True,
            path_specific_locale_exception=False, numerical_protocol_changed=False,
            coverage_limit='Cooperative file-origin and native syscall observer, not adversarial content-flow taint analysis'))
        return report
