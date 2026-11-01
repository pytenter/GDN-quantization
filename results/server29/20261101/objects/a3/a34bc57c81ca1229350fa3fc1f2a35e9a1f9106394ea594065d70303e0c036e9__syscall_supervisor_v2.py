"""Append-only independent supervisor extension; preserves old supervisor bytes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
from capsule_v2 import Inventory, Escape, sha, write

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('preserved_v2_supervisor', P/'preserved_supervisor.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


class Supervisor(old.Supervisor):
    def __init__(self, inv):
        super().__init__(inv)
        self.process_context = {}
        self.channel = None
        self.audit_events = []
        self.audit_installed_pids = set()

    def context(self, pid):
        try:
            exe = os.path.realpath('/proc/%d/exe' % pid)
            argv = Path('/proc/%d/cmdline' % pid).read_bytes().rstrip(b'\0').split(b'\0')
            argv = [os.fsdecode(x) for x in argv]
            role = 'FROZEN_RUNTIME_SUPPORT_READ'
            if self.channel is not None and pid in self.audit_installed_pids:
                # A post-bootstrap arbitrary Python read is not automatically
                # an OS-support consumer. Unknown scientific data flow stops.
                role = 'UNPROVEN_POST_BOOTSTRAP_CONSUMER'
            accepted = [('rev-parse', 'HEAD'), ('status', '--porcelain=v1', '--untracked-files=no')]
            if exe == '/usr/bin/git' and tuple(argv[1:]) in accepted:
                assert 'path:'+exe in self.inv.rows, 'CONSUMER_NOT_IN_EXECUTABLE_CLOSURE'
                role = 'FROZEN_GIT_IDENTITY_QUERY'
            if self.inv.policy.get('synthetic_ambient_test_only'):
                return dict(pid=pid,executable=exe,argv=argv,consumer_role='SYNTHETIC_LOCALE_RESOLUTION',
                            read_only=True,synthetic_only=True)
            return dict(pid=pid, executable=exe, argv=argv, consumer_role=role, read_only=True)
        except (OSError, AssertionError) as exc:
            return dict(pid=pid, consumer_role='UNKNOWN', read_only=True, diagnostic=repr(exc))

    def ingress(self, pid, fd, reason, executable=False):
        link = '/proc/%d/fd/%d' % (pid, old.ctypes.c_int64(fd).value)
        try:
            p = os.readlink(link); mode = os.stat(link).st_mode
        except FileNotFoundError:
            return
        if p.startswith(('pipe:[', 'socket:[')):
            return super().ingress(pid, fd, reason, executable)
        ctx = self.context(pid)
        if not stat.S_ISREG(mode):
            if p in ('/dev/null', '/dev/urandom', '/dev/random') and not executable:
                return self.inv.ambient_record(p, reason, link, ctx, 'OS_DEVICE_SUPPORT',
                    'Read-only OS null/entropy support device; bytes unavailable for stable regular-file SHA256')
            raise Escape('UNCLASSIFIED_NONREGULAR_INGRESS: ' + p)
        # No advisory-prefix bypass. Even former ignored Git/proc files close
        # as B/unknown unless explicitly semantically justified as C.
        self.inv.observe(p, reason, executable=executable, fd_path=link, context=ctx)

    def entry(self, pid, nr, args):
        self.inv.consumer_context = self.context(pid)
        if nr in (59, 322):
            # Preserve generic absent-PATH-probe behavior from original V2.
            p = old.cstring(pid, args[0] if nr == 59 else args[1])
            p = old.path_at(pid, args[0] if nr == 322 else -100, p)
            self.count += 1
            if os.path.isfile(p):
                self.inv.observe(p, 'process_executable', executable=True)
            return
        if nr in (2, 257):
            p = old.path_at(pid, -100 if nr == 2 else args[0], old.cstring(pid, args[0] if nr == 2 else args[1]))
            flags = args[1] if nr == 2 else args[2]
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC) and os.path.realpath(p) in self.inv.ambient:
                self.inv.check_write(p)
        return super().entry(pid, nr, args)

    def audit_pending(self):
        if self.channel is None:
            return
        while True:
            try:
                raw = self.channel.recv(65536)
            except BlockingIOError:
                break
            if not raw:
                break
            event = json.loads(raw)
            self.audit_events.append(event)
            if event['event']=='observer_installed':
                self.audit_installed_pids.add(event['pid'])
            if event['event'] == 'compile' and event.get('source_sha256'):
                for row in self.inv.ambient.values():
                    if row['sha256'] == event['source_sha256']:
                        self.inv.observe(row['path'], 'python_compile', executable=True)
            p = event.get('path')
            if p and os.path.isfile(p):
                self.inv.observe(p, 'python_'+event['event'], executable=True)
            self.channel.send(b'PASS')

    def exit(self, pid, nr, args, result):
        super().exit(pid, nr, args, result)
        self.audit_pending()


def main():
    parser = old.argparse.ArgumentParser()
    for k in ('policy', 'output', 'cwd'):
        parser.add_argument('--'+k, required=True)
    parser.add_argument('--frozen'); parser.add_argument('--audit-channel', action='store_true')
    parser.add_argument('command', nargs=old.argparse.REMAINDER)
    a = parser.parse_args()
    inv = Inventory(json.loads(Path(a.policy).read_text()), a.output,
                    json.loads(Path(a.frozen).read_text()) if a.frozen else None)
    monitor = Supervisor(inv)
    status, error = 'PASS', None
    env = os.environ.copy()
    child_channel = None
    if a.audit_channel:
        monitor.channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        monitor.channel.setblocking(False)
        child_channel.set_inheritable(True)
        env['LING_PROVENANCE_AUDIT_FD'] = str(child_channel.fileno())
        monitor.sockets[os.readlink('/proc/self/fd/%d' % child_channel.fileno())] = 1
    try:
        for p in (Path(__file__), P/'capsule_v2.py', P/'preserved_inventory.py', P/'preserved_supervisor.py', Path(a.policy)):
            inv.observe(str(p), 'observer_implementation', executable=True)
        cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
        monitor.run(cmd, a.cwd, env)
    except BaseException as exc:
        status, error = 'FAIL', repr(exc)
    finally:
        if child_channel: child_channel.close()
        if monitor.channel: monitor.channel.close()
    report = inv.finish(status, error)
    actual = []
    for pid in sorted(monitor.live):
        if Path('/proc/%d' % pid).exists():
            actual.append(pid)
    write(Path(a.output)/'supervisor.json', dict(syscall_stop_count=monitor.count,
        root_exit_status=monitor.root_status, cleanup_ledger_remaining=len(monitor.live),
        actual_surviving_processes=actual, original_v2_cleanup_ledger_remaining=1,
        python_origin_audit_events=monitor.audit_events, numerical_environment_changed=False,
        mechanism='PRESERVED_PTRACE_WITH_SEMANTIC_CLASSIFIER_AND_OPTIONAL_PYTHON_AUDIT'))
    print(json.dumps(dict(gate=report['gate'], error=report['error'], closure_count=len(inv.rows),
        ambient_count=len(inv.ambient))), flush=True)
    return 0 if report['gate']=='PASS' else 1


if __name__=='__main__': sys.exit(main())
