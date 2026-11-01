"""Linux x86-64 independent ptrace ingress observer. No LD/PYTHONPATH changes.

Stops every tracee at syscall boundaries. Hashes regular-file ingress before
read/mmap resumes, including C-level dlopen and preload. No function profiling.
Unknown asynchronous ingress mechanisms fail, rather than pretending coverage.
"""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import stat
import struct
import sys
from capsule_v2 import Escape, Inventory, write

libc = ctypes.CDLL(None, use_errno=True)
libc.ptrace.restype = ctypes.c_long
libc.ptrace.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
OPTIONS = 1 | 2 | 4 | 8 | 16 | (1 << 20)  # TRACESYSGOOD/FORK/VFORK/CLONE/EXEC/EXITKILL
WALL = 0x40000000


def ptrace(op, pid, addr=0, data=0):
    result = libc.ptrace(op, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    if result == -1:
        raise OSError(ctypes.get_errno(), 'ptrace', (op, pid))
    return result


def syscall_info(pid):
    raw = ctypes.create_string_buffer(128)
    n = ptrace(0x420e, pid, 128, ctypes.addressof(raw))
    if n < 24:
        raise Escape('PTRACE_GET_SYSCALL_INFO_UNAVAILABLE')
    return raw.raw[0], raw.raw


def cstring(pid, addr):
    result = bytearray()
    # /proc/pid/mem is read by the independent observer, never by replay.
    with open('/proc/%d/mem' % pid, 'rb', buffering=0) as f:
        f.seek(addr)
        while len(result) < 65536:
            b = f.read(256)
            if not b:
                raise Escape('UNREADABLE_SYSCALL_PATH')
            end = b.find(b'\0')
            if end >= 0:
                return os.fsdecode(result + b[:end])
            result += b
    raise Escape('UNBOUNDED_SYSCALL_PATH')


def fd_target(pid, fd):
    return os.readlink('/proc/%d/fd/%d' % (pid, ctypes.c_int64(fd).value))


def path_at(pid, fd, text):
    if text.startswith('/'):
        return text
    base = os.readlink('/proc/%d/cwd' % pid) if ctypes.c_int64(fd).value == -100 else fd_target(pid, fd)
    return os.path.normpath(os.path.join(base, text))


class Supervisor:
    def __init__(self, inventory):
        self.inv = inventory
        self.live = set()
        self.entries = {}
        self.sockets = {}
        self.count = 0
        self.root_status = None

    def metadata(self, p):
        # Advisory device/OS interfaces, not sources of executable code.
        # Refuse memory, fd alias, arbitrary process files and stdin ingress.
        return p in ('/dev/null', '/dev/urandom', '/dev/random') or p.startswith('/dev/nvidia') or p.startswith('/dev/dri/') or p in self.inv.policy.get('advisory_exact', []) or any(p.startswith(r) for r in self.inv.policy.get('advisory_prefixes', []))

    def ingress(self, pid, fd, reason, executable=False):
        link = '/proc/%d/fd/%d' % (pid, ctypes.c_int64(fd).value)
        try:
            p = os.readlink(link)
            mode = os.stat(link).st_mode
        except FileNotFoundError:
            return  # EBADF syscall introduces no bytes.
        if p.startswith(('pipe:[', 'socket:[')):
            # Pipes are derived from traced child processes; internet sockets
            # cannot be created under the separate network boundary below.
            if p.startswith('socket:[') and p not in self.sockets:
                raise Escape('UNKNOWN_INHERITED_SOCKET_INGRESS: ' + p)
            return
        if not stat.S_ISREG(mode):
            if self.metadata(p) and not executable:
                return
            raise Escape('UNCLASSIFIED_NONREGULAR_INGRESS: ' + p)
        if self.metadata(p) and not executable:
            return
        self.inv.observe(p, reason, executable=executable, fd_path=link)

    def executable_maps(self, pid):
        with open('/proc/%d/maps' % pid) as f:
            for line in f:
                parts = line.rstrip().split(None, 5)
                if 'x' not in parts[1] or len(parts) != 6:
                    continue
                p = parts[5]
                if p.startswith('/'):
                    self.inv.observe(p, 'executable_mapping', executable=True)
                elif p not in ('[vdso]', '[vsyscall]'):
                    raise Escape('UNCLASSIFIED_NAMED_EXECUTABLE_MAPPING: ' + p)
        # Anonymous JIT memory is derived, provided ingress is already closed.

    def entry(self, pid, nr, a):
        self.count += 1
        if nr in (0, 17, 19, 295, 327):
            self.ingress(pid, a[0], 'external_read')
        elif nr == 9 and ctypes.c_int64(a[4]).value != -1:
            self.ingress(pid, a[4], 'file_backed_mmap', bool(a[2] & 4))
        elif nr in (59, 322):
            p = cstring(pid, a[0] if nr == 59 else a[1])
            if nr == 322:
                p = path_at(pid, a[0], p)
            self.inv.observe(p, 'process_executable', executable=True)
        elif nr == 41:
            # AF_INET / AF_INET6 forbidden before creation. AF_UNIX and
            # AF_NETLINK are logged advisory IPC, never silently internet.
            if a[0] not in (1, 16):
                self.inv.network += 1
                raise Escape('NETWORK_SOCKET_ESCAPE: domain=' + str(a[0]))
        elif nr in (101, 310, 311, 425, 426, 427, 438):
            raise Escape('UNSUPPORTED_EXTERNAL_INGRESS_MECHANISM: syscall=' + str(nr))
        elif nr in (2, 257, 437):
            if nr == 2:
                p, flags = cstring(pid, a[0]), a[1]
                p = path_at(pid, -100, p)
            elif nr == 257:
                p, flags = path_at(pid, a[0], cstring(pid, a[1])), a[2]
            else:
                raise Escape('OPENAT2_COVERAGE_NOT_IMPLEMENTED')
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC):
                if not self.metadata(p):
                    self.inv.check_write(p)
        elif nr in (1, 18, 20, 296, 328):
            try:
                p = fd_target(pid, a[0])
            except FileNotFoundError:
                return
            if p.startswith('/') and not self.metadata(p):
                self.inv.check_write(p)

    def exit(self, pid, nr, a, result):
        if result < 0:
            return
        if nr == 41:
            self.sockets[fd_target(pid, result)] = a[0]
        if nr in (9, 10, 329):
            self.executable_maps(pid)

    def run(self, command, cwd, env):
        if os.uname().machine != 'x86_64':
            raise Escape('ARCHITECTURE_UNSUPPORTED')
        pid = os.fork()
        if pid == 0:
            os.chdir(cwd)
            ptrace(0, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            os.execve(command[0], command, env)
        self.live.add(pid)
        os.waitpid(pid, 0)
        ptrace(0x4200, pid, 0, OPTIONS)
        ptrace(24, pid)
        try:
            while self.live:
                tid, status = os.waitpid(-1, WALL)
                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    if tid == pid:
                        self.root_status = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status)
                    self.live.discard(tid)
                    continue
                sig = os.WSTOPSIG(status)
                event = status >> 16
                if event in (1, 2, 3):
                    msg = ctypes.c_ulong()
                    ptrace(0x4201, tid, 0, ctypes.addressof(msg))
                    self.live.add(msg.value)
                elif sig == (signal.SIGTRAP | 0x80):
                    op, raw = syscall_info(tid)
                    if op == 1:
                        nr, *args = struct.unpack_from('7Q', raw, 24)
                        self.entries[tid] = (nr, args)
                        self.entry(tid, nr, args)
                    elif op == 2 and tid in self.entries:
                        nr, args = self.entries.pop(tid)
                        self.exit(tid, nr, args, struct.unpack_from('q', raw, 24)[0])
                    else:
                        raise Escape('UNCLASSIFIED_SYSCALL_STOP')
                elif event == 4:
                    self.entries.pop(tid, None)
                    self.executable_maps(tid)
                elif event or sig not in (signal.SIGSTOP, signal.SIGTRAP):
                    ptrace(24, tid, 0, sig)
                    continue
                ptrace(24, tid)
        finally:
            for tid in self.live:
                try:
                    os.kill(tid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            while self.live:
                try:
                    tid, status = os.waitpid(-1, WALL)
                    if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                        self.live.discard(tid)
                    elif os.WIFSTOPPED(status):
                        ptrace(24, tid, 0, signal.SIGKILL)
                except ChildProcessError:
                    break
        if self.root_status != 0:
            raise Escape('CHILD_FAILURE: ' + str(self.root_status))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--frozen')
    p.add_argument('--cwd', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    policy = json.loads(Path(a.policy).read_text())
    frozen = json.loads(Path(a.frozen).read_text()) if a.frozen else None
    inv = Inventory(policy, a.output, frozen)
    monitor = Supervisor(inv)
    status, error = 'PASS', None
    try:
        # Bind the independent supervisor's own external implementation too.
        for path in (Path(__file__), Path(__file__).with_name('capsule_v2.py'), Path(a.policy)):
            inv.observe(str(path), 'observer_implementation')
        command = a.command[1:] if a.command[:1] == ['--'] else a.command
        monitor.run(command, a.cwd, os.environ.copy())
    except BaseException as exc:
        status, error = 'FAIL', repr(exc)
    report = inv.finish(status, error)
    write(Path(a.output) / 'supervisor.json', dict(syscall_stop_count=monitor.count,
        root_exit_status=monitor.root_status, live_tracees_after_exit=len(monitor.live),
        mechanism='PTRACE_SYSCALL_ALL_TRACEES_X86_64', numerical_environment_changed=False,
        coverage_boundary='regular-file read/mmap/exec; native mprotect/map; Python startup baseline supplied by child; network socket ingress; unsupported asynchronous ingress fails'))
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts'}), flush=True)
    sys.exit(0 if report['gate'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
