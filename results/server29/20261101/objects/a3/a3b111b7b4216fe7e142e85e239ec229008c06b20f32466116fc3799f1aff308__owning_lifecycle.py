"""Own Popen lifecycle. /proc observations never establish an OS exit code."""
import os
from pathlib import Path
import signal
import time


class OwnershipViolation(RuntimeError):
    pass


class ExitReleaseTimeout(RuntimeError):
    pass


def _stat(pid):
    value=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),state=value[0],ppid=int(value[1]),starttick=int(value[19]))


def read_snapshot(pid):
    before=_stat(pid)
    raw=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    after=_stat(pid)
    return dict(after,cmdline=[v.decode() for v in raw.split(b'\0') if v],
                cmdline_empty=not raw.rstrip(b'\0'),
                stable=all(before[k]==after[k] for k in ('pid','ppid','starttick')),
                state_before=before['state'])


def task_snapshots(pid):
    result=[]
    try:paths=list((Path('/proc')/str(pid)/'task').iterdir())
    except OSError:return result
    for path in paths:
        try:
            value=(path/'stat').read_text().rsplit(')',1)[1].split()
            result.append(dict(tid=int(path.name),state=value[0]))
        except (OSError,ValueError):result.append(dict(tid=int(path.name),state='DISAPPEARED_DURING_READ'))
    return result


class OwningLifecycle:
    """RUNNING -> EXIT_PENDING -> EXITED, with a separate hard identity fault.

    A zombie group leader can coexist with live cleanup threads and an unreaped
    Popen. Empty argv or a disappearing /proc entry is an uncertain exit boundary,
    never proof of success, failure, or PID reuse. The existing120s cleanup budget
    bounds this phase. Live nonempty changed argv, wrong starttick or wrong PPID
    is a fault and can never be a signal target.
    """
    def __init__(self,proc,expected,owner,emit,*,cleanup_seconds=120,
                 reader=read_snapshot,clock=time.monotonic):
        if proc.pid!=expected['pid'] or owner['pid']!=os.getpid():
            raise OwnershipViolation('Direct owning Popen PID/parent differs')
        if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
            raise OwnershipViolation('Owning wait requires default SIGCHLD handling')
        if not expected['cmdline'] or any(not x for x in expected['cmdline']):
            raise OwnershipViolation('Registration requires exact nonempty full argv')
        self.proc=proc;self.expected=expected;self.owner=owner;self.emit=emit
        self.cleanup_seconds=cleanup_seconds;self.reader=reader;self.clock=clock
        self.pending_since=None;self.exit_observed=None;self.cleanup_started=None
        self.phase=None;self.failure=None;self.last_event=None;self.last_snapshot=None;self.pending_ever=False
        self.event('REGISTERED',None,reason='Exact child full argv registered before monitor')

    def event(self,phase,snapshot,**values):
        signature=(phase,values.get('reason'),values.get('returncode'),
                   None if snapshot is None else tuple((k,str(snapshot.get(k))) for k in ('pid','starttick','ppid','state','cmdline','stable')))
        self.phase=phase;self.last_snapshot=snapshot
        if signature==self.last_event:return
        self.last_event=signature
        evidence=dict(snapshot or {})
        if phase in ('EXIT_PENDING','OWNERSHIP_VIOLATION','EXIT_RELEASE_TIMEOUT'):
            evidence['tasks']=task_snapshots(self.proc.pid)
        self.emit(dict(phase=phase,expected=self.expected,owner=self.owner,
                       observed=evidence,monotonic=self.clock(),**values))

    def exited(self,rc,snapshot=None):
        if self.exit_observed is None:self.exit_observed=self.clock()
        if snapshot is None:snapshot=self.last_snapshot
        self.event('EXITED',snapshot,returncode=rc,
                   reason='Actual owning Popen poll/wait observed OS exit')
        return dict(phase='EXITED',returncode=rc,snapshot=snapshot,
                    prior_failure=self.failure)

    def fault(self,reason,snapshot):
        if self.failure is None:self.failure=reason
        self.event('OWNERSHIP_VIOLATION',snapshot,reason=reason)
        raise OwnershipViolation(reason)

    def observe(self):
        rc=self.proc.poll()
        if rc is not None:return self.exited(rc)
        missing=None
        try:snapshot=self.reader(self.proc.pid)
        except (OSError,ValueError) as exc:snapshot=None;missing=repr(exc)
        # Exit can occur during the /proc reads. Reap the same Popen first.
        rc=self.proc.poll()
        if rc is not None:return self.exited(rc,snapshot)
        if snapshot is not None:
            if (not snapshot.get('stable') or snapshot['pid']!=self.expected['pid'] or
                snapshot['starttick']!=self.expected['starttick'] or
                snapshot['ppid']!=self.owner['pid']):
                return self.fault('Live owned PID/starttick/PPID is not the registered direct child; never signal it',snapshot)
            pending=snapshot['state'] in ('Z','X','x') or snapshot['cmdline_empty']
            if not pending:
                if snapshot['cmdline']!=self.expected['cmdline']:
                    return self.fault('Live nonempty argv changed; never signal replacement',snapshot)
                self.pending_since=None
                self.event('RUNNING',snapshot,reason='Exact direct child PID/starttick/PPID/full argv')
                return dict(phase='RUNNING',returncode=None,snapshot=snapshot)
            reason='Kernel leader terminal or empty argv while own Popen remains unreaped'
        else:reason='Proc snapshot unavailable while own Popen remains unreaped: '+str(missing)
        self.pending_ever=True
        if self.pending_since is None:self.pending_since=self.clock()
        if self.clock()>=self.deadline():
            self.failure=self.failure or 'Actual own exit/release unresolved within120s cleanup phase'
            self.event('EXIT_RELEASE_TIMEOUT',snapshot,reason=self.failure)
            raise ExitReleaseTimeout(self.failure)
        self.event('EXIT_PENDING',snapshot,reason=reason,
                   cleanup_deadline_monotonic=self.deadline())
        return dict(phase='EXIT_PENDING',returncode=None,snapshot=snapshot)

    def begin_cleanup(self):
        if self.cleanup_started is None:self.cleanup_started=self.clock()

    def deadline(self):
        times=[x for x in (self.pending_since,self.exit_observed,self.cleanup_started) if x is not None]
        return min(times)+self.cleanup_seconds if times else float('inf')

    def can_signal(self):
        if self.failure is not None or self.proc.poll() is not None:return False
        try:value=self.observe()
        except (OwnershipViolation,ExitReleaseTimeout):return False
        return value['phase']=='RUNNING' and self.proc.poll() is None

    def closed(self,released):
        rc=self.proc.poll()
        if rc is not None:self.exited(rc)
        self.event('CLOSED',self.last_snapshot,returncode=rc,released=bool(released),
                   reason='Owning exit/release accounting finalized; no inferred OS exit')
        return dict(status='PASS_OWNING_LIFECYCLE_EXIT0_RELEASED' if rc==0 and released and self.failure is None else 'FAIL_CLOSED_OWNING_LIFECYCLE',
                    identity=self.expected,owner=self.owner,actual_returncode=rc,
                    actual_exited=rc is not None,released=bool(released),
                    ownership_or_exit_failure=self.failure,pending_observed=self.pending_ever,
                    pending_since_monotonic=self.pending_since,exit_observed_monotonic=self.exit_observed,
                    cleanup_budget_seconds=self.cleanup_seconds,no_short_exit_classification_timeout=True)
