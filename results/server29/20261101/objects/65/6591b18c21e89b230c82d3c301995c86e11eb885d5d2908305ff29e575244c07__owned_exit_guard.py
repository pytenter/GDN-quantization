"""CPU candidate only. No GPU admission, scientific dispatch, or source mutation."""
import subprocess

def owned_child_running(proc, identity, same_process, exit_grace_seconds=0.1):
    """Only the owning Popen can establish exit; never infer it from PID absence.

    The short wait classifies a lost-identity/exit race. It is bounded within the
    existing120s own-release policy; it is not a generation elapsed-time cap.
    A genuinely live changed identity still fails closed, without signalling it.
    """
    if proc.poll() is not None:
        return False
    if same_process(identity):
        return True
    try:
        proc.wait(timeout=exit_grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Owned child identity differs and owning Popen has not observed exit; never signal replacement') from exc
    return False
