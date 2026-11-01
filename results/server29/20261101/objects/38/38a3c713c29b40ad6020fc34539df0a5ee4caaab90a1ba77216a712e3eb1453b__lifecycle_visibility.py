"""CPU-only visibility subgate for the NEW experiment, never a resource PASS.

This module does not query devices or initialize CUDA. Full resource admission
also needs memory/graphics checks, raw-byte binding and legal lifecycle gates.
Old evidence and its classifications are not modified by this validator.
"""
from dataclasses import dataclass


APIS = frozenset({"NVML", "SMI_COMPUTE_CSV", "SMI_XML"})


@dataclass(frozen=True)
class Owner:
    boot_id: str
    pid: int
    start_ticks: int
    uid: int
    gpu_uuid: str


def inspect_sample(sample, owner):
    """Validate raw-extracted API intervals and every observed compute identity.

    Returns status plus ordered presence observations. Input identities must
    come from independently captured /proc evidence, not copied intent fields.
    """
    if type(owner.pid) is not int or owner.pid <= 0 or \
       type(owner.start_ticks) is not int or owner.start_ticks <= 0 or \
       type(owner.uid) is not int or owner.uid < 0 or \
       not isinstance(owner.boot_id, str) or not owner.boot_id or \
       not isinstance(owner.gpu_uuid, str) or not owner.gpu_uuid:
        return "UNKNOWN_REGISTERED_OWNER", []
    queries = sample.get("queries", [])
    if len(queries) != 3 or {q.get("api") for q in queries} != APIS:
        return "UNKNOWN_API_COVERAGE", []
    observed = []
    for query in queries:
        if query.get("success") is not True or query.get("gpu_uuid") != owner.gpu_uuid:
            return "UNKNOWN_API_ERROR_OR_UUID", []
        begin, end = query.get("begin_ns"), query.get("end_ns")
        if type(begin) is not int or type(end) is not int or begin < 0 or begin > end:
            return "UNKNOWN_QUERY_TIMESTAMPS", []
        rows = query.get("compute_rows")
        if not isinstance(rows, list):
            return "UNKNOWN_COMPUTE_ROWS", []
        present = False
        for row in rows:
            if not isinstance(row, dict) or type(row.get("pid")) is not int or row["pid"] <= 0:
                return "UNKNOWN_COMPUTE_PID_STOP", []
            if row.get("pid") != owner.pid:
                return "FOREIGN_COMPUTE_STOP", []
            if type(row.get("start_ticks")) is not int or type(row.get("uid")) is not int:
                return "UNKNOWN_OWN_IDENTITY_STOP", []
            expected = {"boot_id": owner.boot_id, "pid": owner.pid,
                        "start_ticks": owner.start_ticks, "uid": owner.uid}
            if any(row.get(key) != value for key, value in expected.items()):
                return "UNKNOWN_OWN_IDENTITY_STOP", []
            present = True
        observed.append((begin, end, present))
    return "VISIBILITY_OBSERVED", sorted(observed)


def reconcile_triplet(previous, disputed, following, owner, phase,
                      lifecycle_begin_ns, lifecycle_end_ns,
                      live_identity_before=False, live_identity_after=False,
                      legal_wait_rc=None, terminal_window_pass=False,
                      actual_result_valid=False, max_adjacent_gap_ns=15_000_000_000):
    """Reconcile one bounded API disagreement using neighboring raw samples.

    A success concerns visibility only. It never waives the memory, graphics,
    numerical, result, terminal or source-identity gates. Failed/unknown source
    data anywhere in this triplet fails closed. Monotonic timestamps required.
    """
    groups = []
    for sample in (previous, disputed, following):
        status, rows = inspect_sample(sample, owner)
        if status != "VISIBILITY_OBSERVED":
            return {"status": status, "resource_pass": False}
        groups.append(rows)
    before, current, after = groups
    if type(max_adjacent_gap_ns) is not int or max_adjacent_gap_ns <= 0:
        return {"status": "UNKNOWN_MONITOR_INTERVAL_CONTRACT", "resource_pass": False}
    if max(q[1] for q in before) > min(q[0] for q in current) or \
       max(q[1] for q in current) > min(q[0] for q in after):
        return {"status": "UNKNOWN_SAMPLE_ORDER", "resource_pass": False}
    if min(q[0] for q in current) - max(q[1] for q in before) > max_adjacent_gap_ns or \
       min(q[0] for q in after) - max(q[1] for q in current) > max_adjacent_gap_ns:
        return {"status": "UNKNOWN_UNBOUNDED_NEIGHBOR_INTERVAL", "resource_pass": False}
    if len({q[2] for q in current}) != 2:
        return {"status": "NOT_A_DISAGREEMENT", "resource_pass": False}
    if type(lifecycle_begin_ns) is not int or type(lifecycle_end_ns) is not int or \
       lifecycle_begin_ns > lifecycle_end_ns:
        return {"status": "UNKNOWN_LIFECYCLE_INTERVAL", "resource_pass": False}
    if phase == "INITIALIZING":
        if live_identity_before is not True or live_identity_after is not True:
            return {"status": "UNKNOWN_LIVE_IDENTITY", "resource_pass": False}
        if any(q[2] for q in before) or not all(q[2] for q in after):
            return {"status": "UNKNOWN_INITIALIZATION_NEIGHBORS", "resource_pass": False}
        # An API call may overlap the transition. A missing PID implies the
        # context transition may be later than its start, not necessarily end.
        lower = max(lifecycle_begin_ns, max(q[0] for q in current if not q[2]))
        upper = min(lifecycle_end_ns, min(q[1] for q in current if q[2]))
        label = "OWN_INITIALIZATION_VISIBILITY_RECONCILED"
    elif phase == "EXITED":
        if type(legal_wait_rc) is not int or legal_wait_rc != 0 or \
           terminal_window_pass is not True or actual_result_valid is not True:
            return {"status": "UNKNOWN_EXIT_CLOSURE", "resource_pass": False}
        if not all(q[2] for q in before) or any(q[2] for q in after):
            return {"status": "UNKNOWN_EXIT_NEIGHBORS", "resource_pass": False}
        lower = max(lifecycle_begin_ns, max(q[0] for q in current if q[2]))
        upper = min(lifecycle_end_ns, min(q[1] for q in current if not q[2]))
        label = "OWN_NATURAL_EXIT_VISIBILITY_RECONCILED"
    else:
        return {"status": "UNKNOWN_NONBOUNDARY_DISAGREEMENT", "resource_pass": False}
    if lower > upper:
        return {"status": "UNKNOWN_IMPOSSIBLE_SINGLE_TRANSITION", "resource_pass": False}
    return {"status": label, "transition_feasible_interval_ns": [lower, upper],
            "resource_pass": False,
            "scope": "API visibility subgate only; all other job gates still required"}
