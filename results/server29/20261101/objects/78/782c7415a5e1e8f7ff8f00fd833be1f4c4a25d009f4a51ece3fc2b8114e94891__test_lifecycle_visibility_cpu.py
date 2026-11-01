"""Adversarial boundary tests; no model import, no GPU or CUDA initialization."""
import copy
import unittest
from lifecycle_visibility import Owner, inspect_sample, reconcile_triplet


OWNER = Owner("boot-test", 700, 90000, 1006, "GPU-test")


def sample(start, presence):
    rows = []
    for i, (api, visible) in enumerate(zip(("NVML", "SMI_COMPUTE_CSV", "SMI_XML"), presence)):
        rows.append(dict(api=api, success=True, gpu_uuid=OWNER.gpu_uuid,
                         begin_ns=start + i * 10, end_ns=start + i * 10 + 5,
                         compute_rows=[dict(boot_id=OWNER.boot_id, pid=OWNER.pid,
                                            start_ticks=OWNER.start_ticks, uid=OWNER.uid)] if visible else []))
    return dict(queries=rows)


class VisibilityTests(unittest.TestCase):
    def reconcile(self, phase="INITIALIZING", **changes):
        init = phase == "INITIALIZING"
        args = dict(previous=sample(0, [not init] * 3),
                    disputed=sample(100, [not init, not init, init]),
                    following=sample(200, [init] * 3), owner=OWNER, phase=phase,
                    lifecycle_begin_ns=100, lifecycle_end_ns=125,
                    live_identity_before=True, live_identity_after=True,
                    legal_wait_rc=0, terminal_window_pass=True, actual_result_valid=True)
        args.update(changes)
        return reconcile_triplet(**args)

    def test_own_initialization_has_bounded_explanation_but_is_not_resource_pass(self):
        result = self.reconcile()
        self.assertEqual(result["status"], "OWN_INITIALIZATION_VISIBILITY_RECONCILED")
        self.assertEqual(result["transition_feasible_interval_ns"], [110, 125])
        self.assertFalse(result["resource_pass"])

    def test_natural_exit_needs_legal_wait_result_and_terminal_closure(self):
        self.assertEqual(self.reconcile("EXITED")["status"], "OWN_NATURAL_EXIT_VISIBILITY_RECONCILED")
        for change in ({"legal_wait_rc": 1}, {"terminal_window_pass": False}, {"actual_result_valid": False}):
            self.assertEqual(self.reconcile("EXITED", **change)["status"], "UNKNOWN_EXIT_CLOSURE")

    def test_foreign_compute_even_in_one_late_source_stops(self):
        changed = sample(100, [False, False, True])
        changed["queries"][2]["compute_rows"][0]["pid"] = 701
        self.assertEqual(self.reconcile(disputed=changed)["status"], "FOREIGN_COMPUTE_STOP")

    def test_same_pid_reuse_or_owner_change_cannot_be_accepted(self):
        for key, value in (("start_ticks", 99999), ("uid", 1008), ("boot_id", "other-boot")):
            changed = sample(100, [False, False, True])
            changed["queries"][2]["compute_rows"][0][key] = value
            self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_OWN_IDENTITY_STOP")

    def test_missing_timestamps_old_samples_cannot_be_retroactively_passed(self):
        changed = sample(100, [False, False, True])
        del changed["queries"][0]["begin_ns"]
        self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_QUERY_TIMESTAMPS")

    def test_api_error_wrong_uuid_or_missing_source_is_unknown(self):
        for key, value in (("success", False), ("gpu_uuid", "GPU-wrong")):
            changed = sample(100, [False, False, True])
            changed["queries"][0][key] = value
            self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_API_ERROR_OR_UUID")
        changed["queries"].pop()
        self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_API_COVERAGE")

    def test_impossible_disappearing_then_reappearing_init_remains_unknown(self):
        changed = sample(100, [True, False, True])
        self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_IMPOSSIBLE_SINGLE_TRANSITION")

    def test_unstable_neighbors_or_steady_state_mismatch_remain_unknown(self):
        self.assertEqual(self.reconcile(following=sample(200, [True, False, True]))["status"], "UNKNOWN_INITIALIZATION_NEIGHBORS")
        self.assertEqual(self.reconcile(phase="RUNNING")["status"], "UNKNOWN_NONBOUNDARY_DISAGREEMENT")

    def test_no_mutation_of_raw_extracted_evidence(self):
        original = sample(100, [False, False, True])
        before = copy.deepcopy(original)
        self.reconcile(disputed=original)
        self.assertEqual(original, before)

    def test_unknown_strings_and_boolean_exit_code_do_not_mean_true_or_zero(self):
        for change in ({"live_identity_before": "UNKNOWN"}, {"live_identity_after": 1}):
            self.assertEqual(self.reconcile(**change)["status"], "UNKNOWN_LIVE_IDENTITY")
        for change in ({"legal_wait_rc": False}, {"terminal_window_pass": "UNKNOWN"},
                       {"actual_result_valid": "UNKNOWN"}):
            self.assertEqual(self.reconcile("EXITED", **change)["status"], "UNKNOWN_EXIT_CLOSURE")

    def test_distant_followup_or_invalid_gap_contract_stops_reconciliation(self):
        self.assertEqual(self.reconcile(following=sample(10**15, [True]*3))["status"],
                         "UNKNOWN_UNBOUNDED_NEIGHBOR_INTERVAL")
        self.assertEqual(self.reconcile(max_adjacent_gap_ns=False)["status"],
                         "UNKNOWN_MONITOR_INTERVAL_CONTRACT")

    def test_invalid_pid_is_unknown_rather_than_proven_foreign(self):
        for pid in (None, "700", 700.0, False, 0, -1):
            changed = sample(100, [False, False, True])
            changed["queries"][2]["compute_rows"][0]["pid"] = pid
            self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_COMPUTE_PID_STOP")
        changed = sample(100, [False, False, True])
        changed["queries"][2]["compute_rows"][0]["uid"] = float(OWNER.uid)
        self.assertEqual(self.reconcile(disputed=changed)["status"], "UNKNOWN_OWN_IDENTITY_STOP")


if __name__ == "__main__":
    unittest.main()
