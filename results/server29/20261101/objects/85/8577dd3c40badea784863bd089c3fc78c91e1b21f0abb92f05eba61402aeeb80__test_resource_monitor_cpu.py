"""CPU adversarial integration tests; fake telemetry never queries any GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from resource_monitor import (ResourceMonitor, ResourceFailure, PAIRS, MIB,
                              RUNNING_FREE, IDLE_FREE, observe_unreaped)

UUID = PAIRS[0]
PID = 700


class Clock:
    def __init__(self):
        self.now = 1_000_000_000_000

    def mono(self):
        return self.now

    def wall(self):
        return self.now + 1_700_000_000_000_000_000

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)


def identity(pid=PID):
    return {"status": "PASS", "pid": pid, "uid": os.geteuid(),
            "start_ticks": 90000, "boot_id": "boot-cpu-test", "ppid": os.getpid(),
            "state": "S", "comm": "python"}


class FakeChild:
    def __init__(self):
        self.pid, self.returncode, self.exited, self.exit_code = PID, None, False, 0
        self.wait_calls = 0

    def wait(self, timeout=None):
        if not self.exited:
            raise AssertionError("test attempted to wait on live fake child")
        self.wait_calls += 1
        self.returncode = self.exit_code
        return self.returncode


def wait_observer(child):
    if child.returncode is not None:
        return {"status": "UNKNOWN", "pid": child.pid, "reason": "already reaped"}
    if child.exited:
        return {"status": "EXITED_UNREAPED", "pid": child.pid, "returncode": child.exit_code}
    return {"status": "LIVE_UNREAPED", "pid": child.pid}


class FakeGraphicsPolicy:
    """A strict dependency fixture; production uses separately frozen policy."""
    def evidence(self, s, uuid):
        if s.get("graphics_identity_changed"):
            raise ResourceFailure("AUTHORIZED_GRAPHICS_IDENTITY_CHANGED")
        m = s["direct_NVML"]["memory_v2"]
        return {"free_bytes": m["free"], "allocated_bytes": m["used"],
                "reserved_bytes": m["reserved"], "graphics": [{"pid": 90, "start_ticks": 5}],
                "signed_unattributed_residual_bytes": m["used"] - 8 * MIB}

    def compute_union(self, s, uuid, owner):
        values = {p["pid"] for p in s["direct_NVML_processes"]["compute"]["processes"]}
        values.update(int(p["pid"]) for p in s["SMI_all_processes"]["processes"] if p["type"] != "G")
        values.update(p["pid"] for p in s["early_SMI_compute_processes"]["processes"])
        if values - (set() if owner is None else {owner["pid"]}):
            raise ResourceFailure("FOREIGN_COMPUTE")
        return values


class Sampler:
    def __init__(self, clock):
        self.clock, self.presence, self.mutate = clock, [False] * 3, None
        self.child, self.exit_during_sample, self.used, self.free = None, False, 8 * MIB, None
        self.count = 0

    def __call__(self, uuid):
        self.count += 1
        start = self.clock.now
        self.clock.now += 50_000_000
        def timing(i):
            a = (start + 1_000_000 + i * 10_000_000) / 1e9
            return {"started_monotonic": a, "completed_monotonic": a + .005,
                    "started_epoch": self.clock.wall() / 1e9,
                    "completed_epoch": self.clock.wall() / 1e9 + .005}
        # Actual collector order is CSV, NVML compute, XML.
        nvml, early, xml = self.presence
        used = self.used
        total, reserved = 24564 * MIB, 350 * MIB
        free = total - used - reserved if self.free is None else self.free
        if self.free is not None:
            used = total - free - reserved
        s = {"uuid": uuid,
             "target": [{"index": "0", "uuid": uuid, "name": "NVIDIA GeForce RTX 4090",
                         "total_MiB": "24564", "used_MiB": str(used / MIB), "GPU_util": "0"}],
             "physical": {"returncode": 0, "stdout": "complete original inventory", "stderr": ""},
             "compute": {"returncode": 0, "stdout": f"{uuid}, {PID}, 16000\n" if early else "",
                         "stderr": "", "query_timing": timing(0)},
             "SMI_XML": {"returncode": 0, "stdout": "<complete-original-xml/>", "stderr": "",
                         "query_timing": timing(2)},
             "direct_NVML": {"status": "PASS", "memory": {"total": total, "used": used + reserved, "free": free},
                 "memory_v2": {"total": total, "used": used, "free": free, "reserved": reserved},
                 "utilization": {"gpu": 0, "memory": 0}, "temperature": 40,
                 "power_mW": 20000, "pstate": 8, "SM_clock_MHz": 210},
             "direct_NVML_processes": {"compute": {"status": "PASS", "query_timing": timing(1),
                    "processes": [{"pid": PID, "usedGpuMemory": 16000 * MIB}] if nvml else []},
                    "graphics": {"status": "PASS", "processes": [{"pid": 90, "usedGpuMemory": 8 * MIB}]}},
             "SMI_all_processes": {"status": "PASS", "processes": [
                 {"pid": "90", "type": "G", "used_memory": "8 MiB"}] +
                 ([{"pid": str(PID), "type": "C", "used_memory": "16000 MiB"}] if xml else [])},
             "early_SMI_compute_processes": {"status": "PASS", "processes":
                 [{"uuid": uuid, "pid": PID, "used_gpu_memory": "16000"}] if early else []},
             "process_ownership": [identity()] if any(self.presence) else []}
        if self.mutate:
            self.mutate(s)
        if self.exit_during_sample:
            self.child.exited = True
        return s


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock, self.fake = Clock(), FakeChild()
        self.sampler = Sampler(self.clock)
        self.sampler.child = self.fake
        self.actual = identity()
        self.monitor = ResourceMonitor(self.root / "gpu0", 0, UUID, self.sampler, FakeGraphicsPolicy(),
            stop_path=self.root / "FIRST_FAILURE.json", proc_reader=lambda pid: copy.deepcopy(self.actual),
            exit_observer=wait_observer, monotonic_ns=self.clock.mono, wall_ns=self.clock.wall,
            sleep=self.clock.sleep)

    def tearDown(self):
        self.temp.cleanup()

    def register(self):
        ref = self.monitor.admit("job")
        return self.monitor.register_child(self.fake, "job", ref,
            spawn_monotonic_ns=self.clock.mono(), spawn_wall_ns=self.clock.wall())

    def running(self):
        child = self.register()
        self.sampler.presence = [True] * 3
        self.sampler.used = 16008 * MIB
        self.assertTrue(self.monitor.observe(child)["resource_pass"])
        return child

    def valid_result(self, child):
        path = self.root / "actual_result.json"
        path.write_text('{"status":"PASS","actual":"CPU fixture"}')
        return {"status": "PASS", "source_verified": True, "numerical_pass": True,
                "profile_verified": True, "environment_verified": True, "gpu_uuid": UUID,
                "evidence_refs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}

    def mark_failed(self):
        with self.assertRaises(ResourceFailure): self.monitor.fail("CPU fixture global failure")

    def test_admission_three_spaced_raw_samples_sha_bound(self):
        ref = self.monitor.admit("job")
        receipt = json.loads(Path(ref["path"]).read_text())
        self.assertEqual(receipt["sample_count"], 3)
        self.assertEqual(self.sampler.count, 3)
        self.assertEqual(hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest(), ref["sha256"])
        for item in receipt["raw_refs"]:
            data = Path(item["path"]).read_bytes()[item["offset"]:item["offset"] + item["length"]]
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])

    def test_foreign_in_any_source_including_desktop_stops_before_init_explanation(self):
        for api in ("NVML", "CSV", "XML"):
            with self.subTest(api=api):
                if self.monitor.stop_path.exists():
                    self.monitor.stop_path.unlink()  # isolated fixture cases only
                def mutate(s):
                    if api == "NVML":
                        s["direct_NVML_processes"]["compute"]["processes"] = [{"pid": 90}]
                    elif api == "CSV":
                        s["compute"]["stdout"] = f"{UUID}, 90, 8\n"
                        s["early_SMI_compute_processes"]["processes"] = [{"uuid": UUID, "pid": 90, "used_gpu_memory": "8"}]
                    else:
                        s["SMI_all_processes"]["processes"].append({"pid": "90", "type": "C+G"})
                self.sampler.mutate = mutate
                with self.assertRaisesRegex(ResourceFailure, "FOREIGN_COMPUTE"):
                    self.monitor.admit("foreign_" + api)
                stop = json.loads(self.monitor.stop_path.read_text())
                ref = stop["evidence_ref"]
                data = Path(ref["path"]).read_bytes()[ref["offset"]:ref["offset"] + ref["length"]]
                self.assertEqual(hashlib.sha256(data).hexdigest(), ref["sha256"])

    def test_first_failure_is_immutable_and_blocks_new_launch(self):
        with self.assertRaises(ResourceFailure):
            self.monitor.fail("first")
        original = self.monitor.stop_path.read_bytes()
        with self.assertRaises(ResourceFailure):
            self.monitor.fail("second")
        self.assertEqual(self.monitor.stop_path.read_bytes(), original)
        with self.assertRaises(ResourceFailure):
            self.monitor.admit("new")
        self.assertEqual(self.sampler.count, 0)

    def test_api_unknown_even_one_source_stops(self):
        for field in ("physical", "compute", "SMI_XML"):
            with self.subTest(field=field):
                if self.monitor.stop_path.exists(): self.monitor.stop_path.unlink()
                self.sampler.mutate = lambda s, key=field: s[key].update(returncode=None)
                with self.assertRaisesRegex(ResourceFailure, "UNKNOWN_SAFETY_API"):
                    self.monitor.admit("unknown_" + field)

    def test_actual_identity_change_cannot_be_filled_from_intent(self):
        child = self.register()
        self.sampler.presence = [True] * 3
        self.actual["start_ticks"] += 1
        with self.assertRaisesRegex(ResourceFailure, "ACTUAL_IDENTITY"):
            self.monitor.observe(child)

    def test_live_owner_raw_unknown_is_not_waived(self):
        child = self.register()
        self.sampler.presence = [True] * 3
        self.sampler.mutate = lambda s: s.update(process_ownership=[{"status": "UNKNOWN", "pid": PID}])
        with self.assertRaisesRegex(ResourceFailure, "UNKNOWN_OWN_RAW_IDENTITY_WHILE_LIVE"):
            self.monitor.observe(child)

    def test_initialization_mismatch_defers_then_all_own_resolves(self):
        child = self.register()
        self.sampler.presence = [False, False, True]
        value = self.monitor.observe(child)
        self.assertFalse(value["resource_pass"])
        self.assertEqual(len(self.monitor.pending), 1)
        self.clock.sleep(5)
        self.sampler.presence = [True] * 3
        self.assertTrue(self.monitor.observe(child)["resource_pass"])
        self.assertEqual(self.monitor.pending, [])
        receipt = json.loads((self.monitor.directory / "derived/job.INIT_RECONCILIATION.json").read_text())
        self.assertFalse(receipt["resource_pass"])

    def test_more_than_two_unresolved_samples_stops(self):
        child = self.register()
        self.sampler.presence = [False, False, True]
        for _ in range(2):
            self.assertFalse(self.monitor.observe(child)["resource_pass"])
            self.clock.sleep(5)
        with self.assertRaisesRegex(ResourceFailure, "PENDING_LIMIT"):
            self.monitor.observe(child)

    def test_bounded_neighbor_time_and_initialization_timeout(self):
        child = self.register()
        self.sampler.presence = [False, False, True]
        self.monitor.observe(child)
        self.clock.sleep(20)
        self.sampler.presence = [True] * 3
        with self.assertRaisesRegex(ResourceFailure, "UNBOUNDED"):
            self.monitor.observe(child)

    def test_empty_pid_lists_do_not_erase_material_memory(self):
        child = self.register()
        self.sampler.used = 100 * MIB
        with self.assertRaisesRegex(ResourceFailure, "MATERIAL_MEMORY"):
            self.monitor.observe(child)

    def test_true_free_running_and_idle_limits_unchanged(self):
        def low_free(s):
            m = s["direct_NVML"]["memory_v2"]
            m["free"] = IDLE_FREE - 1
            m["reserved"] = m["total"] - m["used"] - m["free"]
            s["direct_NVML"]["memory"]["free"] = m["free"]
            s["direct_NVML"]["memory"]["used"] = m["used"] + m["reserved"]
        self.sampler.mutate = low_free
        with self.assertRaisesRegex(ResourceFailure, "IDLE_TRUE_FREE"):
            self.monitor.admit("low")

    def test_running_true_free_below_two_gib_stops(self):
        child = self.register()
        self.sampler.free = RUNNING_FREE - 1
        self.sampler.presence = [True] * 3
        with self.assertRaisesRegex(ResourceFailure, "TRUE_FREE_BELOW_2GIB"):
            self.monitor.observe(child)

    def test_negative_residual_graphics_change_reserved_unknown_fail_closed(self):
        for name in ("negative", "graphics", "reserved"):
            with self.subTest(name=name):
                if self.monitor.stop_path.exists(): self.monitor.stop_path.unlink()
                def mutate(s):
                    if name == "negative":
                        s["direct_NVML"]["memory_v2"]["used"] = -100 * MIB
                    elif name == "graphics":
                        s["graphics_identity_changed"] = True
                    else:
                        s["direct_NVML"]["memory_v2"]["reserved"] = None
                self.sampler.mutate = mutate
                with self.assertRaises(ResourceFailure): self.monitor.admit(name)

    def test_strict_util_rule_not_relaxed(self):
        self.sampler.mutate = lambda s: s["target"][0].update(GPU_util="3")
        with self.assertRaisesRegex(ResourceFailure, "SUSTAINED_OR_CURRENT"):
            self.monitor.admit("busy")

    def test_natural_exit_raw_unknown_derived_unreaped_binding_only(self):
        child = self.running()
        self.clock.sleep(5)
        self.sampler.presence = [False, True, True]  # early CSV own; later NVML absent; XML own => impossible exit
        self.sampler.exit_during_sample = True
        self.sampler.mutate = lambda s: s.update(process_ownership=[{"pid": PID, "status": "UNKNOWN"}])
        value = self.monitor.observe(child)
        self.assertFalse(value["resource_pass"])
        raw = json.loads(Path(value["raw_ref"]["path"]).read_bytes()[value["raw_ref"]["offset"]:][:value["raw_ref"]["length"]])
        self.assertEqual(raw["snapshot"]["process_ownership"][0]["status"], "UNKNOWN")
        self.sampler.presence, self.sampler.used, self.sampler.mutate = [False] * 3, 8 * MIB, None
        with self.assertRaisesRegex(ResourceFailure, "IMPOSSIBLE_SINGLE_TRANSITION"):
            self.monitor.finish(child, self.valid_result)
        self.assertEqual(self.fake.wait_calls, 1)
        self.assertEqual(self.monitor.jobs[-1]["status"], "FAILED_NOT_QUALIFIED")

    def test_successful_exit_needs_wait_result_and_terminal_and_full_cost(self):
        child = self.running()
        self.clock.sleep(5)
        self.sampler.presence = [False, True, False]  # CSV own first, later two absent.
        self.sampler.exit_during_sample = True
        value = self.monitor.observe(child)
        self.assertFalse(value["resource_pass"])
        self.sampler.presence, self.sampler.used = [False] * 3, 8 * MIB
        done = self.monitor.finish(child, self.valid_result)
        self.assertEqual(done["status"], "PASS")
        self.assertEqual(self.fake.wait_calls, 1)
        self.assertIsNone(self.monitor.active)
        self.assertGreater(done["job"]["charged_seconds"], done["job"]["pure_spawn_to_wait_seconds"] + 9.9)
        self.assertIsNone(done["job"]["gpu_time_caps"])
        self.assertEqual(done["job"]["NVML_sampled_owned_peak_bytes"], 16000 * MIB)

    def test_result_truthiness_does_not_qualify_reaped_job(self):
        child = self.running()
        self.fake.exited = True
        with self.assertRaisesRegex(ResourceFailure, "RESULT_BINDING_NOT_PASS"):
            self.monitor.finish(child, lambda child: {"status": "UNKNOWN"})
        self.assertTrue(self.monitor.failed)
        self.assertIsNone(self.monitor.active)
        self.assertEqual(self.monitor.jobs[-1]["status"], "FAILED_NOT_QUALIFIED")

    def test_exit_cannot_close_an_unbounded_gap_after_last_live_sample(self):
        child = self.running()
        self.clock.sleep(20)
        self.fake.exited = True
        with self.assertRaisesRegex(ResourceFailure, "UNBOUNDED_EXIT_OBSERVATION"):
            self.monitor.finish(child, self.valid_result)
        self.assertEqual(self.fake.wait_calls, 0)
        self.assertTrue(self.monitor.failed)

    def test_no_finish_before_natural_exit_or_duplicate_launch(self):
        child = self.running()
        with self.assertRaisesRegex(ResourceFailure, "FINISH_REQUIRES"):
            self.monitor.finish(child, self.valid_result)
        self.assertEqual(self.fake.wait_calls, 0)
        with self.assertRaisesRegex(ResourceFailure, "EXISTING_FIRST_FAILURE"):
            self.monitor.admit("another")

    def test_overlapping_launch_latches_before_second_admission(self):
        self.running()
        count = self.sampler.count
        with self.assertRaisesRegex(ResourceFailure, "OVERLAPPING"):
            self.monitor.admit("another")
        self.assertEqual(self.sampler.count, count)

    def test_initialization_deadline_is_not_gpu_hour_budget(self):
        child = self.register()
        self.clock.sleep(181)
        with self.assertRaisesRegex(ResourceFailure, "UNBOUNDED|INITIALIZATION_BOUND"):
            self.monitor.observe(child)
        self.assertIsNone(self.monitor.accounting()["gpu_time_caps"])

    def test_nan_negative_and_overlapping_query_times_stop_with_raw_retained(self):
        for value in (float("nan"), -1., "missing"):
            with self.subTest(value=value):
                if self.monitor.stop_path.exists(): self.monitor.stop_path.unlink()
                self.sampler.mutate = lambda s: s["direct_NVML_processes"]["compute"]["query_timing"].update(started_monotonic=value)
                with self.assertRaisesRegex(ResourceFailure, "QUERY_TIMESTAMPS"):
                    self.monitor.admit("time_" + str(value))
                self.assertGreater((self.monitor.directory / "RESOURCE_RAW.jsonl").stat().st_size, 0)

    def test_cleanup_second_signal_rechecks_actual_identity(self):
        child = self.running()
        self.mark_failed()
        calls = []
        def waits(timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.fake.returncode = -9
            return -9
        self.fake.wait = waits
        with patch("resource_monitor.os.kill") as kill:
            result = self.monitor.terminate_exact(child)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [15, 9])
        self.assertEqual(result["job"]["returncode"], -9)
        self.assertEqual(calls, [30, 30])

    def test_cleanup_changed_identity_after_term_never_kills_unknown_pid(self):
        child = self.running()
        self.mark_failed()
        def waits(timeout=None):
            self.actual["start_ticks"] += 1
            raise subprocess.TimeoutExpired("fake", timeout)
        self.fake.wait = waits
        with patch("resource_monitor.os.kill") as kill:
            with self.assertRaisesRegex(ResourceFailure, "AFTER_TERM_NO_KILL"):
                self.monitor.terminate_exact(child)
        self.assertEqual(len(kill.call_args_list), 1)
        self.assertIs(self.monitor.active, child)

    def test_cleanup_timeout_is_unknown_not_released_or_success(self):
        child = self.running()
        self.mark_failed()
        def waits(timeout=None):
            raise subprocess.TimeoutExpired("fake", timeout)
        self.fake.wait = waits
        with patch("resource_monitor.os.kill") as kill:
            result = self.monitor.terminate_exact(child)
        self.assertEqual(len(kill.call_args_list), 2)
        self.assertEqual(result["status"], "OWN_CHILD_RELEASE_UNKNOWN_REQUIRES_USER")
        self.assertFalse(result["worker_reaped"])
        self.assertIs(self.monitor.active, child)
        self.assertFalse(child.closed)

    def test_cleanup_cannot_bypass_first_failure_then_launch_more_work(self):
        child = self.running()
        with patch("resource_monitor.os.kill") as kill:
            with self.assertRaisesRegex(ResourceFailure, "PERSISTED_FIRST_FAILURE"):
                self.monitor.terminate_exact(child)
        self.assertFalse(kill.called)
        self.assertTrue(self.monitor.failed)
        with self.assertRaises(ResourceFailure): self.monitor.admit("new")

    def test_empty_then_reappearing_mixed_exit_cannot_be_reconciled(self):
        child = self.running()
        self.clock.sleep(5)
        self.sampler.presence, self.sampler.used = [False] * 3, 8 * MIB
        self.assertFalse(self.monitor.observe(child)["resource_pass"])
        self.clock.sleep(5)
        self.sampler.presence = [False, True, False]
        self.sampler.exit_during_sample = True
        self.assertFalse(self.monitor.observe(child)["resource_pass"])
        self.sampler.presence = [False] * 3
        with self.assertRaisesRegex(ResourceFailure, "REAPPEARED_ACROSS_PENDING"):
            self.monitor.finish(child, self.valid_result)

    def test_completed_and_active_time_account_once(self):
        child = self.running()
        self.clock.sleep(4)
        active = self.monitor.accounting()
        self.assertEqual(active["completed_charged_seconds"], 0)
        self.assertGreater(active["active_charged_seconds"], 4)
        self.fake.exited = True
        self.sampler.presence, self.sampler.used = [False] * 3, 8 * MIB
        result = self.monitor.finish(child, self.valid_result)
        final = self.monitor.accounting()
        self.assertEqual(final["active_charged_seconds"], 0)
        self.assertEqual(final["completed_charged_seconds"], result["job"]["charged_seconds"])
        self.assertEqual(final["prior_gpu_hours"], 0)

    def test_wrong_pair_rejected_without_gpu_query(self):
        with self.assertRaises(ResourceFailure):
            ResourceMonitor(self.root / "bad", 1, UUID, self.sampler, FakeGraphicsPolicy())
        self.assertEqual(self.sampler.count, 0)

    def test_stale_admission_cannot_authorize_late_child(self):
        ref = self.monitor.admit("job")
        self.clock.sleep(20)
        with self.assertRaisesRegex(ResourceFailure, "REGISTRATION_STALE"):
            self.monitor.register_child(self.fake, "job", ref,
                spawn_monotonic_ns=self.clock.mono(), spawn_wall_ns=self.clock.wall())

    def test_query_interval_missing_or_overlap_is_unknown(self):
        child = self.register()
        self.sampler.presence = [True] * 3
        self.sampler.mutate = lambda s: s["direct_NVML_processes"]["compute"].pop("query_timing")
        with self.assertRaisesRegex(ResourceFailure, "QUERY_TIMESTAMPS"):
            self.monitor.observe(child)

    @unittest.skipUnless(hasattr(os, "waitid") and hasattr(os, "WNOWAIT"), "Linux waitid WNOWAIT required")
    def test_actual_cpu_child_waitid_does_not_reap(self):
        child = subprocess.Popen([sys.executable, "-B", "-c", "pass"])
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            observed = observe_unreaped(child)
            if observed["status"] == "EXITED_UNREAPED": break
            time.sleep(.01)
        self.assertEqual(observed["status"], "EXITED_UNREAPED")
        self.assertIsNone(child.returncode)
        self.assertEqual(child.wait(), 0)
        self.assertEqual(observe_unreaped(child)["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
