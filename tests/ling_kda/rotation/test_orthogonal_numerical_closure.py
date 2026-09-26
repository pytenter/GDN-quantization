from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    path = ROOT / "experiments/ling_kda/rotation/orthogonal_numerical_closure_v1/run_ling_orthogonal_numerical_closure.py"
    spec = importlib.util.spec_from_file_location("ling_numerical_closure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interface_audit_has_exactly_one_functional_recovery():
    audit = load_runner().interface_audit()
    assert audit["generic_r_rt_interface_fixed"] == "YES"
    assert audit["functional_recoveries"] == 1
    assert audit["audit_only_recoveries"] == 3
    assert audit["redundant_prefill_endpoint_rotation"] == "NO"


def test_condition_paths_exclude_localization_duplicates():
    runner = load_runner()
    root = Path("previous")
    assert runner.condition_path(root, "r0") == root / "gpu0/condition_r0.json"
    assert runner.condition_path(root, "r2") == root / "gpu1/condition_r2.json"
