from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    path = ROOT / "experiments/qwen_gdn/rotation/orthogonal_numerical_closure_v1/run_qwen_orthogonal_numerical_closure.py"
    spec = importlib.util.spec_from_file_location("qwen_numerical_closure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interface_audit_has_no_functional_key_recovery():
    audit = load_runner().interface_audit()
    assert audit["generic_r_rt_interface_fixed"] == "YES"
    assert audit["functional_recoveries"] == 0
    assert audit["audit_only_recoveries"] >= 1


def test_condition_paths_exclude_localization_duplicates():
    runner = load_runner()
    root = Path("previous")
    assert runner.condition_path(root, "hadamard") == root / "gpu0/condition_hadamard.json"
    assert runner.condition_path(root, "r2") == root / "gpu3/condition_r2.json"
