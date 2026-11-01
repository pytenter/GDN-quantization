import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_aime24_int8_rc_end_to_end_smoke_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LingKdaInt8RcSemanticsTest(unittest.TestCase):
    def test_int8_quantizer_uses_canonical_signed_symmetric_formula(self):
        runner = load_runner()
        state = torch.tensor([[[[-1.1, -0.2, 0.2, 1.1]]]], dtype=torch.float32)

        qdq, meta = runner.fake_quant_ling_state(state, "INT8_R128")

        expected_scale = torch.tensor([[[[1.1 / 127.0]]]], dtype=torch.float32)
        expected_codes = torch.round(state / expected_scale).clamp(-127, 127)
        self.assertEqual(meta["qrange"], [-127, 127])
        self.assertEqual(meta["zero_point"], 0)
        self.assertEqual(meta["rounding"], "torch.round")
        self.assertTrue(torch.equal(meta["codes"], expected_codes))
        self.assertTrue(torch.allclose(meta["scale"], expected_scale))
        self.assertTrue(torch.allclose(qdq, expected_codes * expected_scale))


    def test_r128_groups_each_key_side_row_across_value_axis_on_ling_bhkv_state(self):
        runner = load_runner()
        state = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
        state[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        state[0, 0, 1, :] = torch.tensor([10.0, 20.0, 30.0, 40.0])
        state[0, 0, 2, :] = torch.tensor([100.0, 200.0, 300.0, 400.0])

        _qdq, meta = runner.fake_quant_ling_state(state, "INT8_R128")

        self.assertEqual(list(meta["scale"].shape), [1, 1, 3, 1])
        self.assertTrue(torch.allclose(
            meta["scale"].flatten(),
            torch.tensor([4.0 / 127.0, 40.0 / 127.0, 400.0 / 127.0]),
        ))
        self.assertEqual(meta["ling_semantic_axis"], "Key-side / recurrent-state row orientation")
        self.assertEqual(meta["amax_dims"], [-1])


    def test_c128_groups_each_value_side_column_across_key_axis_on_ling_bhkv_state(self):
        runner = load_runner()
        state = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
        state[:, :, :, 0] = torch.tensor([[[1.0, 10.0, 100.0]]])
        state[:, :, :, 1] = torch.tensor([[[2.0, 20.0, 200.0]]])
        state[:, :, :, 2] = torch.tensor([[[3.0, 30.0, 300.0]]])
        state[:, :, :, 3] = torch.tensor([[[4.0, 40.0, 400.0]]])

        _qdq, meta = runner.fake_quant_ling_state(state, "INT8_C128")

        self.assertEqual(list(meta["scale"].shape), [1, 1, 1, 4])
        self.assertTrue(torch.allclose(
            meta["scale"].flatten(),
            torch.tensor([100.0 / 127.0, 200.0 / 127.0, 300.0 / 127.0, 400.0 / 127.0]),
        ))
        self.assertEqual(meta["ling_semantic_axis"], "Value-side / recurrent-state column orientation")
        self.assertEqual(meta["amax_dims"], [-2])

    def test_selected_configs_rejects_non_protocol_config(self):
        runner = load_runner()

        with self.assertRaises(ValueError):
            runner.selected_configs("INT8_R64")

        self.assertEqual(runner.selected_configs("INT8_C128,FP_STATE"), ["INT8_C128", "FP_STATE"])

    def test_formal_aime_default_max_new_tokens_is_32768(self):
        runner = load_runner()

        self.assertEqual(runner.AIME_MAX_NEW_TOKENS, 32768)

    def test_offline_runtime_env_is_explicit(self):
        runner = load_runner()
        old = {k: os.environ.get(k) for k in runner.offline_env_vars()}
        try:
            for key in runner.offline_env_vars():
                os.environ.pop(key, None)

            runner.configure_offline_runtime()

            for key in ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"]:
                self.assertEqual(os.environ.get(key), "1")
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_revision_source_path_can_be_overridden_for_remote_runs(self):
        old = os.environ.get("LING_ENV_SUMMARY_PATH")
        try:
            os.environ["LING_ENV_SUMMARY_PATH"] = "/tmp/ling-env-summary.txt"
            runner = load_runner()

            self.assertEqual(runner.revision_source_path(), Path("/tmp/ling-env-summary.txt"))
        finally:
            if old is None:
                os.environ.pop("LING_ENV_SUMMARY_PATH", None)
            else:
                os.environ["LING_ENV_SUMMARY_PATH"] = old

    def test_analyze_counts_formal_shard_jsonl_files(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner.RUN_DIR = tmp_path / "run"
            runner.RESULT_DIR = tmp_path / "results"
            runner.REPORT_DIR = tmp_path / "reports"
            runner.FINAL_SUMMARY_PATH = runner.RESULT_DIR / "summary.json"
            runner.FINAL_REPORT_PATH = runner.REPORT_DIR / "report.md"
            runner.RUN_DIR.mkdir(parents=True)
            gates = {
                "R128_SEMANTICS_GATE": "PASS",
                "C128_SEMANTICS_GATE": "PASS",
                "PREFILL_QUANTIZATION_GATE": "PASS",
                "DECODE_QUANTIZATION_GATE": "PASS",
                "FP_IDENTITY_GATE": "PASS",
                "INT8_R128_SANITY_GATE": "PASS",
                "INT8_C128_SANITY_GATE": "PASS",
                "AIME_DATASET_GATE": "PASS",
                "EVALUATOR_GATE": "PASS",
            }
            runner.save_json(runner.RUN_DIR / "stage0_gates.json", gates)
            runner.save_json(runner.RUN_DIR / "protocol.json", {"decoding_protocol": {"max_new_tokens": 32768}})

            for shard_id, problem_id in enumerate(["0", "1"]):
                rec = {
                    "config": "INT8_C128",
                    "problem_id": problem_id,
                    "problem_index": shard_id,
                    "seed": runner.SEED,
                    "generated_tokens": 10,
                    "correct": shard_id == 0,
                    "completed": True,
                    "truncated": False,
                    "nonfinite": False,
                    "runtime_error": False,
                }
                runner.append_jsonl(runner.config_to_file("INT8_C128", shard_id=shard_id, num_shards=2), rec)

            summary = runner.analyze(write_report_flag=False, formal_required=True)

            self.assertEqual(summary["INT8_C128_SUCCESS"], 2)
            self.assertEqual(summary["INT8_C128_CORRECT"], 1)
            self.assertEqual(summary["COMPLETED_GENERATIONS"], 2)


if __name__ == "__main__":
    unittest.main()
