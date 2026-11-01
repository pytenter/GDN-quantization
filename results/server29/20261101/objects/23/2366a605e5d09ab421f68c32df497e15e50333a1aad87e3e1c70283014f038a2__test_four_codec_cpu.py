"""New wiring/math tests only. CPU callbacks are not real-model/GPU qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
import numpy as np
from four_codec_boundary import FourCodecBoundary, ARMS, stepwise_prefill
from canonical_codec import encode as asym_encode
from independent_reference import canonical_group
from matched_baseline import build_reference_encoder


def hmatrix(dtype=torch.float32):
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / (128 ** .5)


class BoundaryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original = {n: getattr(torch.cuda, n) for n in ['is_available', 'device_count', '_lazy_init']}
        def fail(*a, **k): raise AssertionError('CPU_TEST_MUST_NOT_DISCOVER_OR_INITIALIZE_CUDA')
        for n in cls.original: setattr(torch.cuda, n, fail)

    @classmethod
    def tearDownClass(cls):
        for n, f in cls.original.items(): setattr(torch.cuda, n, f)

    def test_01_exact_task_matrix(self):
        self.assertEqual([(a.hadamard, a.asymmetric) for a in ARMS.values()],
                         [(False, False), (False, True), (True, False), (True, True)])

    def test_02_noH_returns_exact_input_without_GEMM(self):
        x = torch.randn(1, 1, 2, 128);x.flatten()[0] = -0.
        for arm in ['B0', 'B1']:
            a = FourCodecBoundary(torch, arm, hmatrix())
            self.assertIs(a.enter_value(x), x);self.assertIs(a.restore_core(x), x)

    def test_03_H_orthogonal_and_unquantized_roundtrip(self):
        h = hmatrix(torch.float64);x = torch.sin(torch.arange(128, dtype=torch.float64))[None]
        self.assertLess(float((h.T @ h - torch.eye(128, dtype=torch.float64)).abs().max()), 1.e-12)
        self.assertLess(float(((x @ h) @ h.T - x).abs().max()), 1.e-12)
        hf = hmatrix();self.assertLess(float((hf.T @ hf - torch.eye(128)).abs().max()), 1.e-6)

    def test_04_Key_gate_H_equivariance_in_independent_float64_recurrence(self):
        h = hmatrix(torch.float64);state = torch.zeros(128, 128, dtype=torch.float64);rotated = state.clone()
        gen = torch.Generator().manual_seed(20261020)
        for _ in range(4):
            q, k, v = [torch.randn(128, dtype=torch.float64, generator=gen) for _ in range(3)]
            k = k / k.norm();q = q / q.norm()
            gate = torch.sigmoid(torch.randn(128, dtype=torch.float64, generator=gen))
            # Original Key-channel diagonal gate and delta update; H touches Value only.
            state = state * gate[:, None]
            state = state + .3 * k[:, None] * (v - k @ state)[None, :]
            rotated = rotated * gate[:, None]
            rotated = rotated + .3 * k[:, None] * (v @ h - k @ rotated)[None, :]
            self.assertLess(float(((q @ rotated) @ h.T - q @ state).abs().max()), 1.e-12)

    def test_05_native_Sym32_matches_sealed_group_reference(self):
        x = torch.sin(torch.arange(128 * 128, dtype=torch.float32)).reshape(1, 1, 128, 128) * .01
        pool = x.transpose(-1, -2).contiguous();idx = torch.tensor([0])
        a = FourCodecBoundary(torch, 'B0', hmatrix());out = a.after_native_kernel(pool, idx, 0, 0)
        ref = build_reference_encoder(torch)(x, 32, False)
        for field in ['codes', 'scales']:
            self.assertTrue(torch.equal(getattr(out, field).view(torch.uint8), getattr(ref, field).view(torch.uint8)))
        self.assertTrue(torch.equal(pool.transpose(-1, -2).contiguous().view(torch.uint8), ref.decode().view(torch.uint8)))

    def test_06_Asym32_V2_both_coordinates_against_independent_bits(self):
        x = torch.linspace(-.02, .03, 128, dtype=torch.float32).reshape(1, 128)
        for value in [x, x @ hmatrix()]:
            out = asym_encode(value.numpy(), capture_stages=True)
            for g in range(4):
                ref = canonical_group(value.numpy().view(np.uint32).reshape(4, 32)[g].tolist())
                for k, expected in ref.items():
                    actual = out.stages[k][g]
                    self.assertEqual(np.asarray(actual).tolist(), expected, (g, k))

    def test_07_persistent_write_and_real_argument_consumption_all_arms(self):
        for arm in ARMS:
            a = FourCodecBoundary(torch, arm, hmatrix());pool = torch.zeros(2, 1, 128, 128);idx = torch.tensor([1])
            for pos in range(3):
                active = pool.index_select(0, idx);a.before_native_kernel(active, 0, pos)
                pool[1] += torch.sin(torch.arange(16384, dtype=torch.float32).reshape(1, 128, 128) + pos) * .001
                out = a.after_native_kernel(pool, idx, 0, pos)
                self.assertEqual(out.codes.dtype, torch.uint8 if a.arm.asymmetric else torch.int8)
                self.assertEqual(out.scales.dtype, torch.float16)
                self.assertTrue(torch.equal(pool[0], torch.zeros_like(pool[0])))
            self.assertEqual((a.writes, a.consumptions), (3, 2));a.close();self.assertFalse(a.previous)

    def test_08_contaminated_fresh_request_rejected(self):
        a = FourCodecBoundary(torch, 'B0', hmatrix())
        with self.assertRaisesRegex(AssertionError, 'ZERO_STATE'):
            a.before_native_kernel(torch.ones(1, 1, 128, 128), 0, 0)

    def test_09_corrupted_consumption_rejected_byte_exact(self):
        a = FourCodecBoundary(torch, 'B1', hmatrix());pool = torch.zeros(1, 1, 128, 128);idx = torch.tensor([0])
        a.after_native_kernel(pool, idx, 0, 0);pool.view(torch.uint8).reshape(-1)[0] ^= 1
        with self.assertRaisesRegex(AssertionError, 'POST_QDQ_BYTES_DIFFER'):
            a.before_native_kernel(pool, 0, 1)

    def test_10_stepwise_prefill_all_tokens_and_gate_arguments_preserved(self):
        for arm in ARMS:
            count = 3;a = FourCodecBoundary(torch, arm, hmatrix());pool = torch.zeros(1, 1, 128, 128);idx = torch.tensor([0])
            q = torch.ones(1, count, 1, 128);k = q * .01;v = q * .02
            g = torch.arange(count * 128, dtype=torch.float32).reshape(count, 128)
            beta = torch.ones(1, count, 1) * .3;A = torch.ones(1);dt = torch.zeros(128);seen = []
            def original(**kw):
                t = len(seen);seen.append(kw)
                self.assertTrue(torch.equal(kw['a'], g[t:t + 1]));self.assertTrue(torch.equal(kw['b'], beta[:, t:t + 1]))
                self.assertIs(kw['A_log'], A);self.assertIs(kw['dt_bias'], dt)
                self.assertEqual(kw['query_start_loc'].tolist(), [0, 1])
                pool[0] += kw['v'][0, 0, 0][:, None] * .01
                return kw['v'].clone()
            out = stepwise_prefill(torch=torch,q=q,k=k,v=v,g=g,beta=beta,A_log=A,dt_bias=dt,
                ssm_states=pool,cache_indices=idx,original_decode=original,boundary=a,layer=0)
            self.assertEqual(out.shape, v.shape);self.assertEqual(len(seen), count)
            self.assertEqual((a.writes, a.consumptions), (3, 2))

    def test_11_nonfinite_and_duplicate_writes_fail_closed(self):
        a = FourCodecBoundary(torch, 'B3', hmatrix());pool = torch.zeros(1, 1, 128, 128);idx = torch.tensor([0])
        a.after_native_kernel(pool, idx, 0, 0)
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE'):a.after_native_kernel(pool, idx, 0, 0)
        pool[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ArithmeticError, 'NONFINITE'):a.after_native_kernel(pool, idx, 0, 1)

    def test_12_no_forward_GPU_claim(self):
        self.assertFalse(torch.cuda.is_initialized())
        for arm in ARMS:self.assertFalse(FourCodecBoundary(torch,arm,hmatrix()).authorized)


if __name__ == '__main__':
    unittest.main(verbosity=2)
