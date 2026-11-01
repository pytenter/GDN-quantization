"""CPU interface negatives for the SGLang hook; not real-model qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from four_codec_boundary import FourCodecBoundary
from sglang_four_codec_adapter import SingleRequestKDAAdapter


def matrix():
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / (128 ** .5)


class AdapterCPU(unittest.TestCase):
    def exercise(self, arm, *, corrupt=False, mutate_gate=False):
        seen = []
        class Dispatcher:
            supports_packed_decode = True
            def decode(self, **kw):
                seen.append(kw)
                q, k, v = (kw[n][0, 0, 0] for n in ('q', 'k', 'v'))
                pool = kw['ssm_states']
                pool[0, 0] += v[:, None] * k[None, :]
                if mutate_gate:
                    kw['a'].add_(1)
                return (pool[0, 0] @ q).reshape(1, 1, 1, 128)
            def extend(self, **kw):
                raise AssertionError('CHUNK_END_ONLY_PREFILL_MUST_NOT_RUN')
        class Backend:
            def __init__(self):
                self.kernel_dispatcher = Dispatcher()
                self.forward_metadata = SimpleNamespace(has_mamba_track_mask=False)
            def forward_extend(self, layer, batch, **kw):
                return self.kernel_dispatcher.extend(**kw)
            def forward_decode(self, layer, batch, **kw):
                self.assert_unpacked = not self.kernel_dispatcher.supports_packed_decode
                return self.kernel_dispatcher.decode(**kw)
        boundary = FourCodecBoundary(torch, arm, matrix())
        adapter = SingleRequestKDAAdapter(boundary, probe_positions=(0, 1, 2, 3))
        backend = Backend()
        old = Dispatcher.decode
        pool = torch.zeros(1, 1, 128, 128)
        index = torch.tensor([0], dtype=torch.int64)
        layer = SimpleNamespace(layer_id=2, head_k_dim=128, head_v_dim=128)
        q = torch.full((1, 3, 1, 128), .01)
        k = torch.full_like(q, .02)
        v = torch.linspace(-.03, .04, 384).reshape(1, 3, 1, 128)
        a = torch.arange(384, dtype=torch.float32).reshape(3, 128)
        b = torch.full((1, 3, 1), .2)
        A = torch.tensor([-.3])
        dt = torch.zeros(128)
        adapter.install(Backend, Dispatcher)
        try:
            kwargs = dict(q=q, k=k, v=v, g=a, beta=b, A_log=A, dt_bias=dt,
                          ssm_states=pool, cache_indices=index,
                          query_start_loc=torch.tensor([0, 3]),
                          return_intermediate_states=False, is_spec_decode=False)
            result = backend.forward_extend(layer, SimpleNamespace(input_ids=torch.ones(3)), **kwargs)
            self.assertEqual(result.shape, (1, 3, 1, 128))
            self.assertEqual((adapter.prefill_writes, boundary.writes, boundary.consumptions), (3, 3, 2))
            if corrupt:
                pool.view(torch.uint8).flatten()[0] ^= 1
            native = dict(q=q[:, :1], k=k[:, :1], v=v[:, :1], a=a[:1], b=b[:, :1],
                          A_log=A, dt_bias=dt, ssm_states=pool, cache_indices=index,
                          query_start_loc=torch.tensor([0, 1]), lower_bound=-5)
            result = backend.forward_decode(layer, SimpleNamespace(input_ids=torch.ones(1)), **native)
            self.assertEqual(result.shape, (1, 1, 1, 128))
            self.assertEqual((adapter.decode_writes, boundary.writes, boundary.consumptions), (1, 4, 3))
            self.assertTrue(backend.assert_unpacked)
            self.assertEqual(len(seen), 4)
            for t in range(3):
                self.assertTrue(torch.equal(seen[t]['a'], a[t:t+1]))
                self.assertTrue(torch.equal(seen[t]['b'], b[:, t:t+1]))
                self.assertIs(seen[t]['A_log'], A)
                self.assertIs(seen[t]['dt_bias'], dt)
            self.assertEqual(seen[-1]['lower_bound'], -5)
            self.assertTrue(all(all(p['unchanged'].values()) for p in adapter.gate_probes))
        finally:
            adapter.close()
            self.assertIs(Dispatcher.decode, old)

    def test_all_four_prefill_decode_own_history(self):
        for arm in ('B0', 'B1', 'B2', 'B3'):
            self.exercise(arm)

    def test_real_argument_corruption_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'POST_QDQ_BYTES_DIFFER'):
            self.exercise('B1', corrupt=True)

    def test_key_gate_mutation_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'GATE_OR_PARAMETERS_MODIFIED'):
            self.exercise('B3', mutate_gate=True)


if __name__ == '__main__':
    unittest.main()
