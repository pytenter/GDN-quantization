"""Qualification observations only; all encode/decode equations stay frozen."""
from pathlib import Path
import os
import json
import hashlib
import numpy as np
from four_codec_boundary import FourCodecBoundary
from canonical_codec import encode as cpu_canonical
from independent_reference import canonical_group
from matched_baseline import build_reference_encoder


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


class QualificationBoundary(FourCodecBoundary):
    def __init__(self, torch, arm, h, *, backend, evidence_directory,
                 progress, probe_positions=(0, 31, 127, 511)):
        self.directory = Path(evidence_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.probe_positions = frozenset(probe_positions)
        self.progress = progress
        self.evidence = []
        self.comparisons = []
        self.counter = 0
        self.cpu_sym = build_reference_encoder(torch)
        super().__init__(torch, arm, h, gpu_authorized=True,
                         canonical_cuda_backend=backend,
                         strict_comparator=self.compare_canonical,
                         durable_evidence=self.persist)

    def persist(self, kind, context, host, stages):
        self.counter += 1
        stem = self.directory / ('%04d_%s_L%s_T%s' %
                                  (self.counter, kind, context['layer'], context['position']))
        refs = {}

        def save(key, array):
            path = Path(str(stem) + '_' + key + '.npy')
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            refs[key] = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                             sha256=digest_bytes(path.read_bytes()))

        save('S0', host.detach().contiguous().numpy())
        if stages:
            for key, value in stages.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        save(key + '_' + subkey, subvalue)
                elif value is not None:
                    save(key, value)
        receipt = Path(str(stem) + '.json')
        raw = (json.dumps(dict(kind=kind, context=context, refs=refs), sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + '\n').encode()
        with receipt.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.evidence.append(dict(path=str(receipt.resolve()), sha256=digest_bytes(raw)))
        self.progress(dict(stage=kind, **context))

    def compare_canonical(self, actual, expected):
        checks = {}
        for key in expected.stages:
            a, b = np.ascontiguousarray(actual.stages[key]), np.ascontiguousarray(expected.stages[key])
            checks[key] = bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
        if actual.stages.keys() != expected.stages.keys() or not all(checks.values()):
            raise AssertionError(('REAL_ASYM_CANONICAL_CPU_GPU_STAGE_MISMATCH', checks))
        count = len(expected.stages['S0_bits32'])
        selected = sorted({0, 1, 2, 3, count // 2, count - 3, count - 2, count - 1})
        for group in selected:
            oracle = canonical_group(expected.stages['S0_bits32'][group].tolist())
            for name, wanted in oracle.items():
                if np.asarray(expected.stages[name][group]).tolist() != wanted:
                    raise AssertionError(('INDEPENDENT_REAL_INPUT_GROUP_ORACLE_MISMATCH', group, name))
        self.comparisons.append(dict(kind='ASYM15_STAGES', status='PASS_EXACT', groups=count,
                                     exact_stage_checks=checks, independent_Fraction_groups=selected))

    def after_native_kernel(self, pool, indices, layer, position, *, probe=False):
        torch = self.torch
        selected = position in self.probe_positions
        host = None
        context = dict(arm=self.arm.identifier, layer=layer, position=position)
        if selected and not self.arm.asymmetric:
            host = pool.index_select(0, indices.to(torch.int64)).transpose(-1, -2).contiguous().cpu()
            self.persist('SYMMETRIC_PRE_ENCODE', context, host, None)
        # The inherited implementation owns actual CUDA encode, immediate write,
        # and the audit-only previous-state clone; nothing here feeds a CPU state.
        actual = super().after_native_kernel(pool, indices, layer, position, probe=selected)
        if host is not None:
            expected = self.cpu_sym(host, 32, False)
            pairs = dict(codes=(actual.codes, expected.codes), scales=(actual.scales, expected.scales),
                         decoded=(actual.decode(torch), expected.decode()))
            checks, saved = {}, {}
            for name, (a, b) in pairs.items():
                a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
                checks[name] = bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
                saved['actual_' + name] = a.numpy()
                saved['cpu_' + name] = b.numpy()
            self.persist('SYMMETRIC_ACTUAL_AND_CPU_STORED_FIELDS', context, host, saved)
            if not all(checks.values()):
                raise AssertionError(('REAL_SYMMETRIC_CPU_GPU_STRICT_PARITY_MISMATCH', checks))
            self.comparisons.append(dict(kind='SYMMETRIC_STORED_FIELDS', status='PASS_EXACT',
                                         context=context, checks=checks))
        return actual

