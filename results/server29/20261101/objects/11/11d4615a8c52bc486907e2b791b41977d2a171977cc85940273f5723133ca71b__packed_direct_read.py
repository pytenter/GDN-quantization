"""Tiled packed mixed-affine base readout. No global dense state output.

Independent implementation of the window-readout identity; not official LQ code.
K/Q normalization and K4 encoding are performed by the unchanged PACE codec.
The CUDA kernel fuses packed decode, two base matvecs, pending vector readouts,
current delta, and current output. State tiles live only in kernel registers.
"""
import torch


def torch_readout(base, keys, updates, alphas, q, k, value, beta):
    """CPU/reference-only direct expansion; explicitly materializes base."""
    state = base.dequantize()
    aq = (state * q.unsqueeze(-1)).sum(-2)
    ak = (state * k.unsqueeze(-1)).sum(-2)
    if keys.shape[0]:
        weight = torch.ones_like(alphas[0])
        for j in range(keys.shape[0]):
            weight = weight * alphas[j]
        aq = aq * weight.unsqueeze(-1)
        ak = ak * weight.unsqueeze(-1)
        for i in range(keys.shape[0]):
            w = torch.ones_like(alphas[0])
            for j in range(i + 1, keys.shape[0]):
                w = w * alphas[j]
            aq = aq + updates[i] * ((keys[i] * q).sum(-1) * w).unsqueeze(-1)
            ak = ak + updates[i] * ((keys[i] * k).sum(-1) * w).unsqueeze(-1)
    a = alphas[-1].unsqueeze(-1)
    u = beta.float().unsqueeze(-1) * (value.float() - a * ak)
    out = (a * aq + u * (k * q).sum(-1).unsqueeze(-1)).to(torch.bfloat16)
    return u, out


def _kernel():
    global tl, libdevice
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def fused(C, M, Z, E, HEAD, Q, K, V, BETA, PK, PU, ALPHA, U, O,
              H: tl.constexpr, NH: tl.constexpr, DK: tl.constexpr,
              DV: tl.constexpr, P: tl.constexpr, BITS: tl.constexpr,
              HAS_E: tl.constexpr, BV: tl.constexpr):
        bh = tl.program_id(0)
        b, lh = bh // NH, bh % NH
        h = tl.load(HEAD + lh)
        vv = tl.program_id(1) * BV + tl.arange(0, BV)
        kk = tl.arange(0, DK)
        flat = ((b * NH + lh) * DV + vv[:, None]) * DK + kk[None, :]
        gi = flat // 32
        if BITS == 4:
            raw = tl.load(C + flat // 2, vv[:, None] < DV, 0).to(tl.int32)
            code = (raw >> ((flat % 2) * 4)) & 15
            zpraw = tl.load(Z + gi // 2, vv[:, None] < DV, 0).to(tl.int32)
            zp = (zpraw >> ((gi % 2) * 4)) & 15
        else:
            code = tl.load(C + flat, vv[:, None] < DV, 0).to(tl.int32)
            zp = tl.load(Z + gi, vv[:, None] < DV, 0).to(tl.int32)
        scale = tl.load(M + gi, vv[:, None] < DV, 1).to(tl.float32)
        if HAS_E:
            exponent = tl.load(E + gi, vv[:, None] < DV, 0).to(tl.int32)
            scale = libdevice.ldexp(scale, exponent)
        tile = (code - zp).to(tl.float32) * scale
        q = tl.load(Q + (b * H + h) * DK + kk)
        k = tl.load(K + (b * H + h) * DK + kk)
        fq = tl.sum(tile * q[None, :], 1)
        fk = tl.sum(tile * k[None, :], 1)
        weight = tl.full((), 1.0, tl.float32)
        for j in tl.static_range(P):
            weight = weight * tl.load(ALPHA + (j * tl.num_programs(0) // NH + b) * H + h)
        fq = fq * weight
        fk = fk * weight
        for i in tl.static_range(P):
            key = tl.load(PK + ((i * tl.num_programs(0) // NH + b) * H + h) * DK + kk)
            uq = tl.sum(key * q, 0)
            uk = tl.sum(key * k, 0)
            tail = tl.full((), 1.0, tl.float32)
            for j in tl.static_range(i + 1, P):
                tail = tail * tl.load(ALPHA + (j * tl.num_programs(0) // NH + b) * H + h)
            uv = tl.load(PU + ((i * tl.num_programs(0) // NH + b) * H + h) * DV + vv, vv < DV, 0)
            fq = fq + uv * (uq * tail)
            fk = fk + uv * (uk * tail)
        batch = tl.num_programs(0) // NH
        alpha = tl.load(ALPHA + (P * batch + b) * H + h)
        beta = tl.load(BETA + b * H + h).to(tl.float32)
        val = tl.load(V + (b * H + h) * DV + vv, vv < DV, 0).to(tl.float32)
        u = beta * (val - alpha * fk)
        out = alpha * fq + u * tl.sum(k * q, 0)
        tl.store(U + (b * H + h) * DV + vv, u, vv < DV)
        tl.store(O + (b * H + h) * DV + vv, out, vv < DV)
    return fused


_FUSED = None


def packed_readout(base, keys, updates, alphas, q, k, value, beta, head_maps):
    global _FUSED
    if value.device.type != "cuda":
        raise RuntimeError("Production packed readout requires CUDA; CPU is validation-only")
    if _FUSED is None:
        _FUSED = _kernel()
    b, h, dk, dv = base.logical_shape
    if dk != 128 or dv != 128:
        raise ValueError("Validated kernel specialization is K=V=128 only")
    u = torch.empty((b, h, dv), device=value.device, dtype=torch.float32)
    out = torch.empty_like(value)
    for bits, heads, part in ((4, base.int4_heads, base.int4), (8, base.int8_heads, base.int8)):
        if part is None:
            continue
        exp = part.scale_exponents
        _FUSED[(b * len(heads), (dv + 15) // 16)](
            part.codes, part.scales, part.zero_points,
            exp if exp is not None else part.zero_points, head_maps[bits],
            q.contiguous(), k.contiguous(), value.contiguous(), beta.contiguous(),
            keys, updates, alphas, u, out,
            H=h, NH=len(heads), DK=dk, DV=dv, P=keys.shape[0], BITS=bits,
            HAS_E=exp is not None, BV=16, num_warps=4, enable_fp_fusion=False)
    return u, out
