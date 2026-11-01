# Original rotation rule: documentary record only

  ROTATION_PARAMETERIZATION_IDENTITY_GATE = PASS

If exact interpolation-compatible parameterization cannot be recovered,
STOP before generating derived rotations.

Do NOT silently switch to arbitrary Euclidean matrix interpolation.

============================================================
14. FROZEN 12-CONDITION ROTATION BANK
============================================================

Create exactly 12 conditions.

Anchors:

  R00 = H
  R01 = C5R
  R02 = C6R
  R03 = C7

Derived rotations:

Using the exact archived Cayley/skew parameterization.

If:

  R(A) = H @ Cayley(A)

then use:

  H→C6R:
    R04 = R(0.20 * A_C6R)
    R05 = R(0.40 * A_C6R)
    R06 = R(0.60 * A_C6R)
    R07 = R(0.80 * A_C6R)

  H→C7:
    R08 = R(0.50 * A_C7)

  H→C5R:
    R09 = R(0.50 * A_C5R)

  C6R↔C7 parameter midpoint:
    R10 = R(0.50*A_C6R + 0.50*A_C7)

  C6R↔C5R parameter midpoint:
    R11 = R(0.50*A_C6R + 0.50*A_C5R)

All interpolation is layerwise using that layer's exact frozen
parameter.

Do NOT include Identity / Native purely to increase dynamic range.

Reason:

The scientific target is the near-useful rotation neighborhood around
H and learned rotations, not trivial separation of catastrophic

This mathematical definition was already specified before measurement. No new coefficients were invented. Its formal recovery/materialization gates were NOT_EVALUATED in this amendment because Discovery rule recovery failed first. No checkpoint tensor was loaded and no new matrix or orthogonality metric computed.
