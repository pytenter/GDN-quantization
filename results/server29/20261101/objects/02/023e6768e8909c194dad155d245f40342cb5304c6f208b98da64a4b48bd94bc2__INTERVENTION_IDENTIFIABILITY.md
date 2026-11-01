# Intervention identifiability

At every token, layer, group, and condition the frozen adaptive baseline uses `s_adapt`. `NO_SCALE_INFLATION` uses `min(s_adapt, s_clean)` and therefore differs exactly on positive deviations. `NO_SCALE_DEFLATION` uses `max(s_adapt, s_clean)` and therefore differs exactly on negative deviations. Whenever `s_adapt != s_clean`, one and only one of the two arms clamps that signed deviation to the paired clean value. Both arms preserve their own recurrent states and their own future adaptive scales, so neither trajectory is definitionally identical to the baseline.

`G3_INTERVENTION_IDENTIFIABILITY = PASS`
