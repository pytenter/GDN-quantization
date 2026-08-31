# Evidence Map

This document is a research evidence map, not a paper abstract and not a method proposal.

## Observation

- INT8 orientation is a first-order experimental variable for GDN recurrent-state quantization.
- R128 degradation is substantially stronger than C128 under the tested protocol.
- The recurrent state has shape `[B,H,K,V] = [1,32,128,128]`; row groups follow the Key-axis rows and column groups follow the Value-axis columns.

## Supported Association

- Finer grouping rescues KL in the tested setting.
- Runtime effective-update metrics track R16/C16 KL rescue across the 6 canonical prompts.
- Random orthogonal single-pulse directions show a broad recurrent directional sensitivity distribution.

## Causal Evidence

- Residual-strength attenuation produces a KL dose response.
- Same-norm residual geometry causally changes model fidelity.
- Same-norm single-pulse perturbations exhibit direction-dependent recurrent propagation.

## Important Negative / Corrective Results

- Same-codebook lost-update metrics do not explain the grouping rescue.
- A simple dynamic-range to lost-update to quality causal chain is not supported.
- Raw state persistence is insufficient to explain future KL.
- The natural R128 orthogonal residual direction is low-gain biased in the direction sensitivity panel.
- The readout-aware single-pulse pilot improves correlation over raw state persistence but remains inconclusive.

## Inconclusive

- Direction sensitivity panel: `PROPAGATION_FIDELITY_LINK = INCONCLUSIVE`.
- Readout-aware propagation audit: `Pilot = INCONCLUSIVE`, `Formal = NOT_STARTED`.

## Unresolved

- Why repeated moderate R128 residuals cause severe long-horizon degradation.
- Whether repeated residual accumulation closes the natural-R128 paradox.
- Whether full-network feedback amplifies recurrent-state error.
- Whether the final mechanism generalizes to C128.
