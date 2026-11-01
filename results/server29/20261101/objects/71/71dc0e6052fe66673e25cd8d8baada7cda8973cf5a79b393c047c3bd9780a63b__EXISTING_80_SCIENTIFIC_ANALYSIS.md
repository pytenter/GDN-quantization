# Existing-80 scientific analysis

## Scope and evidence boundary

This is an exploratory, read-only analysis of the 80 verified historical canonical units. It does not resume Attempt04 or Attempt05, does not merge old and new runtime panels, and does not treat rotation conditions as independent PG19 documents. The two complete documents (`pg19-test-37702`, `pg19-test-5396`) are the core evidence; incomplete documents are descriptive only.

The analysis method was frozen in `ANALYSIS_PLAN.json` before outcome computation. Propagation magnitude is canonical `PULSE_PROP_ENERGY_AUC` (called JPROP here); observability is canonical `POR`; the target is `TARGET_FUTURE_KL_AUC`. Spearman effects, a fixed two-predictor rank model, leave-one-condition-out (LOOCV) errors, leave-one-condition-out DeltaRho, and a predeclared magnitude-match rule were computed without imputation.

## Inventory

- 80 units: Primary 41, Secondary 39.
- 39 complete Primary/Secondary pairs and 2 unmatched Primary units.
- Core: 64 units = 2 documents x 16 rotations x 2 anchors.
- Auxiliary: 16 units = 7 complete pairs + 2 unmatched Primary.
- All canonical/sidecar byte counts and SHA-256 values reverified; duplicate scientific keys = 0.

## Complete-document effects

| document | anchor | n | rho(JPROP, FutureKL) | rho(POR, FutureKL) | DeltaRho | rho(JPROP,POR) | partial rank POR given JPROP | in-sample rank R2 J / J+POR | LOOCV rank RMSE improvement | LOOCV log10 RMSE improvement | leave-one DeltaRho range |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 37702 | Primary 8192 | 16 | 0.6971 | 0.7765 | +0.0794 | 0.8706 | 0.4808 | 0.4859 / 0.6047 | +7.49% | -5.51% | +0.0429 to +0.1179 |
| 37702 | Secondary 2048 | 16 | 0.7029 | 0.8412 | +0.1382 | 0.7676 | 0.6616 | 0.4941 / 0.7155 | -36.89% | +16.42% | -0.0714 to +0.1893 |
| 5396 | Primary 8192 | 16 | 0.9147 | 0.8618 | -0.0529 | 0.8559 | 0.3774 | 0.8367 / 0.8600 | -9.81% | +13.81% | -0.0786 to -0.0321 |
| 5396 | Secondary 2048 | 16 | 0.6265 | 0.7618 | +0.1353 | 0.8529 | 0.5589 | 0.3925 / 0.5823 | +9.88% | +10.78% | 0.0000 to +0.1714 |

Positive LOOCV improvement means the JPROP+POR model reduced held-out RMSE; negative means it worsened it.

### Interpretation

1. Propagation magnitude is already informative: all four JPROP correlations are positive and range from 0.6265 to 0.9147.
2. POR has a stronger univariate rank association in three of four panels, but not in `5396/Primary`, where JPROP is stronger by 0.0529.
3. POR retains positive partial rank association after controlling for JPROP in every panel (0.3774--0.6616), and improves in-sample rank R2 in every panel. This is compatible with incremental information, but not sufficient evidence by itself because JPROP and POR are highly collinear (rho 0.7676--0.8706) and n=16 per panel.
4. Held-out behavior is mixed. Rank-space LOOCV improves only in two panels; log10-space LOOCV improves in three. The conclusion is therefore model/scale sensitive, not a uniform predictive win.
5. Outlier sensitivity is also mixed. `37702/Primary` keeps positive DeltaRho under every single-condition deletion. `5396/Primary` keeps a negative DeltaRho under every deletion. `5396/Secondary` never becomes negative, but reaches zero. `37702/Secondary` becomes negative when an influential condition is omitted. The latter advantage is not robust.

## Magnitude-matched evidence

The rule was frozen before reading FutureKL outcomes: relative JPROP gap <= 10% and POR ratio >= 1.50. All qualifying pairs were reported.

- `37702/Primary`: 0 qualifying pairs.
- `37702/Secondary`: 3 qualifying pairs; 3/3 have larger FutureKL for the higher-POR member; 0 counterexamples.
- `5396/Primary`: 12 qualifying pairs; 12/12 support; 0 counterexamples.
- `5396/Secondary`: 3 qualifying pairs; 3/3 support; 0 counterexamples.
- Total: 18/18 directional support, 0 counterexamples under the frozen rule.

The complete pair list is in `MAGNITUDE_MATCHED_PAIRS.csv`. These comparisons are scientifically useful, but they reuse rotations inside only two documents and are not randomized direction interventions. They are conditional descriptive evidence, not causal confirmation.

No condition met the separately frozen rule “JPROP top quartile and POR bottom quartile” in any complete panel. Therefore the stronger claim that very large propagated error can remain functionally mild solely because it is readout-insensitive is not directly tested by these 80 units.

## Anchor stability

Condition rankings are substantially preserved between Primary and Secondary:

| document | rho Primary-vs-Secondary JPROP | POR | FutureKL |
|---|---:|---:|---:|
| 37702 | 0.7471 | 0.9000 | 0.8588 |
| 5396 | 0.7559 | 0.8618 | 0.8500 |

This supports a real document/rotation structure across anchors, but Primary and Secondary are paired measurements, not independent replications. The POR-over-JPROP advantage itself is not anchor-invariant: it is positive in both 37702 anchors and only the 5396 Secondary anchor.

## Auxiliary incomplete documents

Missingness is an exact scheduler execution prefix caused by the host reboot, so these are selection-biased descriptive checks.

- `28988/Primary`, n=7: rho(JPROP,FutureKL)=0.9643; rho(POR,FutureKL)=0.8214; DeltaRho=-0.1429.
- `28988/Secondary`, n=6: 0.7143 versus 0.7714; DeltaRho=+0.0571.
- `26493/Primary`, n=2, and `26493/Secondary`, n=1: too few observations for a meaningful rank correlation.

The observed 28988 subset repeats the pattern “JPROP stronger at Primary, POR slightly stronger at Secondary.” It does not constitute a complete 16-condition ranking and cannot be pooled as another full document.

## Answers to mechanism questions

### A. Similar magnitude, different observability

Within the strict predeclared rule, 18/18 pairs have higher FutureKL for the higher-POR condition. This is the clearest current exploratory result. Its limitation is lack of prospective or randomized direction manipulation.

### B. Large magnitude, low sensitivity

Not answered: no complete-panel condition fell into the frozen high-JPROP/low-POR cell. Absence of such a naturally occurring condition is not evidence that the phenomenon cannot occur.

### C. Primary/Secondary consistency

The sign is not fully consistent. POR exceeds JPROP in both 37702 anchors and in 5396 Secondary, but not in 5396 Primary. Conditional partial associations are positive in all four panels, while LOOCV gains depend on scale. The evidence is compatible with an observability contribution, but not a stable universal superiority claim.

### D. Target leakage

No direct label leakage was found. `POR` is computed from propagated state-path differences and the realized future K3 query/readout records; `FutureKL` is computed separately from full logits. POR does not consume the FutureKL scalar or logits. It is nevertheless future-aware and retrospective, so it is not an online/trainable objective as currently defined.

### E. Diagnostic versus trainable surrogate

Current POR is a useful retrospective mechanism diagnostic. A trainable surrogate must replace access to the full realized future query/readout trajectory with information available at training time (for example, a preregistered short-horizon or predicted readout-sensitivity statistic) and then pass independent prospective validation. That key bridge has not yet been demonstrated.

## Scientific conclusion

Existing data are sufficient for a nontrivial descriptive conclusion: propagated error magnitude matters, and future readout observability appears to add information in several panels and in every strict magnitude-matched comparison. The evidence is not sufficient for a causal claim or a trainable loss. Only two complete documents exist, the panel rows are correlated rotations, collinearity is high, auxiliary missingness is nonrandom, and predictive improvements are not uniformly robust. The correct route is a small preregistered independent pilot before any larger panel or training effort.
