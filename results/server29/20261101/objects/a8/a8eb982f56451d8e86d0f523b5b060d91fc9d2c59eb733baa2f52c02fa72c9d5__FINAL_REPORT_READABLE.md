# Ling error-coherence Discovery: final readable report

This is a presentation-only supplement. The frozen scientific scripts,
raw units, original analysis JSON/CSV/Markdown and preregistration are unchanged.

## Required status

```ini
EXPERIMENT = LING_ERROR_COHERENCE_DIAGNOSIS_V1
PREREGISTRATION_GATE = PASS
PROVENANCE_GATE = PASS
RUNTIME_IDENTITY_GATE = PASS
DECOMPOSITION_CLOSURE_GATE = PASS
DISCOVERY_COMPLETE = 64/64
CONFIRMATION_EXECUTED = 0
L9_TRAINING_EXECUTED = NO
AIME_EXECUTED = NO
ERROR_COHERENCE_DISCOVERY_LABEL = DESCRIPTIVE_SIGNAL_NOT_PRESENT
CONFIRMATORY_STATUS = NOT_EVALUATED
ENERGY_DECOMPOSITION_IDENTITY_IS_TAUTOLOGICAL = YES
SCIENTIFIC_QUESTION = IS_ALIGNMENT_STRUCTURED_AND_RANK_DECISIVE
PRIMARY_NEXT_STEP_CATEGORY = CONSIDER_READOUT_SENSITIVITY_DIAGNOSIS
PARENT_GLOBAL_GATE = FAIL_PRESERVED
CROSS_HARDWARE_FP_BITWISE = FAIL_PRESERVED
FP_REFERENCES_MIXED_ACROSS_HARDWARE = NO
NO_NEW_MODEL_FORWARD = NO; Route B teacher-forced FP capture
PREMEASUREMENT_COMMIT_SHA = d4906196ab2921035f4c175804fe27af15eb1314
FINAL_ACTION = STOP_AFTER_DISCOVERY
L9_AUTHORIZED = NO
AIME_AUTHORIZED = NO
```

## Horizon ranking diagnostics

These are the unchanged predeclared document-level medians (four documents,
16 rotations each). No pooled 64-point primary correlation or p-value is used.

| horizon | J_INJ_rho | J_PROP_rho | J_NOALIGN_rho | J_NOALIGN_concordance | alignment_decisive_pair_rate | median_NET_COHERENCE_RATIO |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 0.8955882353 | 0.8573529412 | 1 | 1 | 0 | 0.001687820239 |
| 8192 | 0.8794117647 | 0.6485294118 | 1 | 1 | 0 | 0.0009196322564 |
| 32768 | 0.5602941176 | 0.7705882353 | 1 | 1 | 0 | 0.0005062027471 |

### Per-document values

| document_id | horizon | J_INJ_rho | J_PROP_rho | J_NOALIGN_rho | J_NOALIGN_concordance | alignment_decisive_pair_rate | median_NET_COHERENCE_RATIO | total_non_tied_pairs | alignment_decisive_pair_count | median_abs_NET_COHERENCE_RATIO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19-test-37702 | 2048 | 0.8941176471 | 0.9029411765 | 1 | 1 | 0 | 0.001798587867 | 120 | 0 | 0.001798587867 |
| pg19-test-37702 | 8192 | 0.9 | 0.7088235294 | 1 | 1 | 0 | 0.0008914148737 | 120 | 0 | 0.0008914148737 |
| pg19-test-37702 | 32768 | 0.5176470588 | 0.7970588235 | 1 | 1 | 0 | 0.0004629516623 | 120 | 0 | 0.0004629516623 |
| pg19-test-5396 | 2048 | 0.85 | 0.9264705882 | 1 | 1 | 0 | 0.001692327509 | 120 | 0 | 0.001692327509 |
| pg19-test-5396 | 8192 | 0.9 | 0.6941176471 | 1 | 1 | 0 | 0.0009513770438 | 120 | 0 | 0.0009513770438 |
| pg19-test-5396 | 32768 | 0.6970588235 | 0.9088235294 | 1 | 1 | 0 | 0.0005494538318 | 120 | 0 | 0.0005494538318 |
| pg19-test-28988 | 2048 | 0.8970588235 | 0.8117647059 | 1 | 1 | 0 | 0.001664120549 | 120 | 0 | 0.001664120549 |
| pg19-test-28988 | 8192 | 0.8588235294 | 0.5911764706 | 1 | 1 | 0 | 0.0009478496391 | 120 | 0 | 0.0009478496391 |
| pg19-test-28988 | 32768 | 0.5823529412 | 0.7441176471 | 1 | 1 | 0 | 0.0006107768434 | 120 | 0 | 0.0006107768434 |
| pg19-test-26493 | 2048 | 0.9470588235 | 0.7470588235 | 1 | 1 | 0 | 0.00168331297 | 120 | 0 | 0.00168331297 |
| pg19-test-26493 | 8192 | 0.8352941176 | 0.6029411765 | 1 | 1 | 0 | 0.0008399790327 | 120 | 0 | 0.0008399790327 |
| pg19-test-26493 | 32768 | 0.5382352941 | 0.5352941176 | 1 | 1 | 0 | 0.0004598026108 | 120 | 0 | 0.0004598026108 |

## Signed interaction and segment diagnostics

The following are descriptive displays only: medians over the fixed 16 rotations
within each document, without treating rotations as independent documents.
POS_ALIGNMENT is the amplification magnitude; NEG_ALIGNMENT is signed negative
(its absolute value is cancellation magnitude). No ambiguous amplification-driven
or cancellation-driven pair-flip classification is assigned.

| document_id | horizon | positive_alignment_fraction | negative_alignment_fraction | zero_alignment_fraction | POS_ALIGNMENT | NEG_ALIGNMENT | NET_COHERENCE_RATIO | COHERENCE_BIAS | mean_token_coherence | mean_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19-test-26493 | 2048 | 0.7763685637 | 0.2236314363 | 0 | 2.930204046 | -0.7074828685 | 0.00168331297 | 0.6100047928 | 0.00190093666 | 0.05424542094 |
| pg19-test-26493 | 8192 | 0.7850738519 | 0.2149261481 | 0 | 34.84556633 | -8.374404245 | 0.0008399790327 | 0.619483996 | 0.001242357367 | 0.05493772193 |
| pg19-test-26493 | 32768 | 0.79276789 | 0.20723211 | 0 | 259.2836581 | -55.06880908 | 0.0004598026108 | 0.6889377088 | 0.0007757844761 | 0.05422010815 |
| pg19-test-28988 | 2048 | 0.7837940379 | 0.2162059621 | 0 | 3.562970895 | -0.859296606 | 0.001664120549 | 0.6089916167 | 0.001916298491 | 0.05588407084 |
| pg19-test-28988 | 8192 | 0.7863951272 | 0.2136048728 | 0 | 39.49653228 | -9.756553289 | 0.0009478496391 | 0.6231188784 | 0.001277974056 | 0.06004568437 |
| pg19-test-28988 | 32768 | 0.7996526137 | 0.2003473863 | 0 | 303.8053608 | -55.00257745 | 0.0006107768434 | 0.7346267836 | 0.0008272149476 | 0.06448715979 |
| pg19-test-37702 | 2048 | 0.7794308943 | 0.2205691057 | 0 | 3.781458756 | -0.8216502724 | 0.001798587867 | 0.6359246843 | 0.001907110653 | 0.05760489423 |
| pg19-test-37702 | 8192 | 0.7840354303 | 0.2159645697 | 0 | 40.03608569 | -9.429318526 | 0.0008914148737 | 0.6306650233 | 0.00123705145 | 0.05875823635 |
| pg19-test-37702 | 32768 | 0.7914658476 | 0.2085341524 | 0 | 212.2762632 | -49.26817115 | 0.0004629516623 | 0.6604575014 | 0.0007753375707 | 0.05733360094 |
| pg19-test-5396 | 2048 | 0.7949051491 | 0.2050948509 | 0 | 3.794415229 | -0.8447537975 | 0.001692327509 | 0.6206307404 | 0.001961602214 | 0.0592785026 |
| pg19-test-5396 | 8192 | 0.7904984424 | 0.2095015576 | 0 | 42.4435902 | -9.252221699 | 0.0009513770438 | 0.6207723379 | 0.001277686055 | 0.06235758173 |
| pg19-test-5396 | 32768 | 0.8006536462 | 0.1993463538 | 0 | 294.0111209 | -57.82013798 | 0.0005494538318 | 0.7107974744 | 0.0008208503216 | 0.06596758507 |

### Segments

PRIMARY_SUPPORT_INTERSECTION follows the frozen scoring support: [1024,2048],
[2049,8192], [8193,32768]. FULL_SEGMENT_DIAGNOSTIC additionally displays tokens
[1,2048] for the first segment; it does not replace primary support.
Every field below is a marginal median over 16 rotations, so the displayed
median energies should not be algebraically combined to reconstruct the
separately displayed median NET ratio. Exact per-rotation sums are in the CSV.

| document_id | support | segment | SUM_HISTORY | SUM_INJECTION | SUM_ALIGNMENT | NET_COHERENCE_RATIO | positive_alignment_fraction | negative_alignment_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19-test-26493 | PRIMARY_SUPPORT_INTERSECTION | 0 | 1385.063957 | 0.3984643241 | 2.246752543 | 0.00168331297 | 0.7763685637 | 0.2236314363 |
| pg19-test-26493 | PRIMARY_SUPPORT_INTERSECTION | 1 | 29365.21516 | 1.975831651 | 23.56035304 | 0.0008044960244 | 0.7865261502 | 0.2134738498 |
| pg19-test-26493 | PRIMARY_SUPPORT_INTERSECTION | 2 | 354622.5227 | 7.197351153 | 175.5860701 | 0.0004366652624 | 0.7962861238 | 0.2037138762 |
| pg19-test-28988 | PRIMARY_SUPPORT_INTERSECTION | 0 | 1646.167547 | 0.427411818 | 2.71165638 | 0.001664120549 | 0.7837940379 | 0.2162059621 |
| pg19-test-28988 | PRIMARY_SUPPORT_INTERSECTION | 1 | 29541.61146 | 2.030140223 | 26.46632828 | 0.0009172179768 | 0.7871545862 | 0.2128454138 |
| pg19-test-28988 | PRIMARY_SUPPORT_INTERSECTION | 2 | 352651.8223 | 6.87571248 | 228.5543446 | 0.0005952255189 | 0.8052164714 | 0.1947835286 |
| pg19-test-37702 | PRIMARY_SUPPORT_INTERSECTION | 0 | 1680.850107 | 0.4239004945 | 2.958086245 | 0.001798587867 | 0.7794308943 | 0.2205691057 |
| pg19-test-37702 | PRIMARY_SUPPORT_INTERSECTION | 1 | 36247.19099 | 1.994046745 | 27.38869034 | 0.0008485377968 | 0.7839988426 | 0.2160011574 |
| pg19-test-37702 | PRIMARY_SUPPORT_INTERSECTION | 2 | 317050.5012 | 6.218435074 | 138.2920104 | 0.0004291896401 | 0.7950823749 | 0.2049176251 |
| pg19-test-5396 | PRIMARY_SUPPORT_INTERSECTION | 0 | 1669.035076 | 0.4241223292 | 2.934862624 | 0.001692327509 | 0.7949051491 | 0.2050948509 |
| pg19-test-5396 | PRIMARY_SUPPORT_INTERSECTION | 1 | 33958.77402 | 2.051745593 | 29.42744408 | 0.0009148394758 | 0.789763274 | 0.210236726 |
| pg19-test-5396 | PRIMARY_SUPPORT_INTERSECTION | 2 | 404550.7317 | 6.955526565 | 212.6155297 | 0.000532775273 | 0.8051317003 | 0.1948682997 |
| pg19-test-26493 | FULL_SEGMENT_DIAGNOSTIC | 0 | 1932.827819 | 1.030784404 | 3.949948941 | 0.00213058926 | 0.7620035807 | 0.237508138 |
| pg19-test-26493 | FULL_SEGMENT_DIAGNOSTIC | 1 | 29365.21516 | 1.975831651 | 23.56035304 | 0.0008044960244 | 0.7865261502 | 0.2134738498 |
| pg19-test-26493 | FULL_SEGMENT_DIAGNOSTIC | 2 | 354622.5227 | 7.197351153 | 175.5860701 | 0.0004366652624 | 0.7962861238 | 0.2037138762 |
| pg19-test-28988 | FULL_SEGMENT_DIAGNOSTIC | 0 | 2260.237411 | 1.072130926 | 4.765444725 | 0.002116980572 | 0.7725965712 | 0.2269151476 |
| pg19-test-28988 | FULL_SEGMENT_DIAGNOSTIC | 1 | 29541.61146 | 2.030140223 | 26.46632828 | 0.0009172179768 | 0.7871545862 | 0.2128454138 |
| pg19-test-28988 | FULL_SEGMENT_DIAGNOSTIC | 2 | 352651.8223 | 6.87571248 | 228.5543446 | 0.0005952255189 | 0.8052164714 | 0.1947835286 |
| pg19-test-37702 | FULL_SEGMENT_DIAGNOSTIC | 0 | 2244.3856 | 1.054975295 | 4.772050852 | 0.002197051089 | 0.7687445747 | 0.2307671441 |
| pg19-test-37702 | FULL_SEGMENT_DIAGNOSTIC | 1 | 36247.19099 | 1.994046745 | 27.38869034 | 0.0008485377968 | 0.7839988426 | 0.2160011574 |
| pg19-test-37702 | FULL_SEGMENT_DIAGNOSTIC | 2 | 317050.5012 | 6.218435074 | 138.2920104 | 0.0004291896401 | 0.7950823749 | 0.2049176251 |
| pg19-test-5396 | FULL_SEGMENT_DIAGNOSTIC | 0 | 2206.577887 | 1.035774664 | 5.030734943 | 0.00219389285 | 0.7896999783 | 0.2098117405 |
| pg19-test-5396 | FULL_SEGMENT_DIAGNOSTIC | 1 | 33958.77402 | 2.051745593 | 29.42744408 | 0.0009148394758 | 0.789763274 | 0.210236726 |
| pg19-test-5396 | FULL_SEGMENT_DIAGNOSTIC | 2 | 404550.7317 | 6.955526565 | 212.6155297 | 0.000532775273 | 0.8051317003 | 0.1948682997 |

## Numerical closure

```json
{
  "global_max": {
    "max_abs_residual": 1.1102230246251565e-16,
    "max_energy_abs_residual": 7.63429852668196e-09,
    "max_energy_scaled_residual": 4.496333930483613e-16,
    "relative_L2_residual": 9.331083041648673e-18
  },
  "per_document": {
    "pg19-test-26493": {
      "max_abs_residual": 1.1102230246251565e-16,
      "max_energy_abs_residual": 1.007549599307822e-09,
      "max_energy_scaled_residual": 4.496333930483613e-16,
      "relative_L2_residual": 9.331083041648673e-18
    },
    "pg19-test-28988": {
      "max_abs_residual": 1.1102230246251565e-16,
      "max_energy_abs_residual": 4.762114258483052e-09,
      "max_energy_scaled_residual": 4.006168586217735e-16,
      "relative_L2_residual": 7.94585297057409e-18
    },
    "pg19-test-37702": {
      "max_abs_residual": 1.1102230246251565e-16,
      "max_energy_abs_residual": 3.342019994079237e-10,
      "max_energy_scaled_residual": 4.075950139256283e-16,
      "relative_L2_residual": 9.282901618399175e-18
    },
    "pg19-test-5396": {
      "max_abs_residual": 1.1102230246251565e-16,
      "max_energy_abs_residual": 7.63429852668196e-09,
      "max_energy_scaled_residual": 4.014848006119953e-16,
      "relative_L2_residual": 8.451639082401059e-18
    }
  }
}
```

## Interpretation boundary

E = H + I and the corresponding energy decomposition are mathematical identities;
closure PASS is not predictive or confirmatory evidence. The scientific question
is the size/structure of signed interaction and its effect on rotation ranking.
The actual-Q-versus-native-F injection includes numerical basis/write rounding;
pure QDQ injection and basis discrepancy remain separately audited.
The descriptive label follows the prospectively frozen 32K rule unchanged.
Shorter-horizon diagnostics do not change that rule. These four historical Discovery
documents are not fresh holdout evidence. No reasoning/AIME/training efficacy claim
is made. No follow-on experiment is authorized or started.

## Engineering recovery and artifact locations

The source controller stopped after raw freeze because the remote provenance
directory was absent. The original STOP.json remains byte-identical. The user
explicitly authorized an engineering-only recovery, recorded in
provenance/ENGINEERING_RECOVERY_V1.json. No measurement was repeated and no frozen
scientific source was modified. Protected parent files and inherited inputs were
reverified before and after analysis. The frozen remote SHA inventory was not replaced.
Scalar results, reports and audits are copied byte-identically to the source worktree.
Large FP capture tensors remain on the assigned GPU6 server with their hashes.

Source: /data01/user2/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1
Remote: /data/zypan/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1

Original machine-readable report: FINAL_REPORT.md and analysis/coherence_analysis.json.
Final action: STOP_AFTER_DISCOVERY.
