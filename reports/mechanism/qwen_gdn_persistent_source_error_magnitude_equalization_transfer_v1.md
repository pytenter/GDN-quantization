# Qwen GDN Persistent Source Error Magnitude Equalization Transfer V1

## Status
COMPLETE

## Canonical Manifest
Source: `/data/zypan/GDN-quantization/results/propagation/gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json`
SHA256: `230a54d93358074cc9a0b82e6c0891559276ddce3e17028779cef24001b38579`
Units: 18
Execution runner SHA256: `['052b24d4e57e48c07d60910392046efc818c53e22bcf8d70c0edc772d1e9b50f']`
Analysis runner SHA256: `f4f80c5ad9ce005596e6d9810949ef8423f189db2e0941afb6c7e24eab548f2f`

## Environment
Canonical replay inputs and the offline model are available.

## Gates
```json
{
  "APPLIED_MAGNITUDE_GATE": "PASS",
  "C128_QUANTIZER_IDENTITY_GATE": "PASS",
  "CANONICAL_MANIFEST_GATE": "PASS",
  "CURRENT_LOGIT_NONRETROACTIVITY_GATE": "PASS",
  "C_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "INTERVENTION_TIMING_GATE": "PASS",
  "L1_SEMANTICS_GATE": "PASS",
  "NO_FURTHER_QUANTIZATION_GATE": "PASS",
  "R128_QUANTIZER_IDENTITY_GATE": "PASS",
  "R_NATIVE_ENDPOINT_IDENTITY_GATE": "PASS",
  "SAME_STATE_COUNTERFACTUAL_GATE": "PASS",
  "SOURCE_DIRECTION_PRESERVATION_GATE": "PASS",
  "STATE_SEMANTICS_GATE": "PASS"
}
```

## Source Magnitude Orientation
```json
{
  "C_NORM_GT_R": "0 / 108",
  "C_SOURCE_NORM": 2.211046136048981,
  "MEDIAN_R_OVER_C_SOURCE_NORM": 1.919175004677653,
  "QWEN_SOURCE_MAGNITUDE_ORIENTATION": "R_GT_C",
  "R_NORM_GT_C": "108 / 108",
  "R_SOURCE_NORM": 3.245965218550537,
  "per_unit_history_length": [
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.1867268262042607,
      "mean_R_source_norm": 4.390716021225521,
      "median_R_over_C_source_norm": 2.0078941587984978,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.0560343589934256,
      "mean_R_source_norm": 3.185255760665993,
      "median_R_over_C_source_norm": 2.0580112975650313,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.0232013881459876,
      "mean_R_source_norm": 2.9119642149331697,
      "median_R_over_C_source_norm": 1.9782677386322325,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.033140779155155,
      "mean_R_source_norm": 2.827586708875392,
      "median_R_over_C_source_norm": 1.9861129538271491,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.0382919422590065,
      "mean_R_source_norm": 2.7564065937526143,
      "median_R_over_C_source_norm": 1.9732482232943065,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.0710378064725736,
      "mean_R_source_norm": 2.7480080931307524,
      "median_R_over_C_source_norm": 1.945926933733685,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1214.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.296211162873186,
      "mean_R_source_norm": 4.504718238387953,
      "median_R_over_C_source_norm": 1.9618048684814167,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.1884023102794488,
      "mean_R_source_norm": 3.3108948610830327,
      "median_R_over_C_source_norm": 1.9701222620849195,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.1225704461139,
      "mean_R_source_norm": 3.0359110503777686,
      "median_R_over_C_source_norm": 1.9607260183557291,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.142367079545056,
      "mean_R_source_norm": 2.944823299026692,
      "median_R_over_C_source_norm": 1.935233826404711,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.165552498065915,
      "mean_R_source_norm": 2.8987999037864536,
      "median_R_over_C_source_norm": 1.9629343590719315,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.151218400493389,
      "mean_R_source_norm": 2.866890723382362,
      "median_R_over_C_source_norm": 1.9575408084241372,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1214.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.14144114041051,
      "mean_R_source_norm": 4.243432462457199,
      "median_R_over_C_source_norm": 1.981577911425462,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.034769344385182,
      "mean_R_source_norm": 3.117938752442866,
      "median_R_over_C_source_norm": 1.9945271283960972,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.013971911920683,
      "mean_R_source_norm": 2.865713226278646,
      "median_R_over_C_source_norm": 2.012660229803232,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 1.9389265767871742,
      "mean_R_source_norm": 2.68512166271589,
      "median_R_over_C_source_norm": 1.9653366362688238,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 1.9012162801533559,
      "mean_R_source_norm": 2.6026809151514168,
      "median_R_over_C_source_norm": 1.9628703427498086,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4890625,
      "fraction_R_norm_gt_C_norm": 0.5109375,
      "history_length": 64,
      "mean_C_source_norm": 1.8475911436940016,
      "mean_R_source_norm": 2.4989025994764438,
      "median_R_over_C_source_norm": 1.9421958731496218,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1214.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.418221147799845,
      "mean_R_source_norm": 4.713024166178394,
      "median_R_over_C_source_norm": 1.9489632577502125,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.2927667765934396,
      "mean_R_source_norm": 3.4270230092036464,
      "median_R_over_C_source_norm": 1.9372778070960057,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.2638418951990547,
      "mean_R_source_norm": 3.1951331457787,
      "median_R_over_C_source_norm": 1.9957769735280348,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.2443835941131174,
      "mean_R_source_norm": 3.0583432739404617,
      "median_R_over_C_source_norm": 1.919050159972236,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.204412188799139,
      "mean_R_source_norm": 2.9544492476759165,
      "median_R_over_C_source_norm": 1.9524265309440088,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.1731310969204354,
      "mean_R_source_norm": 2.8972777464366803,
      "median_R_over_C_source_norm": 1.9535591787161928,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1332.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.6240780034168307,
      "mean_R_source_norm": 4.853950765954059,
      "median_R_over_C_source_norm": 1.8497738099369923,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.4198901934025496,
      "mean_R_source_norm": 3.495277399416692,
      "median_R_over_C_source_norm": 1.8487414099038404,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.3867736344085873,
      "mean_R_source_norm": 3.24190567142063,
      "median_R_over_C_source_norm": 1.8669787798400854,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.383676895758798,
      "mean_R_source_norm": 3.1309610321703856,
      "median_R_over_C_source_norm": 1.8933839681023141,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.484375,
      "fraction_R_norm_gt_C_norm": 0.515625,
      "history_length": 32,
      "mean_C_source_norm": 2.355710743774039,
      "mean_R_source_norm": 3.061850323893012,
      "median_R_over_C_source_norm": 1.8917208443181235,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4921875,
      "fraction_R_norm_gt_C_norm": 0.5078125,
      "history_length": 64,
      "mean_C_source_norm": 2.3489730361614267,
      "mean_R_source_norm": 3.021556762143831,
      "median_R_over_C_source_norm": 1.8796773628292835,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1332.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.2445301419960155,
      "mean_R_source_norm": 4.555059165448951,
      "median_R_over_C_source_norm": 2.0294043195143727,
      "n_same_state_observations": 10,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.142314773115007,
      "mean_R_source_norm": 3.322944596438392,
      "median_R_over_C_source_norm": 2.0314013436264844,
      "n_same_state_observations": 40,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.1258314435102568,
      "mean_R_source_norm": 3.0505105693738073,
      "median_R_over_C_source_norm": 2.0110114347289128,
      "n_same_state_observations": 80,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.1086062276924613,
      "mean_R_source_norm": 2.917551485731288,
      "median_R_over_C_source_norm": 1.9649763445325554,
      "n_same_state_observations": 160,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.097600976591564,
      "mean_R_source_norm": 2.8500925160704984,
      "median_R_over_C_source_norm": 1.9691037957144149,
      "n_same_state_observations": 320,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.009309547156875,
      "mean_R_source_norm": 2.706450243097191,
      "median_R_over_C_source_norm": 1.9315646441745358,
      "n_same_state_observations": 640,
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.4246814181916148,
      "mean_R_source_norm": 4.5441981734459835,
      "median_R_over_C_source_norm": 1.874142367467508,
      "n_same_state_observations": 10,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.344208624157075,
      "mean_R_source_norm": 3.326111298396365,
      "median_R_over_C_source_norm": 1.854379352975041,
      "n_same_state_observations": 40,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.280211507561474,
      "mean_R_source_norm": 3.081898376822891,
      "median_R_over_C_source_norm": 1.9099665052552615,
      "n_same_state_observations": 80,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.2289880988797255,
      "mean_R_source_norm": 2.9441361490182127,
      "median_R_over_C_source_norm": 1.907252316201611,
      "n_same_state_observations": 160,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.170608254131897,
      "mean_R_source_norm": 2.8387031049152176,
      "median_R_over_C_source_norm": 1.9164720202964332,
      "n_same_state_observations": 320,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.0872955526603825,
      "mean_R_source_norm": 2.739710616729626,
      "median_R_over_C_source_norm": 1.8870961687109122,
      "n_same_state_observations": 640,
      "unit_id": "test/counting_and_probability/119.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.564155856671149,
      "mean_R_source_norm": 4.633072822773334,
      "median_R_over_C_source_norm": 1.8068608469012089,
      "n_same_state_observations": 10,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.4147580776990654,
      "mean_R_source_norm": 3.4110692548048633,
      "median_R_over_C_source_norm": 1.8597151471373323,
      "n_same_state_observations": 40,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.360940596385555,
      "mean_R_source_norm": 3.156498762753648,
      "median_R_over_C_source_norm": 1.879056436352007,
      "n_same_state_observations": 80,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.310629608037308,
      "mean_R_source_norm": 3.0293625185975754,
      "median_R_over_C_source_norm": 1.8882067381105332,
      "n_same_state_observations": 160,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.484375,
      "fraction_R_norm_gt_C_norm": 0.515625,
      "history_length": 32,
      "mean_C_source_norm": 2.2863645213468424,
      "mean_R_source_norm": 2.964258230557176,
      "median_R_over_C_source_norm": 1.8752560333247286,
      "n_same_state_observations": 320,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4921875,
      "fraction_R_norm_gt_C_norm": 0.5078125,
      "history_length": 64,
      "mean_C_source_norm": 2.2602464009156926,
      "mean_R_source_norm": 2.930104211640402,
      "median_R_over_C_source_norm": 1.9218483879509978,
      "n_same_state_observations": 640,
      "unit_id": "test/counting_and_probability/119.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.162160551761514,
      "mean_R_source_norm": 4.139337081846875,
      "median_R_over_C_source_norm": 1.914444826251519,
      "n_same_state_observations": 10,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.07375564823714,
      "mean_R_source_norm": 3.0306906738535933,
      "median_R_over_C_source_norm": 1.8876703909267745,
      "n_same_state_observations": 40,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.0714573888741996,
      "mean_R_source_norm": 2.851305486816831,
      "median_R_over_C_source_norm": 1.8684115492282218,
      "n_same_state_observations": 80,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.064247843969945,
      "mean_R_source_norm": 2.7590917772193104,
      "median_R_over_C_source_norm": 1.91251644921764,
      "n_same_state_observations": 160,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.0645058483511525,
      "mean_R_source_norm": 2.6963048997457335,
      "median_R_over_C_source_norm": 1.8904795068718643,
      "n_same_state_observations": 320,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 1.9753807929471079,
      "mean_R_source_norm": 2.5722580406737015,
      "median_R_over_C_source_norm": 1.8616692142042364,
      "n_same_state_observations": 640,
      "unit_id": "test/counting_and_probability/119.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.2081860813518834,
      "mean_R_source_norm": 4.484895577274318,
      "median_R_over_C_source_norm": 2.0310315399355154,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.087355916708307,
      "mean_R_source_norm": 3.2800570315700575,
      "median_R_over_C_source_norm": 2.0980663350667696,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.061096239179559,
      "mean_R_source_norm": 3.0282608581226103,
      "median_R_over_C_source_norm": 1.9784689679867262,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.0763459018068327,
      "mean_R_source_norm": 2.9227418819791504,
      "median_R_over_C_source_norm": 1.9837398818115226,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.0861591049366917,
      "mean_R_source_norm": 2.86158894573025,
      "median_R_over_C_source_norm": 1.9755106124853705,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.101687730001907,
      "mean_R_source_norm": 2.831428596750094,
      "median_R_over_C_source_norm": 1.9538659658339295,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/477.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.5091951605967595,
      "mean_R_source_norm": 4.708525001935388,
      "median_R_over_C_source_norm": 1.8765080834978531,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.298831294534001,
      "mean_R_source_norm": 3.4127955785456536,
      "median_R_over_C_source_norm": 1.9136412780286975,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.239979202645906,
      "mean_R_source_norm": 3.162502847095385,
      "median_R_over_C_source_norm": 1.9099691140286303,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.2575085122531013,
      "mean_R_source_norm": 3.0439606322239348,
      "median_R_over_C_source_norm": 1.9146690973993041,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.238398474634512,
      "mean_R_source_norm": 2.979327818924653,
      "median_R_over_C_source_norm": 1.9298627306339382,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.2325565895824484,
      "mean_R_source_norm": 2.957312241756388,
      "median_R_over_C_source_norm": 1.93519257301717,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/477.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.269108108888255,
      "mean_R_source_norm": 4.513430367953165,
      "median_R_over_C_source_norm": 1.989076831673094,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.150782098994548,
      "mean_R_source_norm": 3.297002725929757,
      "median_R_over_C_source_norm": 2.008531819407352,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.1372816972894144,
      "mean_R_source_norm": 3.0398001753690225,
      "median_R_over_C_source_norm": 2.009404592413593,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.0774156416804344,
      "mean_R_source_norm": 2.889193632704816,
      "median_R_over_C_source_norm": 1.9897802512293126,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.0697704688690313,
      "mean_R_source_norm": 2.794729435931752,
      "median_R_over_C_source_norm": 1.9698510363146047,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 1.9842632608823947,
      "mean_R_source_norm": 2.6330034891801803,
      "median_R_over_C_source_norm": 1.87497786029,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/477.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.4870779839227324,
      "mean_R_source_norm": 4.601437573932691,
      "median_R_over_C_source_norm": 1.8501380349454275,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.3993592876796344,
      "mean_R_source_norm": 3.3967485354002633,
      "median_R_over_C_source_norm": 1.8401504324171973,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.355528145987097,
      "mean_R_source_norm": 3.1356318914476233,
      "median_R_over_C_source_norm": 1.8535330128398828,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.3394613230107,
      "mean_R_source_norm": 3.023023641607795,
      "median_R_over_C_source_norm": 1.8363696177360973,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.484375,
      "fraction_R_norm_gt_C_norm": 0.515625,
      "history_length": 32,
      "mean_C_source_norm": 2.3135941776921802,
      "mean_R_source_norm": 2.958037481054263,
      "median_R_over_C_source_norm": 1.8147553836752142,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4921875,
      "fraction_R_norm_gt_C_norm": 0.5078125,
      "history_length": 64,
      "mean_C_source_norm": 2.2957057484858328,
      "mean_R_source_norm": 2.892096604057169,
      "median_R_over_C_source_norm": 1.8201525987294094,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/702.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.573057706934887,
      "mean_R_source_norm": 4.72026757826204,
      "median_R_over_C_source_norm": 1.834497363016062,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.387409209570137,
      "mean_R_source_norm": 3.382830874241665,
      "median_R_over_C_source_norm": 1.8246526800633338,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.3293405047181053,
      "mean_R_source_norm": 3.167406334749285,
      "median_R_over_C_source_norm": 1.8394267447380663,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.3257416654230267,
      "mean_R_source_norm": 3.06841067807394,
      "median_R_over_C_source_norm": 1.79624439963211,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.484375,
      "fraction_R_norm_gt_C_norm": 0.515625,
      "history_length": 32,
      "mean_C_source_norm": 2.308159536313875,
      "mean_R_source_norm": 2.99482179601804,
      "median_R_over_C_source_norm": 1.8621486812580055,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4921875,
      "fraction_R_norm_gt_C_norm": 0.5078125,
      "history_length": 64,
      "mean_C_source_norm": 2.2872004102868866,
      "mean_R_source_norm": 2.944327084582732,
      "median_R_over_C_source_norm": 1.8922594021377268,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/702.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.515059135489732,
      "mean_R_source_norm": 4.542871365743735,
      "median_R_over_C_source_norm": 1.806268211207424,
      "n_same_state_observations": 10,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.343910525748869,
      "mean_R_source_norm": 3.2821505294177045,
      "median_R_over_C_source_norm": 1.8002730520546135,
      "n_same_state_observations": 40,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.3249903532678453,
      "mean_R_source_norm": 3.05768658154208,
      "median_R_over_C_source_norm": 1.8348351791039628,
      "n_same_state_observations": 80,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.46875,
      "fraction_R_norm_gt_C_norm": 0.53125,
      "history_length": 16,
      "mean_C_source_norm": 2.332263583131454,
      "mean_R_source_norm": 2.960433429447955,
      "median_R_over_C_source_norm": 1.7798800543321438,
      "n_same_state_observations": 160,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.484375,
      "fraction_R_norm_gt_C_norm": 0.515625,
      "history_length": 32,
      "mean_C_source_norm": 2.3105529613003046,
      "mean_R_source_norm": 2.907609573357042,
      "median_R_over_C_source_norm": 1.8815038816191496,
      "n_same_state_observations": 320,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.240710869755806,
      "mean_R_source_norm": 2.824072310778147,
      "median_R_over_C_source_norm": 1.8651046749400286,
      "n_same_state_observations": 640,
      "unit_id": "test/geometry/702.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.387834564684509,
      "mean_R_source_norm": 4.56642692273308,
      "median_R_over_C_source_norm": 1.9123715647086732,
      "n_same_state_observations": 10,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.1684774418114565,
      "mean_R_source_norm": 3.2807943439454244,
      "median_R_over_C_source_norm": 1.9106373422735865,
      "n_same_state_observations": 40,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.1535441788843004,
      "mean_R_source_norm": 3.031582799096479,
      "median_R_over_C_source_norm": 1.9192998493830697,
      "n_same_state_observations": 80,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.159178889175971,
      "mean_R_source_norm": 2.897206193494357,
      "median_R_over_C_source_norm": 1.9239911086717414,
      "n_same_state_observations": 160,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.146758841592805,
      "mean_R_source_norm": 2.835199267351441,
      "median_R_over_C_source_norm": 1.9331298766720235,
      "n_same_state_observations": 320,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.1325146159800563,
      "mean_R_source_norm": 2.793790177115228,
      "median_R_over_C_source_norm": 1.9421050081946214,
      "n_same_state_observations": 640,
      "unit_id": "test/intermediate_algebra/207.json|128"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.450916211833182,
      "mean_R_source_norm": 4.667884813776033,
      "median_R_over_C_source_norm": 1.9045468756693018,
      "n_same_state_observations": 10,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.375,
      "fraction_R_norm_gt_C_norm": 0.625,
      "history_length": 4,
      "mean_C_source_norm": 2.278503146018138,
      "mean_R_source_norm": 3.343065565149866,
      "median_R_over_C_source_norm": 1.8842914481376796,
      "n_same_state_observations": 40,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4375,
      "fraction_R_norm_gt_C_norm": 0.5625,
      "history_length": 8,
      "mean_C_source_norm": 2.2434230296611117,
      "mean_R_source_norm": 3.1034144438584503,
      "median_R_over_C_source_norm": 1.923952112079623,
      "n_same_state_observations": 80,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 2.2259977360667764,
      "mean_R_source_norm": 2.9785364938597048,
      "median_R_over_C_source_norm": 1.9244139599540118,
      "n_same_state_observations": 160,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 2.194608366361067,
      "mean_R_source_norm": 2.898228515139441,
      "median_R_over_C_source_norm": 1.921766055519236,
      "n_same_state_observations": 320,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 2.1971570808164747,
      "mean_R_source_norm": 2.8636624729781492,
      "median_R_over_C_source_norm": 1.9132232656904804,
      "n_same_state_observations": 640,
      "unit_id": "test/intermediate_algebra/207.json|256"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.0,
      "fraction_R_norm_gt_C_norm": 1.0,
      "history_length": 1,
      "mean_C_source_norm": 2.152786648925459,
      "mean_R_source_norm": 4.229946224770957,
      "median_R_over_C_source_norm": 1.964870149524723,
      "n_same_state_observations": 10,
      "unit_id": "test/intermediate_algebra/207.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.35,
      "fraction_R_norm_gt_C_norm": 0.65,
      "history_length": 4,
      "mean_C_source_norm": 2.070620129711105,
      "mean_R_source_norm": 3.079578689797506,
      "median_R_over_C_source_norm": 1.9386990128848123,
      "n_same_state_observations": 40,
      "unit_id": "test/intermediate_algebra/207.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.425,
      "fraction_R_norm_gt_C_norm": 0.575,
      "history_length": 8,
      "mean_C_source_norm": 2.007063052383797,
      "mean_R_source_norm": 2.8391271638752418,
      "median_R_over_C_source_norm": 1.9937547744099235,
      "n_same_state_observations": 80,
      "unit_id": "test/intermediate_algebra/207.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.4625,
      "fraction_R_norm_gt_C_norm": 0.5375,
      "history_length": 16,
      "mean_C_source_norm": 1.9903008935477953,
      "mean_R_source_norm": 2.731192547715576,
      "median_R_over_C_source_norm": 1.9481655756334644,
      "n_same_state_observations": 160,
      "unit_id": "test/intermediate_algebra/207.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.48125,
      "fraction_R_norm_gt_C_norm": 0.51875,
      "history_length": 32,
      "mean_C_source_norm": 1.9949110765542677,
      "mean_R_source_norm": 2.666020979516122,
      "median_R_over_C_source_norm": 1.886411893427368,
      "n_same_state_observations": 320,
      "unit_id": "test/intermediate_algebra/207.json|64"
    },
    {
      "fraction_C_norm_gt_R_norm": 0.490625,
      "fraction_R_norm_gt_C_norm": 0.509375,
      "history_length": 64,
      "mean_C_source_norm": 1.9420218725860856,
      "mean_R_source_norm": 2.560927598460103,
      "median_R_over_C_source_norm": 1.8766945348423447,
      "n_same_state_observations": 640,
      "unit_id": "test/intermediate_algebra/207.json|64"
    }
  ]
}
```

## Primary Contrasts By History Length
```json
{
  "C_GRAFT_VS_R_NATIVE_BY_L": {
    "1": {
      "bootstrap_ci": [
        -0.0007248426763235422,
        -0.00045530557779279904
      ],
      "mean": -0.000583333645185709,
      "median": -0.000502290108203518,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.00029875475416543765,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        -0.0019683090341047755,
        -0.0010912607445305823
      ],
      "mean": -0.0015005311930139167,
      "median": -0.001280759060041216,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.0010215478942991892,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        -0.0027798631404547993,
        -0.0016821590842931644
      ],
      "mean": -0.002179570827217753,
      "median": -0.001914624405862078,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.0012455031406516874,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        -0.000975310086109844,
        -0.0006256962260715805
      ],
      "mean": -0.0007985724847637161,
      "median": -0.0007401837592983524,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.0003854061757429238,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        -0.005040037604486132,
        -0.0036850272226690455
      ],
      "mean": -0.004366396886201464,
      "median": -0.0041371372532074974,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.0015224468035414565,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        -0.0011477271081227615,
        -0.0006759989355111721
      ],
      "mean": -0.000900177984564121,
      "median": -0.0009134399145383909,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.0005092765147576175,
      "ties": 0
    }
  },
  "C_RMAG_GRAFT_DAMAGE_BY_L": {
    "1": {
      "bootstrap_ci": [
        7.205060972237149e-06,
        5.8233018377658224e-05
      ],
      "mean": 3.253672203668794e-05,
      "median": 2.2060177484379043e-05,
      "n": 18,
      "negative": 4,
      "paired_sign_p": 0.0308837890625,
      "positive": 14,
      "std": 5.689751667986211e-05,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        6.862237678196859e-05,
        0.00017496847348052967
      ],
      "mean": 0.00011814135549781157,
      "median": 9.276483556158995e-05,
      "n": 18,
      "negative": 2,
      "paired_sign_p": 0.001312255859375,
      "positive": 16,
      "std": 0.00011635378366053036,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        0.00012405174364085996,
        0.0002900357888075237
      ],
      "mean": 0.0002006323424082037,
      "median": 0.0001306332901186367,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0001834901870135726,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        4.172969638584308e-06,
        6.430246496996196e-05
      ],
      "mean": 3.562734612782431e-05,
      "median": 4.041565025797793e-05,
      "n": 18,
      "negative": 5,
      "paired_sign_p": 0.09625244140625,
      "positive": 13,
      "std": 6.697472479663986e-05,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        0.00021891547664255407,
        0.0005534153271818128
      ],
      "mean": 0.00036973477621505,
      "median": 0.0002757292951145119,
      "n": 18,
      "negative": 1,
      "paired_sign_p": 0.00014495849609375,
      "positive": 17,
      "std": 0.0003683444642323249,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        3.069906271592584e-05,
        9.105963731446525e-05
      ],
      "mean": 6.187252859211404e-05,
      "median": 4.650808339394402e-05,
      "n": 18,
      "negative": 2,
      "paired_sign_p": 0.001312255859375,
      "positive": 16,
      "std": 6.713020863776405e-05,
      "ties": 0
    }
  },
  "MAGNITUDE_MAIN_BY_L": {
    "1": {
      "bootstrap_ci": [
        0.0001829456974316711,
        0.0003025186135232521
      ],
      "mean": 0.00023932471386561328,
      "median": 0.00021955374390680477,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.00013099571881838162,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        -0.00015189814196837613,
        5.9428585418356316e-05
      ],
      "mean": -4.247280425491059e-05,
      "median": -1.2494733712030205e-05,
      "n": 18,
      "negative": 10,
      "paired_sign_p": 0.8145294189453125,
      "positive": 8,
      "std": 0.00023380324609572153,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        -0.0012215894041475814,
        -0.00025299532112876903
      ],
      "mean": -0.000655068200643148,
      "median": -0.0004727252446643077,
      "n": 18,
      "negative": 15,
      "paired_sign_p": 0.007537841796875,
      "positive": 3,
      "std": 0.0011273434562039015,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        -8.974609424056681e-05,
        1.5711233594513554e-05
      ],
      "mean": -3.5253479122620616e-05,
      "median": -1.0758637054609661e-05,
      "n": 18,
      "negative": 9,
      "paired_sign_p": 1.0,
      "positive": 9,
      "std": 0.0001189535839580674,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        -0.004645630588441232,
        -0.00028738537381959304
      ],
      "mean": -0.0018794704456274963,
      "median": -0.0006495678405820974,
      "n": 18,
      "negative": 13,
      "paired_sign_p": 0.09625244140625,
      "positive": 5,
      "std": 0.005649826939237627,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        -9.913662347968506e-05,
        2.8521247042675498e-05
      ],
      "mean": -3.530018300218786e-05,
      "median": -4.023264651156694e-05,
      "n": 18,
      "negative": 12,
      "paired_sign_p": 0.237884521484375,
      "positive": 6,
      "std": 0.00014509593103829714,
      "ties": 0
    }
  },
  "NATIVE_R_MINUS_C_GAP_BY_L": {
    "1": {
      "bootstrap_ci": [
        0.00048422905506778657,
        0.0007573858256623141
      ],
      "mean": 0.0006158703672223969,
      "median": 0.0005229701847023309,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.00030342718503032244,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        0.00118875209091891,
        0.002099207491474972
      ],
      "mean": 0.0016186725485117283,
      "median": 0.001403001500706672,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0010465022787353787,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        0.0018388883068733003,
        0.0030116726716222504
      ],
      "mean": 0.0023802031696259565,
      "median": 0.002015918500554296,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0013290691163039658,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        0.0006578807281131373,
        0.001009025699760191
      ],
      "mean": 0.0008341998308915408,
      "median": 0.0007814714346033721,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.000383962682487617,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        0.004066066059958024,
        0.005445531176618513
      ],
      "mean": 0.004736131662416513,
      "median": 0.004445050051394811,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0015562566237343964,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        0.0007423868207383582,
        0.0011901783594349005
      ],
      "mean": 0.0009620505131562353,
      "median": 0.0010151376013164347,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0004924771120680548,
      "ties": 0
    }
  },
  "R_EQ_VS_C_NATIVE_BY_L": {
    "1": {
      "bootstrap_ci": [
        0.00013132873914015655,
        0.00020911244471696837
      ],
      "mean": 0.00016975766152785843,
      "median": 0.00015781080927855696,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 8.775698400859297e-05,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        0.0013430391799491683,
        0.0023704347729113437
      ],
      "mean": 0.0018217595125193606,
      "median": 0.0014158518114006794,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0011563630072005256,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        0.0028310013405265164,
        0.005160813790737907
      ],
      "mean": 0.0038909719133204557,
      "median": 0.003172105069235977,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0026531901985979646,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        0.000757303151864023,
        0.0011400171695672527
      ],
      "mean": 0.0009403341352646063,
      "median": 0.0008498391690280519,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.00042096575034590003,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        0.005454674614358594,
        0.014670772663096826
      ],
      "mean": 0.008864807329886556,
      "median": 0.006072393189617826,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.011581759829222616,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        0.0008628935918240956,
        0.0013529971024456243
      ],
      "mean": 0.0010945234077527248,
      "median": 0.0011144441573010956,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0005466504607143418,
      "ties": 0
    }
  },
  "R_MAG_EQUALIZATION_RESCUE_BY_L": {
    "1": {
      "bootstrap_ci": [
        0.00033790585093297957,
        0.0005652826135097231
      ],
      "mean": 0.00044611270569453853,
      "median": 0.0004099359290196389,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.0002479731767704279,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        -0.0004464514131523505,
        2.658496796969127e-05
      ],
      "mean": -0.00020308696400763273,
      "median": -9.148038092319534e-05,
      "n": 18,
      "negative": 11,
      "paired_sign_p": 0.480682373046875,
      "positive": 7,
      "std": 0.0005213720761975793,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        -0.002650388613160397,
        -0.0007181233798071268
      ],
      "mean": -0.0015107687436945,
      "median": -0.0010672588830478147,
      "n": 18,
      "negative": 16,
      "paired_sign_p": 0.001312255859375,
      "positive": 2,
      "std": 0.0022287024688003704,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        -0.00020400329807405293,
        -1.93622422222134e-05
      ],
      "mean": -0.00010613430437306554,
      "median": -0.00010628826126737553,
      "n": 18,
      "negative": 12,
      "paired_sign_p": 0.237884521484375,
      "positive": 6,
      "std": 0.0002048335404969273,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        -0.009641673265499598,
        -0.0009519706616722021
      ],
      "mean": -0.004128675667470043,
      "median": -0.0017670101828222004,
      "n": 18,
      "negative": 16,
      "paired_sign_p": 0.001312255859375,
      "positive": 2,
      "std": 0.011242506420339375,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        -0.00024185043580269588,
        -2.5116293230875082e-05
      ],
      "mean": -0.00013247289459648976,
      "median": -0.0001579235575014167,
      "n": 18,
      "negative": 14,
      "paired_sign_p": 0.0308837890625,
      "positive": 4,
      "std": 0.00024638467604687055,
      "ties": 0
    }
  },
  "STATE_ERROR_GRAFT_DAMAGE_BY_L": {
    "1": {
      "bootstrap_ci": [
        2.1251175041042067,
        2.208229402208701
      ],
      "mean": 2.1665425427958582,
      "median": 2.1889610611239383,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.09343356638728008,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        5.792004621147533,
        6.3486307323629845
      ],
      "mean": 6.0693340968388,
      "median": 5.93798516067025,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.6238647741550897,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        8.035801041462042,
        8.609702418512917
      ],
      "mean": 8.31979332849236,
      "median": 8.499980018984129,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.6488457991794805,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        3.1629917287958356,
        3.5098922110909196
      ],
      "mean": 3.3356969601214503,
      "median": 3.295446614101835,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.3795103006085223,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        10.433627096685056,
        11.047473175231065
      ],
      "mean": 10.746182888792038,
      "median": 10.742921686521859,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.6821918814845176,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        4.260625241453637,
        4.684529081509521
      ],
      "mean": 4.483652160271223,
      "median": 4.478782783411194,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.4661866474175666,
      "ties": 0
    }
  },
  "STATE_ERROR_RESCUE_BY_L": {
    "1": {
      "bootstrap_ci": [
        2.1251174831136104,
        2.2082293610020973
      ],
      "mean": 2.166542546833237,
      "median": 2.188961122282774,
      "n": 18,
      "negative": 0,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 18,
      "std": 0.09343353531889984,
      "ties": 0
    },
    "16": {
      "bootstrap_ci": [
        -1.1925572815702046,
        -0.9711175661025939
      ],
      "mean": -1.0916292416852775,
      "median": -1.1357162133945677,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.2437419672112196,
      "ties": 0
    },
    "32": {
      "bootstrap_ci": [
        -1.8426250259523504,
        -1.6293415226838353
      ],
      "mean": -1.7361397490595267,
      "median": -1.7764844810787972,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.2416590622236424,
      "ties": 0
    },
    "4": {
      "bootstrap_ci": [
        -0.5142454686943226,
        -0.25762442925144646
      ],
      "mean": -0.38512351828973446,
      "median": -0.41735221917759624,
      "n": 18,
      "negative": 16,
      "paired_sign_p": 0.001312255859375,
      "positive": 2,
      "std": 0.2910356517539629,
      "ties": 0
    },
    "64": {
      "bootstrap_ci": [
        -2.542672714949824,
        -2.255312241200795
      ],
      "mean": -2.4056016223588657,
      "median": -2.4737768846000554,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.32017408560432475,
      "ties": 0
    },
    "8": {
      "bootstrap_ci": [
        -0.9432245193134953,
        -0.6722423959907624
      ],
      "mean": -0.7988550057170204,
      "median": -0.8455550073455069,
      "n": 18,
      "negative": 18,
      "paired_sign_p": 7.62939453125e-06,
      "positive": 0,
      "std": 0.29618086224022006,
      "ties": 0
    }
  }
}
```

## Branch Completion
```json
{
  "C_DIR_G0": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "C_DIR_G0p25": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "C_DIR_G0p5": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "C_DIR_G0p75": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "C_DIR_G1": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "FP": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "R_DIR_G0": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "R_DIR_G0p25": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "R_DIR_G0p5": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "R_DIR_G0p75": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  },
  "R_DIR_G1": {
    "n_expected": 108,
    "n_success": 108,
    "status": "108 / 108"
  }
}
```

## Gamma Dose Response
```json
{
  "C_DIR": {
    "by_history_length": {
      "1": {
        "endpoint_effect": {
          "bootstrap_ci": [
            7.205060972237149e-06,
            5.8233018377658224e-05
          ],
          "mean": 3.253672203668794e-05,
          "median": 2.2060177484379043e-05,
          "n": 18,
          "negative": 4,
          "paired_sign_p": 0.0308837890625,
          "positive": 14,
          "std": 5.689751667986211e-05,
          "ties": 0
        },
        "median_spearman_rho": 0.49999999999994993,
        "monotonic_step_counts": 42,
        "monotonic_unit_count": 1,
        "n_units": 18,
        "positive_rho_count": 13
      },
      "16": {
        "endpoint_effect": {
          "bootstrap_ci": [
            6.862237678196859e-05,
            0.00017496847348052967
          ],
          "mean": 0.00011814135549781157,
          "median": 9.276483556158995e-05,
          "n": 18,
          "negative": 2,
          "paired_sign_p": 0.001312255859375,
          "positive": 16,
          "std": 0.00011635378366053036,
          "ties": 0
        },
        "median_spearman_rho": 0.549999999999945,
        "monotonic_step_counts": 47,
        "monotonic_unit_count": 2,
        "n_units": 18,
        "positive_rho_count": 15
      },
      "32": {
        "endpoint_effect": {
          "bootstrap_ci": [
            0.00012405174364085996,
            0.0002900357888075237
          ],
          "mean": 0.0002006323424082037,
          "median": 0.0001306332901186367,
          "n": 18,
          "negative": 0,
          "paired_sign_p": 7.62939453125e-06,
          "positive": 18,
          "std": 0.0001834901870135726,
          "ties": 0
        },
        "median_spearman_rho": 0.5999999999999399,
        "monotonic_step_counts": 48,
        "monotonic_unit_count": 2,
        "n_units": 18,
        "positive_rho_count": 18
      },
      "4": {
        "endpoint_effect": {
          "bootstrap_ci": [
            4.172969638584308e-06,
            6.430246496996196e-05
          ],
          "mean": 3.562734612782431e-05,
          "median": 4.041565025797793e-05,
          "n": 18,
          "negative": 5,
          "paired_sign_p": 0.09625244140625,
          "positive": 13,
          "std": 6.697472479663986e-05,
          "ties": 0
        },
        "median_spearman_rho": 0.24999999999997496,
        "monotonic_step_counts": 41,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 12
      },
      "64": {
        "endpoint_effect": {
          "bootstrap_ci": [
            0.00021891547664255407,
            0.0005534153271818128
          ],
          "mean": 0.00036973477621505,
          "median": 0.0002757292951145119,
          "n": 18,
          "negative": 1,
          "paired_sign_p": 0.00014495849609375,
          "positive": 17,
          "std": 0.0003683444642323249,
          "ties": 0
        },
        "median_spearman_rho": 0.8999999999999099,
        "monotonic_step_counts": 54,
        "monotonic_unit_count": 5,
        "n_units": 18,
        "positive_rho_count": 15
      },
      "8": {
        "endpoint_effect": {
          "bootstrap_ci": [
            3.069906271592584e-05,
            9.105963731446525e-05
          ],
          "mean": 6.187252859211404e-05,
          "median": 4.650808339394402e-05,
          "n": 18,
          "negative": 2,
          "paired_sign_p": 0.001312255859375,
          "positive": 16,
          "std": 6.713020863776405e-05,
          "ties": 0
        },
        "median_spearman_rho": 0.39999999999995994,
        "monotonic_step_counts": 42,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 14
      }
    },
    "endpoint_effect": {
      "bootstrap_ci": [
        9.827810165680561e-05,
        0.00017949191722847508
      ],
      "mean": 0.00013642417847961532,
      "median": 8.639636620222969e-05,
      "n": 108,
      "negative": 14,
      "paired_sign_p": 1.0076928692422089e-15,
      "positive": 94,
      "std": 0.00021299292314761186,
      "ties": 0
    },
    "median_spearman_rho": 0.49999999999994993,
    "monotonic_step_counts": 274,
    "monotonic_unit_count": 10,
    "n_units": 108,
    "positive_rho_count": 87,
    "source_direction": "C_DIR"
  },
  "R_DIR": {
    "by_history_length": {
      "1": {
        "endpoint_effect": {
          "bootstrap_ci": [
            0.00033790585093297957,
            0.0005652826135097231
          ],
          "mean": 0.00044611270569453853,
          "median": 0.0004099359290196389,
          "n": 18,
          "negative": 0,
          "paired_sign_p": 7.62939453125e-06,
          "positive": 18,
          "std": 0.0002479731767704279,
          "ties": 0
        },
        "median_spearman_rho": 0.9499999999999049,
        "monotonic_step_counts": 61,
        "monotonic_unit_count": 9,
        "n_units": 18,
        "positive_rho_count": 18
      },
      "16": {
        "endpoint_effect": {
          "bootstrap_ci": [
            -0.0004464514131523505,
            2.658496796969127e-05
          ],
          "mean": -0.00020308696400763273,
          "median": -9.148038092319534e-05,
          "n": 18,
          "negative": 11,
          "paired_sign_p": 0.480682373046875,
          "positive": 7,
          "std": 0.0005213720761975793,
          "ties": 0
        },
        "median_spearman_rho": -0.24999999999997496,
        "monotonic_step_counts": 29,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 7
      },
      "32": {
        "endpoint_effect": {
          "bootstrap_ci": [
            -0.002650388613160397,
            -0.0007181233798071268
          ],
          "mean": -0.0015107687436945,
          "median": -0.0010672588830478147,
          "n": 18,
          "negative": 16,
          "paired_sign_p": 0.001312255859375,
          "positive": 2,
          "std": 0.0022287024688003704,
          "ties": 0
        },
        "median_spearman_rho": -0.5999999999999399,
        "monotonic_step_counts": 24,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 2
      },
      "4": {
        "endpoint_effect": {
          "bootstrap_ci": [
            -0.00020400329807405293,
            -1.93622422222134e-05
          ],
          "mean": -0.00010613430437306554,
          "median": -0.00010628826126737553,
          "n": 18,
          "negative": 12,
          "paired_sign_p": 0.237884521484375,
          "positive": 6,
          "std": 0.0002048335404969273,
          "ties": 0
        },
        "median_spearman_rho": -0.549999999999945,
        "monotonic_step_counts": 33,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 6
      },
      "64": {
        "endpoint_effect": {
          "bootstrap_ci": [
            -0.009641673265499598,
            -0.0009519706616722021
          ],
          "mean": -0.004128675667470043,
          "median": -0.0017670101828222004,
          "n": 18,
          "negative": 16,
          "paired_sign_p": 0.001312255859375,
          "positive": 2,
          "std": 0.011242506420339375,
          "ties": 0
        },
        "median_spearman_rho": -0.5999999999999399,
        "monotonic_step_counts": 26,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 3
      },
      "8": {
        "endpoint_effect": {
          "bootstrap_ci": [
            -0.00024185043580269588,
            -2.5116293230875082e-05
          ],
          "mean": -0.00013247289459648976,
          "median": -0.0001579235575014167,
          "n": 18,
          "negative": 14,
          "paired_sign_p": 0.0308837890625,
          "positive": 4,
          "std": 0.00024638467604687055,
          "ties": 0
        },
        "median_spearman_rho": -0.49999999999994993,
        "monotonic_step_counts": 33,
        "monotonic_unit_count": 0,
        "n_units": 18,
        "positive_rho_count": 4
      }
    },
    "endpoint_effect": {
      "bootstrap_ci": [
        -0.0019465672740367783,
        -0.0003183363767496704
      ],
      "mean": -0.0009391709780745321,
      "median": -0.00014903025866146857,
      "n": 108,
      "negative": 69,
      "paired_sign_p": 0.0050235183334827235,
      "positive": 39,
      "std": 0.004831790673300387,
      "ties": 0
    },
    "median_spearman_rho": -0.29999999999996996,
    "monotonic_step_counts": 206,
    "monotonic_unit_count": 9,
    "n_units": 108,
    "positive_rho_count": 40,
    "source_direction": "R_DIR"
  }
}
```

## Scientific Decision
- Qwen persistent source magnitude causal: `PARTIAL`
- Ling-to-Qwen transfer: `PARTIAL`
- Residual functional geometry required: `YES`
- Qwen functional geometry dominance: `REQUIRED_ADDITIONAL_FACTOR`
- Cross-architecture common magnitude mechanism: `PARTIAL`
- Cross-architecture hypothesis: `H2_SHARED_MULTIFACTOR_MECHANISM_WITH_QWEN_FUNCTIONAL_GEOMETRY`
- Final classification: `QWEN_RECURRENT_MAGNITUDE_ACCUMULATION_PARTIALLY_SUPPORTED`

### Classification Evidence
```json
{
  "dose_consistent": false,
  "graft_fraction_of_native_gap": 0.062030638994933295,
  "history_length": 64,
  "native_median": 0.004445050051394811,
  "rescue_fraction_of_native_gap": -0.397523123923027,
  "residual_fraction_of_native_gap": 1.3661023201181433,
  "source_magnitude_aligned": true,
  "state_error_tracks": false,
  "strong_thresholds_met": false
}
```

## Cross-Architecture Comparison
| Factor | Qwen/GDN | Ling/KDA |
|---|---|---|
| Native worse orientation | R | C |
| Source magnitude orientation | R_GT_C | C > R |
| Magnitude causal effect | PARTIAL | STRONG |
| Direction harmfulness | PRIOR_STRONG | NOT_SUPPORTED_AS_PRIMARY |
| Temporal accumulation | MEASURED_THIS_RUN | STRONG |
| Temporal coherence | NOT_PRIMARY_THIS_RUN | PARTIAL |
| Functional geometry | PRIOR_STRONG | WEAK_NOT_PRIMARY |
| Closed-loop magnitude closure | QWEN_RECURRENT_MAGNITUDE_ACCUMULATION_PARTIALLY_SUPPORTED | STRONG |

METHOD_DESIGN_READY = NO

## Required Artifacts
- `/data/zypan/GDN-quantization/runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1/results.jsonl`
- `/data/zypan/GDN-quantization/runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1/aggregate_summary.json`
- `/data/zypan/GDN-quantization/runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1/report.md`
