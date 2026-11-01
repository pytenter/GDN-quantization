# GDN Canonical Evidence Rebaseline V1

## Gate
`PASS`, final classification `CANONICAL_EVIDENCE_REBASELINE_VALIDATED`, method design ready `NO`.

## Legacy Evidence Provenance
```json
{
  "LEGACY_CADENCE": {
    "cadence_values": [
      1,
      4,
      16,
      32
    ],
    "note": "all 24 GDN layers, 64-token horizon in cited dynamic run",
    "quantizer": "INT4-row",
    "source": "/data/zypan/results/gdn_dynamic_sensitivity_validation_v1.json",
    "status": "IDENTIFIED",
    "steps": 64
  },
  "LEGACY_LONG_DECODE_PROTOCOL_STATUS": {
    "note": "free decode; quantize_cache_recurrent_states iterates recurrent states in cache, effectively all recurrent GDN states; no rerun performed",
    "result_file_present": true,
    "source": "/data/zypan/results/long_decode_granularity_ablation.json",
    "status": "PARTIALLY_IDENTIFIED"
  },
  "LEGACY_SELECTED_5_LAYER_QUANTIZER_PROFILE": {
    "horizon": 128,
    "layer_scope": [
      1,
      9,
      17,
      25,
      29
    ],
    "metrics": {
      "INT3-row": 0.09745651757251214,
      "INT4-column": 0.0042469703360867046,
      "INT4-row": 0.05477945233754376,
      "INT5-row": 0.0432007075902512,
      "INT8-column": 0.0002909589788173954
    },
    "prompt_limit": 8,
    "renamed_from": "dynamic_sensitivity quantizer profile",
    "source": "/data/zypan/results/gdn_dynamic_sensitivity_validation_v1.json",
    "status": "IDENTIFIED"
  },
  "protocol_audit_source": "/data/zypan/results/gdn_protocol_consistency_audit_v1.json"
}
```

## Canonical Cadence
|checkpoint|K|KL|Top1|global divergence|mean error|total injection|call count|
|-|-|-|-|-|-|-|-|
|64|1|0.2474213619729797|0.875|0.15712814899953378|0.09218680453528255|15.335683961945318|12288|
|64|4|0.18387492184304696|0.892578125|0.12909128703570524|0.13921867309526959|8.43772495426633|3072|
|64|16|0.13861891438253443|0.916015625|0.09878978464037433|0.19336785347260593|3.915025125097831|768|
|64|32|0.11404950290890382|0.916015625|0.08399014364262529|0.2230881313406687|2.5448162028955883|384|
|128|1|0.4222424483718781|0.837890625|0.1803414340365131|0.08449023963378369|25.439620281951747|24576|
|128|4|0.2651492927100089|0.876953125|0.1434423353589775|0.1262857434078003|13.815673101794644|6144|
|128|16|0.16477904465429452|0.908203125|0.1092107297802971|0.17207415385955138|6.2735822154628105|1536|
|128|32|0.11847591648345183|0.9150390625|0.09294669149782073|0.19866125040422292|4.101382135521716|768|

## Cadence Prompt Consistency
```json
[
  {
    "K1": 0.3958905308588285,
    "K16": 0.10106797159587444,
    "K32": 0.07236147879035565,
    "K32_lt_K1": true,
    "K4": 0.22008119467738219,
    "monotonic": true,
    "prompt_id": 0
  },
  {
    "K1": 0.30807772001995204,
    "K16": 0.13862251987031549,
    "K32": 0.08245968404626151,
    "K32_lt_K1": true,
    "K4": 0.23615400458738378,
    "monotonic": true,
    "prompt_id": 1
  },
  {
    "K1": 0.297594676860399,
    "K16": 0.15074611681211536,
    "K32": 0.1009520698541877,
    "K32_lt_K1": true,
    "K4": 0.19676297041786484,
    "monotonic": true,
    "prompt_id": 2
  },
  {
    "K1": 0.3535462042710922,
    "K16": 0.12427253755220091,
    "K32": 0.126441116058856,
    "K32_lt_K1": true,
    "K4": 0.23406232773995672,
    "monotonic": false,
    "prompt_id": 3
  },
  {
    "K1": 0.26261105743126656,
    "K16": 0.06323370773747261,
    "K32": 0.04769249943427445,
    "K32_lt_K1": true,
    "K4": 0.14721537875478577,
    "monotonic": true,
    "prompt_id": 4
  },
  {
    "K1": 0.7123188404978009,
    "K16": 0.17691420442882588,
    "K32": 0.13776028481765223,
    "K32_lt_K1": true,
    "K4": 0.29843712621300256,
    "monotonic": true,
    "prompt_id": 5
  },
  {
    "K1": 0.801905989201039,
    "K16": 0.5044868213341012,
    "K32": 0.3125214437571824,
    "K32_lt_K1": true,
    "K4": 0.674533620028499,
    "monotonic": true,
    "prompt_id": 6
  },
  {
    "K1": 0.24599456783464646,
    "K16": 0.05888847790345042,
    "K32": 0.06761875510884474,
    "K32_lt_K1": true,
    "K4": 0.1139477192611964,
    "monotonic": false,
    "prompt_id": 7
  }
]
```

## Canonical Profiles
```json
[
  {
    "KL": {
      "max": 1.091934366661917,
      "mean": 0.577854138967572,
      "median": 0.5062733026975295,
      "min": 0.37267440899458393,
      "std": 0.24079151111285818
    },
    "cadence": 1,
    "checkpoint": 64,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.25085590725850404,
      "mean": 0.2300444743934031,
      "median": 0.23347735064258893,
      "min": 0.21112451525290382,
      "std": 0.014087210504238317
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.13364744158729802,
      "mean": 0.12352517444539583,
      "median": 0.12284390426378422,
      "min": 0.11195678192772136,
      "std": 0.006969733747128281
    },
    "prompt_count": 8,
    "quantization_call_count": 12288,
    "quantized_scope_divergence": {
      "max": 0.25085590725850404,
      "mean": 0.2300444743934031,
      "median": 0.23347735064258893,
      "min": 0.21112451525290382,
      "std": 0.014087210504238317
    },
    "quantizer": "INT3-row",
    "theoretical_quantization_call_count": 12288,
    "top1_agreement": {
      "max": 0.890625,
      "mean": 0.8125,
      "median": 0.84375,
      "min": 0.640625,
      "std": 0.0859882155547242
    },
    "total_normalized_squared_injection_energy": {
      "max": 33.42326258523446,
      "mean": 29.073587350343328,
      "median": 28.65691171367446,
      "min": 24.301121135203132,
      "std": 2.9720577079058974
    },
    "total_raw_injection_energy": {
      "max": 5008.846920987591,
      "mean": 4308.37619293062,
      "median": 4237.533711661585,
      "min": 3644.390576913953,
      "std": 412.2860832422982
    }
  },
  {
    "KL": {
      "max": 1.5281124750384834,
      "mean": 0.8541262323988716,
      "median": 0.709860662369568,
      "min": 0.598097193491526,
      "std": 0.32491642456401437
    },
    "cadence": 1,
    "checkpoint": 128,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.3052856178379823,
      "mean": 0.2812739452706311,
      "median": 0.28927236625595376,
      "min": 0.25584926359384275,
      "std": 0.018165807094105396
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.11439229816319306,
      "mean": 0.10811536676698087,
      "median": 0.10785862443061245,
      "min": 0.10246286422028687,
      "std": 0.004750027089896697
    },
    "prompt_count": 8,
    "quantization_call_count": 24576,
    "quantized_scope_divergence": {
      "max": 0.3052856178379823,
      "mean": 0.2812739452706311,
      "median": 0.28927236625595376,
      "min": 0.25584926359384275,
      "std": 0.018165807094105396
    },
    "quantizer": "INT3-row",
    "theoretical_quantization_call_count": 24576,
    "top1_agreement": {
      "max": 0.8359375,
      "mean": 0.7490234375,
      "median": 0.78125,
      "min": 0.5546875,
      "std": 0.09797381402097687
    },
    "total_normalized_squared_injection_energy": {
      "max": 49.23450352443955,
      "mean": 44.00418631109264,
      "median": 43.55007022222358,
      "min": 39.435082804812744,
      "std": 3.5917812819694466
    },
    "total_raw_injection_energy": {
      "max": 9683.81589556858,
      "mean": 8036.314680563519,
      "median": 7705.666058573872,
      "min": 6519.49104321003,
      "std": 1016.525632166277
    }
  },
  {
    "KL": {
      "max": 0.030579087617278643,
      "mean": 0.015297837718836818,
      "median": 0.009344927145523069,
      "min": 0.005659347116265756,
      "std": 0.010430103614741268
    },
    "cadence": 1,
    "checkpoint": 64,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.4069633932483366,
      "mean": 0.37031603854647327,
      "median": 0.3786428413584383,
      "min": 0.3207550848756033,
      "std": 0.033070543907925465
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.09282893415066752,
      "mean": 0.08551235943304943,
      "median": 0.08468854844216625,
      "min": 0.07772182753494604,
      "std": 0.004825124703160962
    },
    "prompt_count": 8,
    "quantization_call_count": 12288,
    "quantized_scope_divergence": {
      "max": 0.4069633932483366,
      "mean": 0.37031603854647327,
      "median": 0.3786428413584383,
      "min": 0.3207550848756033,
      "std": 0.033070543907925465
    },
    "quantizer": "INT4-column",
    "theoretical_quantization_call_count": 12288,
    "top1_agreement": {
      "max": 1.0,
      "mean": 0.982421875,
      "median": 0.984375,
      "min": 0.953125,
      "std": 0.0175936191634313
    },
    "total_normalized_squared_injection_energy": {
      "max": 14.79996980046713,
      "mean": 12.697091851801572,
      "median": 12.337617681597672,
      "min": 10.8267755307243,
      "std": 1.3369915447121317
    },
    "total_raw_injection_energy": {
      "max": 12787.538562072441,
      "mean": 12100.761507826508,
      "median": 12432.988630726002,
      "min": 10884.877820534632,
      "std": 743.6169524428581
    }
  },
  {
    "KL": {
      "max": 0.06914766296419605,
      "mean": 0.025136747059578413,
      "median": 0.017316319563860594,
      "min": 0.0078285655431172,
      "std": 0.021591863231257072
    },
    "cadence": 1,
    "checkpoint": 128,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 1.4050760081329812,
      "mean": 0.6037762481757349,
      "median": 0.48853185909903757,
      "min": 0.39823200795252905,
      "std": 0.331968949974189
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.07981377156815284,
      "mean": 0.07679379814255106,
      "median": 0.0772092661338909,
      "min": 0.0741214522251551,
      "std": 0.0021170634870361074
    },
    "prompt_count": 8,
    "quantization_call_count": 24576,
    "quantized_scope_divergence": {
      "max": 1.4050760081329812,
      "mean": 0.6037762481757349,
      "median": 0.48853185909903757,
      "min": 0.39823200795252905,
      "std": 0.331968949974189
    },
    "quantizer": "INT4-column",
    "theoretical_quantization_call_count": 24576,
    "top1_agreement": {
      "max": 1.0,
      "mean": 0.97265625,
      "median": 0.98828125,
      "min": 0.8984375,
      "std": 0.03340765523905305
    },
    "total_normalized_squared_injection_energy": {
      "max": 22.291653818278313,
      "mean": 20.786740268577645,
      "median": 21.011150451902928,
      "min": 19.11839531850054,
      "std": 1.1872440975200995
    },
    "total_raw_injection_energy": {
      "max": 883818.3205508906,
      "mean": 135159.49966389907,
      "median": 25793.095058741048,
      "min": 21629.914576085284,
      "std": 302641.68806361157
    }
  },
  {
    "KL": {
      "max": 0.5851192089321557,
      "mean": 0.2474213619729797,
      "median": 0.1944301065916889,
      "min": 0.11784663438830378,
      "std": 0.15839611308986304
    },
    "cadence": 1,
    "checkpoint": 64,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.1730922145044484,
      "mean": 0.15712814899953378,
      "median": 0.16022657635945842,
      "min": 0.13960587471354335,
      "std": 0.012256173286019292
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.09953340739254224,
      "mean": 0.09218680453528255,
      "median": 0.09256608963492133,
      "min": 0.0830047316469629,
      "std": 0.005260970646498057
    },
    "prompt_count": 8,
    "quantization_call_count": 12288,
    "quantized_scope_divergence": {
      "max": 0.1730922145044484,
      "mean": 0.15712814899953378,
      "median": 0.16022657635945842,
      "min": 0.13960587471354335,
      "std": 0.012256173286019292
    },
    "quantizer": "INT4-row",
    "theoretical_quantization_call_count": 12288,
    "top1_agreement": {
      "max": 0.9375,
      "mean": 0.875,
      "median": 0.90625,
      "min": 0.6875,
      "std": 0.0847626676162836
    },
    "total_normalized_squared_injection_energy": {
      "max": 17.80392423933009,
      "mean": 15.335683961945318,
      "median": 15.240338347940831,
      "min": 12.623592916225476,
      "std": 1.673346470771514
    },
    "total_raw_injection_energy": {
      "max": 2354.196743723005,
      "mean": 1944.6116972707678,
      "median": 1911.0114112952724,
      "min": 1684.7066419236362,
      "std": 202.2432084616447
    }
  },
  {
    "KL": {
      "max": 0.801905989201039,
      "mean": 0.4222424483718781,
      "median": 0.3308119621455221,
      "min": 0.24599456783464646,
      "std": 0.21341390486357892
    },
    "cadence": 1,
    "checkpoint": 128,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.19311541178650146,
      "mean": 0.1803414340365131,
      "median": 0.1833506925056796,
      "min": 0.16054339781446755,
      "std": 0.011220511801268441
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.08930111199665414,
      "mean": 0.08449023963378369,
      "median": 0.08486258159004255,
      "min": 0.08015130115000807,
      "std": 0.0032424209640730044
    },
    "prompt_count": 8,
    "quantization_call_count": 24576,
    "quantized_scope_divergence": {
      "max": 0.19311541178650146,
      "mean": 0.1803414340365131,
      "median": 0.1833506925056796,
      "min": 0.16054339781446755,
      "std": 0.011220511801268441
    },
    "quantizer": "INT4-row",
    "theoretical_quantization_call_count": 24576,
    "top1_agreement": {
      "max": 0.9140625,
      "mean": 0.837890625,
      "median": 0.875,
      "min": 0.6328125,
      "std": 0.09307326433236422
    },
    "total_normalized_squared_injection_energy": {
      "max": 28.551390862356918,
      "mean": 25.439620281951747,
      "median": 25.572784674018283,
      "min": 22.98763015144651,
      "std": 1.943447602953753
    },
    "total_raw_injection_energy": {
      "max": 4510.215656498447,
      "mean": 3685.9168340236647,
      "median": 3596.377746300306,
      "min": 3047.1961740544066,
      "std": 445.578621795605
    }
  },
  {
    "KL": {
      "max": 0.568545577282066,
      "mean": 0.18387492184304696,
      "median": 0.12620704397164753,
      "min": 0.07033847156114703,
      "std": 0.16850107885538648
    },
    "cadence": 4,
    "checkpoint": 64,
    "failures": 0,
    "global_all_gdn_divergence": {
      "max": 0.13989850879115553,
      "mean": 0.12909128703570524,
      "median": 0.12988063989029858,
      "min": 0.11623163741608958,
      "std": 0.008778800769174613
    },
    "layer_scope": "all_gdn",
    "mean_relative_reconstruction_error": {
      "max": 0.14665255624349546,
      "mean": 0.13921867309526959,
      "median": 0.1394154087623169,
      "min": 0.12815374878610192,
      "std": 0.006487491506016127
    },
    "prompt_count": 8,
    "quantization_call_count": 3072,
    "quantized_scope_divergence": {
      "max": 0.13989850879115553,
      "mean": 0.12909128703570524,
      "median": 0.12988063989029858,
      "min": 0.11623163741608958,
      "std": 0.008778800769174613
    },
    "quantizer": "INT4-row",
    "theoretical_quantization_call_count": 3072,
    "top1_agreement": {
      "max": 0.953125,
      "mean": 0.892578125,
      "median": 0.921875,
      "min": 0.703125,
      "std": 0.08575975937539121
    },
    "total_normalized_squared_injection_energy": {
      "max": 9.337366894089456,
      "mean": 8.43772495426633,
      "median": 8.378708713890322,
      "min": 7.261275415201014,
      "std": 0.7263651241084161
    },
    "total_raw_injection_energy": {
      "max": 1166.9887292645872,
      "mean": 1024.4194541755132,
      "median": 1005.143253704533,
      "min": 943.3179078195244,
      "std": 77.1440558629914
    }
  },
  {
    "KL": {
      "max": 0.674533620028499,
      "mean": 0.2651492927100089,
      "median": 0.22707176120866945,
      "min": 0.1139477192611964,
      "std": 0.17484232890922732
    },
    "cadence": 4,
    "checkpoint": 128,
    "failures": 0,
    "global_all_gdn_divergence": {
      "m
```

## Bootstrap
```json
{
  "INT4_row_vs_INT4_column_KL_all_gdn_128": {
    "ci95": [
      -0.542062659385212,
      -0.2822807144768371
    ],
    "mean_difference_b_minus_a": -0.39710570131229966,
    "metric": "KL",
    "n_pairs": 8
  },
  "K1_vs_K32_KL_all_gdn_128": {
    "ci95": [
      -0.4075536910005634,
      -0.21625695382710075
    ],
    "mean_difference_b_minus_a": -0.30376653188842623,
    "metric": "KL",
    "n_pairs": 8
  },
  "K1_vs_K32_global_divergence_all_gdn_128": {
    "ci95": [
      -0.09359548166915986,
      -0.0813413232335573
    ],
    "mean_difference_b_minus_a": -0.08739474253869238,
    "metric": "global_all_gdn_divergence",
    "n_pairs": 8
  },
  "selected_5_vs_all_gdn_INT4_row_KL_128": {
    "ci95": [
      0.26874541188776185,
      0.48554178481984445
    ],
    "mean_difference_b_minus_a": 0.36746299603433424,
    "metric": "KL",
    "n_pairs": 8
  }
}
```

## Scientific Classification
```json
{
  "ALL_GDN_PROFILE_STATUS": "COMPLETED",
  "BITWIDTH_5_4_3_STATUS": "CANONICALLY_SUPPORTED",
  "CADENCE_STATUS": "CANONICALLY_SUPPORTED",
  "LAYER_SCOPE_STATUS": "CANONICALLY_SUPPORTED",
  "ROW_COLUMN_SHORT_TERM_STATUS": "CANONICALLY_SUPPORTED",
  "SHORT_LONG_HYPOTHESIS_STATUS": "SUPPORTED_SHORT_TERM_SIGNAL_ONLY"
}
```

## Interpretation
This is a short-horizon canonical rebaseline. It does not validate 4096/8192-token behavior and does not propose a method.
