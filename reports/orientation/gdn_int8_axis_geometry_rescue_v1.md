# GDN INT8 Axis Geometry Rescue Causal Diagnostic V1

## Observation
- stage: `pilot`
- FORMAL_STATUS: `NOT_RUN`
- PROTOCOL_GATE: `PASS`; METRIC_GATE: `PASS`
- METHOD_DESIGN_READY: `NO`
- METHOD_DESIGN_READY_CANDIDATE: `NO`
- AXIS_GEOMETRY_CAUSAL_SUPPORT: `NO`
- FINAL_CLASSIFICATION: `AXIS_GEOMETRY_CAUSAL_HYPOTHESIS_NOT_SUPPORTED`

## Table 1: Quantization Geometry
```json
[
  {
    "bit_width": 8,
    "config": "R128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "R16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "row",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C128",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 128,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 128,
    "scale_count_per_layer": 4096,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C64",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 64,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 256,
    "scale_count_per_layer": 8192,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C32",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 32,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 512,
    "scale_count_per_layer": 16384,
    "zero_point": 0
  },
  {
    "bit_width": 8,
    "config": "C16",
    "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
    "group_size": 16,
    "orientation": "column",
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale_count_per_head": 1024,
    "scale_count_per_layer": 32768,
    "zero_point": 0
  }
]
```

## Table 2: Global Metrics
```json
{
  "C128": {
    "geometry": {
      "bit_width": 8,
      "config": "C128",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 128,
      "orientation": "column",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 128,
      "scale_count_per_layer": 4096,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.0009054763947956287,
        "median": 0.0009054763947956287,
        "p90": 0.0009054763947956287,
        "p95": 0.0009054763947956287
      },
      "delta_norm": {
        "mean": 2.5989234479899177,
        "median": 2.5989234479899177,
        "p90": 2.5989234479899177,
        "p95": 2.5989234479899177
      },
      "lost_update_fraction": {
        "mean": 0.03607998771443544,
        "median": 0.03607998771443544,
        "p90": 0.03607998771443544,
        "p95": 0.03607998771443544
      },
      "represented_update_norm_ratio": {
        "mean": 0.963346796323126,
        "median": 0.963346796323126,
        "p90": 0.963346796323126,
        "p95": 0.963346796323126
      },
      "residual_update_cosine": {
        "mean": -0.017618137344191737,
        "median": -0.017618137344191737,
        "p90": -0.017618137344191737,
        "p95": -0.017618137344191737
      },
      "residual_update_projection_coeff": {
        "mean": -0.0017833629956718836,
        "median": -0.0017833629956718836,
        "p90": -0.0017833629956718836,
        "p95": -0.0017833629956718836
      },
      "saturation_fraction": {
        "mean": 0.00791986581164097,
        "median": 0.00791986581164097,
        "p90": 0.00791986581164097,
        "p95": 0.00791986581164097
      },
      "scale_max": {
        "mean": 0.028478025487021597,
        "median": 0.028478025487021597,
        "p90": 0.028478025487021597,
        "p95": 0.028478025487021597
      },
      "scale_mean": {
        "mean": 0.00029104354788321986,
        "median": 0.00029104354788321986,
        "p90": 0.00029104354788321986,
        "p95": 0.00029104354788321986
      },
      "scale_median": {
        "mean": 0.00013247547944035686,
        "median": 0.00013247547944035686,
        "p90": 0.00013247547944035686,
        "p95": 0.00013247547944035686
      },
      "scale_p95_over_median": {
        "mean": 6.319217541457384,
        "median": 6.319217541457384,
        "p90": 6.319217541457384,
        "p95": 6.319217541457384
      },
      "state_change_error_E_Delta": {
        "mean": 0.22644575256815908,
        "median": 0.22644575256815908,
        "p90": 0.22644575256815908,
        "p95": 0.22644575256815908
      },
      "state_norm": {
        "mean": 19.448504131464748,
        "median": 19.448504131464748,
        "p90": 19.448504131464748,
        "p95": 19.448504131464748
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.008128743745961396,
        "median": 0.008128743745961396,
        "p90": 0.008128743745961396,
        "p95": 0.008128743745961396
      },
      "survival_fraction": {
        "mean": 0.5768330506227105,
        "median": 0.5768330506227105,
        "p90": 0.5768330506227105,
        "p95": 0.5768330506227105
      },
      "top1_agreement": {
        "mean": 0.99609375,
        "median": 0.99609375,
        "p90": 0.99609375,
        "p95": 0.99609375
      },
      "update_cosine": {
        "mean": 0.9686186634106185,
        "median": 0.9686186634106185,
        "p90": 0.9686186634106185,
        "p95": 0.9686186634106185
      },
      "update_distortion_ratio": {
        "mean": 0.06124043507003784,
        "median": 0.06124043507003784,
        "p90": 0.06124043507003784,
        "p95": 0.06124043507003784
      },
      "within_group_dynamic_range_p95": {
        "mean": 32.372964799462466,
        "median": 32.372964799462466,
        "p90": 32.372964799462466,
        "p95": 32.372964799462466
      }
    }
  },
  "C16": {
    "geometry": {
      "bit_width": 8,
      "config": "C16",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 16,
      "orientation": "column",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 1024,
      "scale_count_per_layer": 32768,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.0004923097791447977,
        "median": 0.0004923097791447977,
        "p90": 0.0004923097791447977,
        "p95": 0.0004923097791447977
      },
      "delta_norm": {
        "mean": 2.5968801829237704,
        "median": 2.5968801829237704,
        "p90": 2.5968801829237704,
        "p95": 2.5968801829237704
      },
      "lost_update_fraction": {
        "mean": 0.10009919258881021,
        "median": 0.10009919258881021,
        "p90": 0.10009919258881021,
        "p95": 0.10009919258881021
      },
      "represented_update_norm_ratio": {
        "mean": 0.9074456516302101,
        "median": 0.9074456516302101,
        "p90": 0.9074456516302101,
        "p95": 0.9074456516302101
      },
      "residual_update_cosine": {
        "mean": -0.007500705622547074,
        "median": -0.007500705622547074,
        "p90": -0.007500705622547074,
        "p95": -0.007500705622547074
      },
      "residual_update_projection_coeff": {
        "mean": -0.00047619869751563504,
        "median": -0.00047619869751563504,
        "p90": -0.00047619869751563504,
        "p95": -0.00047619869751563504
      },
      "saturation_fraction": {
        "mean": 0.06306295852138567,
        "median": 0.06306295852138567,
        "p90": 0.06306295852138567,
        "p95": 0.06306295852138567
      },
      "scale_max": {
        "mean": 0.0285640529883955,
        "median": 0.0285640529883955,
        "p90": 0.0285640529883955,
        "p95": 0.0285640529883955
      },
      "scale_mean": {
        "mean": 0.00014890722298213727,
        "median": 0.00014890722298213727,
        "p90": 0.00014890722298213727,
        "p95": 0.00014890722298213727
      },
      "scale_median": {
        "mean": 6.204876940961328e-05,
        "median": 6.204876940961328e-05,
        "p90": 6.204876940961328e-05,
        "p95": 6.204876940961328e-05
      },
      "scale_p95_over_median": {
        "mean": 6.9744380926485,
        "median": 6.9744380926485,
        "p90": 6.9744380926485,
        "p95": 6.9744380926485
      },
      "state_change_error_E_Delta": {
        "mean": 0.3523536962277574,
        "median": 0.3523536962277574,
        "p90": 0.3523536962277574,
        "p95": 0.3523536962277574
      },
      "state_norm": {
        "mean": 19.357204640137173,
        "median": 19.357204640137173,
        "p90": 19.357204640137173,
        "p95": 19.357204640137173
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004755237710596466,
        "median": 0.004755237710596466,
        "p90": 0.004755237710596466,
        "p95": 0.004755237710596466
      },
      "survival_fraction": {
        "mean": 0.6574036173002437,
        "median": 0.6574036173002437,
        "p90": 0.6574036173002437,
        "p95": 0.6574036173002437
      },
      "top1_agreement": {
        "mean": 0.998046875,
        "median": 0.998046875,
        "p90": 0.998046875,
        "p95": 0.998046875
      },
      "update_cosine": {
        "mean": 0.9268837593257698,
        "median": 0.9268837593257698,
        "p90": 0.9268837593257698,
        "p95": 0.9268837593257698
      },
      "update_distortion_ratio": {
        "mean": 0.13863714631869709,
        "median": 0.13863714631869709,
        "p90": 0.13863714631869709,
        "p95": 0.13863714631869709
      },
      "within_group_dynamic_range_p95": {
        "mean": 20.88419809856993,
        "median": 20.88419809856993,
        "p90": 20.88419809856993,
        "p95": 20.88419809856993
      }
    }
  },
  "C32": {
    "geometry": {
      "bit_width": 8,
      "config": "C32",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 32,
      "orientation": "column",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 512,
      "scale_count_per_layer": 16384,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.0005512515830088849,
        "median": 0.0005512515830088849,
        "p90": 0.0005512515830088849,
        "p95": 0.0005512515830088849
      },
      "delta_norm": {
        "mean": 2.597804629934283,
        "median": 2.597804629934283,
        "p90": 2.597804629934283,
        "p95": 2.597804629934283
      },
      "lost_update_fraction": {
        "mean": 0.07326967483959028,
        "median": 0.07326967483959028,
        "p90": 0.07326967483959028,
        "p95": 0.07326967483959028
      },
      "represented_update_norm_ratio": {
        "mean": 0.9299875831775506,
        "median": 0.9299875831775506,
        "p90": 0.9299875831775506,
        "p95": 0.9299875831775506
      },
      "residual_update_cosine": {
        "mean": -0.010603924846218528,
        "median": -0.010603924846218528,
        "p90": -0.010603924846218528,
        "p95": -0.010603924846218528
      },
      "residual_update_projection_coeff": {
        "mean": -0.0008022530529010707,
        "median": -0.0008022530529010707,
        "p90": -0.0008022530529010707,
        "p95": -0.0008022530529010707
      },
      "saturation_fraction": {
        "mean": 0.03157020439451533,
        "median": 0.03157020439451533,
        "p90": 0.03157020439451533,
        "p95": 0.03157020439451533
      },
      "scale_max": {
        "mean": 0.02855611723101655,
        "median": 0.02855611723101655,
        "p90": 0.02855611723101655,
        "p95": 0.02855611723101655
      },
      "scale_mean": {
        "mean": 0.00019160874282425275,
        "median": 0.00019160874282425275,
        "p90": 0.00019160874282425275,
        "p95": 0.00019160874282425275
      },
      "scale_median": {
        "mean": 8.334057582355435e-05,
        "median": 8.334057582355435e-05,
        "p90": 8.334057582355435e-05,
        "p95": 8.334057582355435e-05
      },
      "scale_p95_over_median": {
        "mean": 6.614963164351578,
        "median": 6.614963164351578,
        "p90": 6.614963164351578,
        "p95": 6.614963164351578
      },
      "state_change_error_E_Delta": {
        "mean": 0.3057135116021125,
        "median": 0.3057135116021125,
        "p90": 0.3057135116021125,
        "p95": 0.3057135116021125
      },
      "state_norm": {
        "mean": 19.394502986861596,
        "median": 19.394502986861596,
        "p90": 19.394502986861596,
        "p95": 19.394502986861596
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.00586784772110354,
        "median": 0.00586784772110354,
        "p90": 0.00586784772110354,
        "p95": 0.00586784772110354
      },
      "survival_fraction": {
        "mean": 0.6319445431504621,
        "median": 0.6319445431504621,
        "p90": 0.6319445431504621,
        "p95": 0.6319445431504621
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9442988081561223,
        "median": 0.9442988081561223,
        "p90": 0.9442988081561223,
        "p95": 0.9442988081561223
      },
      "update_distortion_ratio": {
        "mean": 0.10680447066945876,
        "median": 0.10680447066945876,
        "p90": 0.10680447066945876,
        "p95": 0.10680447066945876
      },
      "within_group_dynamic_range_p95": {
        "mean": 23.92671463407492,
        "median": 23.92671463407492,
        "p90": 23.92671463407492,
        "p95": 23.92671463407492
      }
    }
  },
  "C64": {
    "geometry": {
      "bit_width": 8,
      "config": "C64",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 64,
      "orientation": "column",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 256,
      "scale_count_per_layer": 8192,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.0006257958647952178,
        "median": 0.0006257958647952178,
        "p90": 0.0006257958647952178,
        "p95": 0.0006257958647952178
      },
      "delta_norm": {
        "mean": 2.5987530726824555,
        "median": 2.5987530726824555,
        "p90": 2.5987530726824555,
        "p95": 2.5987530726824555
      },
      "lost_update_fraction": {
        "mean": 0.05173438537536032,
        "median": 0.05173438537536032,
        "p90": 0.05173438537536032,
        "p95": 0.05173438537536032
      },
      "represented_update_norm_ratio": {
        "mean": 0.9488672542378545,
        "median": 0.9488672542378545,
        "p90": 0.9488672542378545,
        "p95": 0.9488672542378545
      },
      "residual_update_cosine": {
        "mean": -0.013291465621555696,
        "median": -0.013291465621555696,
        "p90": -0.013291465621555696,
        "p95": -0.013291465621555696
      },
      "residual_update_projection_coeff": {
        "mean": -0.0011967988950784727,
        "median": -0.0011967988950784727,
        "p90": -0.0011967988950784727,
        "p95": -0.0011967988950784727
      },
      "saturation_fraction": {
        "mean": 0.015810462282284058,
        "median": 0.015810462282284058,
        "p90": 0.015810462282284058,
        "p95": 0.015810462282284058
      },
      "scale_max": {
        "mean": 0.028626568331707417,
        "median": 0.028626568331707417,
        "p90": 0.028626568331707417,
        "p95": 0.028626568331707417
      },
      "scale_mean": {
        "mean": 0.00023853680132467618,
        "median": 0.00023853680132467618,
        "p90": 0.00023853680132467618,
        "p95": 0.00023853680132467618
      },
      "scale_median": {
        "mean": 0.00010831606750664671,
        "median": 0.00010831606750664671,
        "p90": 0.00010831606750664671,
        "p95": 0.00010831606750664671
      },
      "scale_p95_over_median": {
        "mean": 6.371639494703156,
        "median": 6.371639494703156,
        "p90": 6.371639494703156,
        "p95": 6.371639494703156
      },
      "state_change_error_E_Delta": {
        "mean": 0.2620523226326617,
        "median": 0.2620523226326617,
        "p90": 0.2620523226326617,
        "p95": 0.2620523226326617
      },
      "state_norm": {
        "mean": 19.42903058871442,
        "median": 19.42903058871442,
        "p90": 19.42903058871442,
        "p95": 19.42903058871442
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006973029626759775,
        "median": 0.006973029626759775,
        "p90": 0.006973029626759775,
        "p95": 0.006973029626759775
      },
      "survival_fraction": {
        "mean": 0.6034883259949315,
        "median": 0.6034883259949315,
        "p90": 0.6034883259949315,
        "p95": 0.6034883259949315
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9584414612572321,
        "median": 0.9584414612572321,
        "p90": 0.9584414612572321,
        "p95": 0.9584414612572321
      },
      "update_distortion_ratio": {
        "mean": 0.08045958181813882,
        "median": 0.08045958181813882,
        "p90": 0.08045958181813882,
        "p95": 0.08045958181813882
      },
      "within_group_dynamic_range_p95": {
        "mean": 27.9301242890579,
        "median": 27.9301242890579,
        "p90": 27.9301242890579,
        "p95": 27.9301242890579
      }
    }
  },
  "R128": {
    "geometry": {
      "bit_width": 8,
      "config": "R128",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 128,
      "orientation": "row",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 128,
      "scale_count_per_layer": 4096,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.04548890240825434,
        "median": 0.04548890240825434,
        "p90": 0.04548890240825434,
        "p95": 0.04548890240825434
      },
      "delta_norm": {
        "mean": 2.6187616060363497,
        "median": 2.6187616060363497,
        "p90": 2.6187616060363497,
        "p95": 2.6187616060363497
      },
      "lost_update_fraction": {
        "mean": 0.12790876524815967,
        "median": 0.12790876524815967,
        "p90": 0.12790876524815967,
        "p95": 0.12790876524815967
      },
      "represented_update_norm_ratio": {
        "mean": 0.9011130339783663,
        "median": 0.9011130339783663,
        "p90": 0.9011130339783663,
        "p95": 0.9011130339783663
      },
      "residual_update_cosine": {
        "mean": -0.021727998987636957,
        "median": -0.021727998987636957,
        "p90": -0.021727998987636957,
        "p95": -0.021727998987636957
      },
      "residual_update_projection_coeff": {
        "mean": -0.0024689195578258384,
        "median": -0.0024689195578258384,
        "p90": -0.0024689195578258384,
        "p95": -0.0024689195578258384
      },
      "saturation_fraction": {
        "mean": 0.007856133088362591,
        "median": 0.007856133088362591,
        "p90": 0.007856133088362591,
        "p95": 0.007856133088362591
      },
      "scale_max": {
        "mean": 0.028526837796760927,
        "median": 0.028526837796760927,
        "p90": 0.028526837796760927,
        "p95": 0.028526837796760927
      },
      "scale_mean": {
        "mean": 0.0010692120905733747,
        "median": 0.0010692120905733747,
        "p90": 0.0010692120905733747,
        "p95": 0.0010692120905733747
      },
      "scale_median": {
        "mean": 0.0005086005557674417,
        "median": 0.0005086005557674417,
        "p90": 0.0005086005557674417,
        "p95": 0.0005086005557674417
      },
      "scale_p95_over_median": {
        "mean": 8.37075308984346,
        "median": 8.37075308984346,
        "p90": 8.37075308984346,
        "p95": 8.37075308984346
      },
      "state_change_error_E_Delta": {
        "mean": 0.385343537327697,
        "median": 0.385343537327697,
        "p90": 0.385343537327697,
        "p95": 0.385343537327697
      },
      "state_norm": {
        "mean": 19.33357479182706,
        "median": 19.33357479182706,
        "p90": 19.33357479182706,
        "p95": 19.33357479182706
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.010535141899127196,
        "median": 0.010535141899127196,
        "p90": 0.010535141899127196,
        "p95": 0.010535141899127196
      },
      "survival_fraction": {
        "mean": 0.46001865516981105,
        "median": 0.46001865516981105,
        "p90": 0.46001865516981105,
        "p95": 0.46001865516981105
      },
      "top1_agreement": {
        "mean": 0.955078125,
        "median": 0.955078125,
        "p90": 0.955078125,
        "p95": 0.955078125
      },
      "update_cosine": {
        "mean": 0.9094661787876448,
        "median": 0.9094661787876448,
        "p90": 0.9094661787876448,
        "p95": 0.9094661787876448
      },
      "update_distortion_ratio": {
        "mean": 0.16819663661361292,
        "median": 0.16819663661361292,
        "p90": 0.16819663661361292,
        "p95": 0.16819663661361292
      },
      "within_group_dynamic_range_p95": {
        "mean": 5298.3161560902445,
        "median": 5298.3161560902445,
        "p90": 5298.3161560902445,
        "p95": 5298.3161560902445
      }
    }
  },
  "R16": {
    "geometry": {
      "bit_width": 8,
      "config": "R16",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 16,
      "orientation": "row",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 1024,
      "scale_count_per_layer": 32768,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.0036277183902955336,
        "median": 0.0036277183902955336,
        "p90": 0.0036277183902955336,
        "p95": 0.0036277183902955336
      },
      "delta_norm": {
        "mean": 2.599830536726128,
        "median": 2.599830536726128,
        "p90": 2.599830536726128,
        "p95": 2.599830536726128
      },
      "lost_update_fraction": {
        "mean": 0.17810101441753484,
        "median": 0.17810101441753484,
        "p90": 0.17810101441753484,
        "p95": 0.17810101441753484
      },
      "represented_update_norm_ratio": {
        "mean": 0.8581968780807796,
        "median": 0.8581968780807796,
        "p90": 0.8581968780807796,
        "p95": 0.8581968780807796
      },
      "residual_update_cosine": {
        "mean": -0.008214666517402807,
        "median": -0.008214666517402807,
        "p90": -0.008214666517402807,
        "p95": -0.008214666517402807
      },
      "residual_update_projection_coeff": {
        "mean": -0.0004496155045879314,
        "median": -0.0004496155045879314,
        "p90": -0.0004496155045879314,
        "p95": -0.0004496155045879314
      },
      "saturation_fraction": {
        "mean": 0.06296184618179115,
        "median": 0.06296184618179115,
        "p90": 0.06296184618179115,
        "p95": 0.06296184618179115
      },
      "scale_max": {
        "mean": 0.028630048311210698,
        "median": 0.028630048311210698,
        "p90": 0.028630048311210698,
        "p95": 0.028630048311210698
      },
      "scale_mean": {
        "mean": 0.00023482671486204076,
        "median": 0.00023482671486204076,
        "p90": 0.00023482671486204076,
        "p95": 0.00023482671486204076
      },
      "scale_median": {
        "mean": 6.407536754856564e-05,
        "median": 6.407536754856564e-05,
        "p90": 6.407536754856564e-05,
        "p95": 6.407536754856564e-05
      },
      "scale_p95_over_median": {
        "mean": 15.899708106774252,
        "median": 15.899708106774252,
        "p90": 15.899708106774252,
        "p95": 15.899708106774252
      },
      "state_change_error_E_Delta": {
        "mean": 0.44808847585188144,
        "median": 0.44808847585188144,
        "p90": 0.44808847585188144,
        "p95": 0.44808847585188144
      },
      "state_norm": {
        "mean": 19.328433725348653,
        "median": 19.328433725348653,
        "p90": 19.328433725348653,
        "p95": 19.328433725348653
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004509695226106723,
        "median": 0.004509695226106723,
        "p90": 0.004509695226106723,
        "p95": 0.004509695226106723
      },
      "survival_fraction": {
        "mean": 0.6412383929935067,
        "median": 0.6412383929935067,
        "p90": 0.6412383929935067,
        "p95": 0.6412383929935067
      },
      "top1_agreement": {
        "mean": 0.986328125,
        "median": 0.986328125,
        "p90": 0.986328125,
        "p95": 0.986328125
      },
      "update_cosine": {
        "mean": 0.8798198125070594,
        "median": 0.8798198125070594,
        "p90": 0.8798198125070594,
        "p95": 0.8798198125070594
      },
      "update_distortion_ratio": {
        "mean": 0.21969030151399557,
        "median": 0.21969030151399557,
        "p90": 0.21969030151399557,
        "p95": 0.21969030151399557
      },
      "within_group_dynamic_range_p95": {
        "mean": 237.91733224736745,
        "median": 237.91733224736745,
        "p90": 237.91733224736745,
        "p95": 237.91733224736745
      }
    }
  },
  "R32": {
    "geometry": {
      "bit_width": 8,
      "config": "R32",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 32,
      "orientation": "row",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 512,
      "scale_count_per_layer": 16384,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.006427407162209736,
        "median": 0.006427407162209736,
        "p90": 0.006427407162209736,
        "p95": 0.006427407162209736
      },
      "delta_norm": {
        "mean": 2.602553206880391,
        "median": 2.602553206880391,
        "p90": 2.602553206880391,
        "p95": 2.602553206880391
      },
      "lost_update_fraction": {
        "mean": 0.16102050425419426,
        "median": 0.16102050425419426,
        "p90": 0.16102050425419426,
        "p95": 0.16102050425419426
      },
      "represented_update_norm_ratio": {
        "mean": 0.8723968312840529,
        "median": 0.8723968312840529,
        "p90": 0.8723968312840529,
        "p95": 0.8723968312840529
      },
      "residual_update_cosine": {
        "mean": -0.011604596749601915,
        "median": -0.011604596749601915,
        "p90": -0.011604596749601915,
        "p95": -0.011604596749601915
      },
      "residual_update_projection_coeff": {
        "mean": -0.0008247942410037072,
        "median": -0.0008247942410037072,
        "p90": -0.0008247942410037072,
        "p95": -0.0008247942410037072
      },
      "saturation_fraction": {
        "mean": 0.031491241965197474,
        "median": 0.031491241965197474,
        "p90": 0.031491241965197474,
        "p95": 0.031491241965197474
      },
      "scale_max": {
        "mean": 0.028606736834083695,
        "median": 0.028606736834083695,
        "p90": 0.028606736834083695,
        "p95": 0.028606736834083695
      },
      "scale_mean": {
        "mean": 0.00038033478944960705,
        "median": 0.00038033478944960705,
        "p90": 0.00038033478944960705,
        "p95": 0.00038033478944960705
      },
      "scale_median": {
        "mean": 9.410641925665329e-05,
        "median": 9.410641925665329e-05,
        "p90": 9.410641925665329e-05,
        "p95": 9.410641925665329e-05
      },
      "scale_p95_over_median": {
        "mean": 18.477281338312725,
        "median": 18.477281338312725,
        "p90": 18.477281338312725,
        "p95": 18.477281338312725
      },
      "state_change_error_E_Delta": {
        "mean": 0.4260235798065262,
        "median": 0.4260235798065262,
        "p90": 0.4260235798065262,
        "p95": 0.4260235798065262
      },
      "state_norm": {
        "mean": 19.327539181969797,
        "median": 19.327539181969797,
        "p90": 19.327539181969797,
        "p95": 19.327539181969797
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006058533607051571,
        "median": 0.006058533607051571,
        "p90": 0.006058533607051571,
        "p95": 0.006058533607051571
      },
      "survival_fraction": {
        "mean": 0.5941296043694603,
        "median": 0.5941296043694603,
        "p90": 0.5941296043694603,
        "p95": 0.5941296043694603
      },
      "top1_agreement": {
        "mean": 0.978515625,
        "median": 0.978515625,
        "p90": 0.978515625,
        "p95": 0.978515625
      },
      "update_cosine": {
        "mean": 0.8903699882999206,
        "median": 0.8903699882999206,
        "p90": 0.8903699882999206,
        "p95": 0.8903699882999206
      },
      "update_distortion_ratio": {
        "mean": 0.2013450292113844,
        "median": 0.2013450292113844,
        "p90": 0.2013450292113844,
        "p95": 0.2013450292113844
      },
      "within_group_dynamic_range_p95": {
        "mean": 921.3794070856694,
        "median": 921.3794070856694,
        "p90": 921.3794070856694,
        "p95": 921.3794070856694
      }
    }
  },
  "R64": {
    "geometry": {
      "bit_width": 8,
      "config": "R64",
      "definition": "CAUSAL / MECHANISM DIAGNOSTIC CONTROL",
      "group_size": 64,
      "orientation": "row",
      "qrange": [
        -127,
        127
      ],
      "rounding": "torch.round",
      "scale_count_per_head": 256,
      "scale_count_per_layer": 8192,
      "zero_point": 0
    },
    "metrics": {
      "KL": {
        "mean": 0.015947314681811008,
        "median": 0.015947314681811008,
        "p90": 0.015947314681811008,
        "p95": 0.015947314681811008
      },
      "delta_norm": {
        "mean": 2.608197441887975,
        "median": 2.608197441887975,
        "p90": 2.608197441887975,
        "p95": 2.608197441887975
      },
      "lost_update_fraction": {
        "mean": 0.14395230944934342,
        "median": 0.14395230944934342,
        "p90": 0.14395230944934342,
        "p95": 0.14395230944934342
      },
      "represented_update_norm_ratio": {
        "mean": 0.8862168133211571,
        "median": 0.8862168133211571,
        "p90": 0.8862168133211571,
        "p95": 0.8862168133211571
      },
      "residual_update_cosine": {
        "mean": -0.016082025036271774,
        "median": -0.016082025036271774,
        "p90": -0.016082025036271774,
        "p95": -0.016082025036271774
      },
      "residual_update_projection_coeff": {
        "mean": -0.001462368468553923,
        "median": -0.001462368468553923,
        "p90": -0.001462368468553923,
        "p95": -0.001462368468553923
      },
      "saturation_fraction": {
        "mean": 0.015738939196940586,
        "median": 0.015738939196940586,
        "p90": 0.015738939196940586,
        "p95": 0.015738939196940586
      },
      "scale_max": {
        "mean": 0.028582462094798144,
        "median": 0.028582462094798144,
        "p90": 0.028582462094798144,
        "p95": 0.028582462094798144
      },
      "scale_mean": {
        "mean": 0.0006289856448427334,
        "median": 0.0006289856448427334,
        "p90": 0.0006289856448427334,
        "p95": 0.0006289856448427334
      },
      "scale_median": {
        "mean": 0.00014729533532095972,
        "median": 0.00014729533532095972,
        "p90": 0.00014729533532095972,
        "p95": 0.00014729533532095972
      },
      "scale_p95_over_median": {
        "mean": 14.569142995367153,
        "median": 14.569142995367153,
        "p90": 14.569142995367153,
        "p95": 14.569142995367153
      },
      "state_change_error_E_Delta": {
        "mean": 0.40504150672122385,
        "median": 0.40504150672122385,
        "p90": 0.40504150672122385,
        "p95": 0.40504150672122385
      },
      "state_norm": {
        "mean": 19.33342275348118,
        "median": 19.33342275348118,
        "p90": 19.33342275348118,
        "p95": 19.33342275348118
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.007970985406872515,
        "median": 0.007970985406872515,
        "p90": 0.007970985406872515,
        "p95": 0.007970985406872515
      },
      "survival_fraction": {
        "mean": 0.5350095978455064,
        "median": 0.5350095978455064,
        "p90": 0.5350095978455064,
        "p95": 0.5350095978455064
      },
      "top1_agreement": {
        "mean": 0.96875,
        "median": 0.96875,
        "p90": 0.96875,
        "p95": 0.96875
      },
      "update_cosine": {
        "mean": 0.900235983910645,
        "median": 0.900235983910645,
        "p90": 0.900235983910645,
        "p95": 0.900235983910645
      },
      "update_distortion_ratio": {
        "mean": 0.1842942622870021,
        "median": 0.1842942622870021,
        "p90": 0.1842942622870021,
        "p95": 0.1842942622870021
      },
      "within_group_dynamic_range_p95": {
        "mean": 2357.517658654774,
        "median": 2357.517658654774,
        "p90": 2357.517658654774,
        "p95": 2357.517658654774
      }
    }
  }
}
```

## Table 3: Row Dose Response
```json
{
  "KL": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.04548890240825434,
          "R16": 0.0036277183902955336,
          "R32": 0.006427407162209736,
          "R64": 0.015947314681811008
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "lost_update_fraction": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.12790876524815967,
          "R16": 0.17810101441753484,
          "R32": 0.16102050425419426,
          "R64": 0.14395230944934342
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_change_error_E_Delta": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.385343537327697,
          "R16": 0.44808847585188144,
          "R32": 0.4260235798065262,
          "R64": 0.40504150672122385
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_reconstruction_relative_error_E_S": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.010535141899127196,
          "R16": 0.004509695226106723,
          "R32": 0.006058533607051571,
          "R64": 0.007970985406872515
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "survival_fraction": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.46001865516981105,
          "R16": 0.6412383929935067,
          "R32": 0.5941296043694603,
          "R64": 0.5350095978455064
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "top1_agreement": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.955078125,
          "R16": 0.986328125,
          "R32": 0.978515625,
          "R64": 0.96875
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "update_cosine": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.9094661787876448,
          "R16": 0.8798198125070594,
          "R32": 0.8903699882999206,
          "R64": 0.900235983910645
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "update_distortion_ratio": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.16819663661361292,
          "R16": 0.21969030151399557,
          "R32": 0.2013450292113844,
          "R64": 0.1842942622870021
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "within_group_dynamic_range_p95": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 5298.3161560902445,
          "R16": 237.91733224736745,
          "R32": 921.3794070856694,
          "R64": 2357.517658654774
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  }
}
```

## Table 4: Column Dose Response
```json
{
  "KL": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "C128": 0.0009054763947956287,
          "C16": 0.0004923097791447977,
          "C32": 0.0005512515830088849,
          "C64": 0.0006257958647952178
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "lost_update_fraction": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "C128": 0.03607998771443544,
          "C16": 0.10009919258881021,
          "C32": 0.07326967483959028,
          "C64": 0.05173438537536032
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_change_error_E_Delta": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "C128": 0.22644575256815908,
          "C16": 0.3523536962277574,
          "C32": 0.3057135116021125,
          "C64": 0.2620523226326617
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_reconstruction_relative_error_E_S": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "C128": 0.008128743745961396,
          "C16": 0.004755237710596466,
          "C32": 0.00586784772110354,
          "C64": 0.006973029626759775
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "survival_fraction": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "C128": 0.5768330506227105,
          "C16": 0.6574036173002437,
          "C32": 0.6319445431504621,
          "C64": 0.6034883259949315
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "top1_agreement": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.31622776601683794,
        "values": {
          "C128": 0.99609375,
          "C16": 0.998046875,
          "C32": 1.0,
          "C64": 1.0
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "update_cosine": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "C128": 0.9686186634106185,
          "C16": 0.9268837593257698,
          "C32": 0.9442988081561223,
          "C64": 0.9584414612572321
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "update_distortion_ratio": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "C128": 0.06124043507003784,
          "C16": 0.13863714631869709,
          "C32": 0.10680447066945876,
          "C64": 0.08045958181813882
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "within_group_dynamic_range_p95": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "C128": 32.372964799462466,
          "C16": 20.88419809856993,
          "C32": 23.92671463407492,
          "C64": 27.9301242890579
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  }
}
```

## Table 5: Matched Group Size R/C
```json
{
  "G128": {
    "KL": {
      "C": 0.0009054763947956287,
      "R": 0.04548890240825434,
      "R_minus_C": 0.04458342601345871,
      "R_over_C": 50.237535367801
    },
    "lost_update_fraction": {
      "C": 0.03607998771443544,
      "R": 0.12790876524815967,
      "R_minus_C": 0.09182877753372423,
      "R_over_C": 3.545144368133583
    },
    "state_change_error_E_Delta": {
      "C": 0.22644575256815908,
      "R": 0.385343537327697,
      "R_minus_C": 0.15889778475953792,
      "R_over_C": 1.7017035336607185
    },
    "survival_fraction": {
      "C": 0.5768330506227105,
      "R": 0.46001865516981105,
      "R_minus_C": -0.11681439545289946,
      "R_over_C": 0.7974901137741771
    },
    "top1_agreement": {
      "C": 0.99609375,
      "R": 0.955078125,
      "R_minus_C": -0.041015625,
      "R_over_C": 0.9588235294117647
    },
    "update_cosine": {
      "C": 0.9686186634106185,
      "R": 0.9094661787876448,
      "R_minus_C": -0.05915248462297373,
      "R_over_C": 0.9389310914011393
    },
    "update_distortion_ratio": {
      "C": 0.06124043507003784,
      "R": 0.16819663661361292,
      "R_minus_C": 0.10695620154357507,
      "R_over_C": 2.7464964352597145
    },
    "within_group_dynamic_range_p95": {
      "C": 32.372964799462466,
      "R": 5298.3161560902445,
      "R_minus_C": 5265.943191290782,
      "R_over_C": 163.66484160197214
    }
  },
  "G16": {
    "KL": {
      "C": 0.0004923097791447977,
      "R": 0.0036277183902955336,
      "R_minus_C": 0.003135408611150736,
      "R_over_C": 7.368771744890633
    },
    "lost_update_fraction": {
      "C": 0.10009919258881021,
      "R": 0.17810101441753484,
      "R_minus_C": 0.07800182182872463,
      "R_over_C": 1.7792452647359738
    },
    "state_change_error_E_Delta": {
      "C": 0.3523536962277574,
      "R": 0.44808847585188144,
      "R_minus_C": 0.09573477962412402,
      "R_over_C": 1.2717007956750428
    },
    "survival_fraction": {
      "C": 0.6574036173002437,
      "R": 0.6412383929935067,
      "R_minus_C": -0.016165224306736947,
      "R_over_C": 0.9754105029523223
    },
    "top1_agreement": {
      "C": 0.998046875,
      "R": 0.986328125,
      "R_minus_C": -0.01171875,
      "R_over_C": 0.9882583170254403
    },
    "update_cosine": {
      "C": 0.9268837593257698,
      "R": 0.8798198125070594,
      "R_minus_C": -0.0470639468187104,
      "R_over_C": 0.9492234637351447
    },
    "update_distortion_ratio": {
      "C": 0.13863714631869709,
      "R": 0.21969030151399557,
      "R_minus_C": 0.08105315519529849,
      "R_over_C": 1.5846424089614097
    },
    "within_group_dynamic_range_p95": {
      "C": 20.88419809856993,
      "R": 237.91733224736745,
      "R_minus_C": 217.03313414879753,
      "R_over_C": 11.392217748770499
    }
  },
  "G32": {
    "KL": {
      "C": 0.0005512515830088849,
      "R": 0.006427407162209736,
      "R_minus_C": 0.005876155579200851,
      "R_over_C": 11.659662049634678
    },
    "lost_update_fraction": {
      "C": 0.07326967483959028,
      "R": 0.16102050425419426,
      "R_minus_C": 0.08775082941460398,
      "R_over_C": 2.1976418566987963
    },
    "state_change_error_E_Delta": {
      "C": 0.3057135116021125,
      "R": 0.4260235798065262,
      "R_minus_C": 0.1203100682044137,
      "R_over_C": 1.3935386027719894
    },
    "survival_fraction": {
      "C": 0.6319445431504621,
      "R": 0.5941296043694603,
      "R_minus_C": -0.03781493878100184,
      "R_over_C": 0.9401609853414015
    },
    "top1_agreement": {
      "C": 1.0,
      "R": 0.978515625,
      "R_minus_C": -0.021484375,
      "R_over_C": 0.978515625
    },
    "update_cosine": {
      "C": 0.9442988081561223,
      "R": 0.8903699882999206,
      "R_minus_C": -0.05392881985620168,
      "R_over_C": 0.942890090096052
    },
    "update_distortion_ratio": {
      "C": 0.10680447066945876,
      "R": 0.2013450292113844,
      "R_minus_C": 0.09454055854192563,
      "R_over_C": 1.8851741687341177
    },
    "within_group_dynamic_range_p95": {
      "C": 23.92671463407492,
      "R": 921.3794070856694,
      "R_minus_C": 897.4526924515945,
      "R_over_C": 38.50839620804015
    }
  },
  "G64": {
    "KL": {
      "C": 0.0006257958647952178,
      "R": 0.015947314681811008,
      "R_minus_C": 0.01532151881701579,
      "R_over_C": 25.483253531931734
    },
    "lost_update_fraction": {
      "C": 0.05173438537536032,
      "R": 0.14395230944934342,
      "R_minus_C": 0.0922179240739831,
      "R_over_C": 2.782526716126872
    },
    "state_change_error_E_Delta": {
      "C": 0.2620523226326617,
      "R": 0.40504150672122385,
      "R_minus_C": 0.14298918408856215,
      "R_over_C": 1.5456512754859295
    },
    "survival_fraction": {
      "C": 0.6034883259949315,
      "R": 0.5350095978455064,
      "R_minus_C": -0.06847872814942513,
      "R_over_C": 0.886528495747571
    },
    "top1_agreement": {
      "C": 1.0,
      "R": 0.96875,
      "R_minus_C": -0.03125,
      "R_over_C": 0.96875
    },
    "update_cosine": {
      "C": 0.9584414612572321,
      "R": 0.900235983910645,
      "R_minus_C": -0.0582054773465871,
      "R_over_C": 0.9392707017596711
    },
    "update_distortion_ratio": {
      "C": 0.08045958181813882,
      "R": 0.1842942622870021,
      "R_minus_C": 0.10383468046886328,
      "R_over_C": 2.290519763122293
    },
    "within_group_dynamic_range_p95": {
      "C": 27.9301242890579,
      "R": 2357.517658654774,
      "R_minus_C": 2329.5875343657162,
      "R_over_C": 84.40770382029312
    }
  }
}
```

## Table 6: 6-Prompt Consistency
```json
{
  "KL": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.04548890240825434,
          "R16": 0.0036277183902955336,
          "R32": 0.006427407162209736,
          "R64": 0.015947314681811008
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "lost_update_fraction": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.12790876524815967,
          "R16": 0.17810101441753484,
          "R32": 0.16102050425419426,
          "R64": 0.14395230944934342
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_change_error_E_Delta": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.385343537327697,
          "R16": 0.44808847585188144,
          "R32": 0.4260235798065262,
          "R64": 0.40504150672122385
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "state_reconstruction_relative_error_E_S": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.010535141899127196,
          "R16": 0.004509695226106723,
          "R32": 0.006058533607051571,
          "R64": 0.007970985406872515
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "survival_fraction": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.46001865516981105,
          "R16": 0.6412383929935067,
          "R32": 0.5941296043694603,
          "R64": 0.5350095978455064
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "top1_agreement": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.955078125,
          "R16": 0.986328125,
          "R32": 0.978515625,
          "R64": 0.96875
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "update_cosine": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.9094661787876448,
          "R16": 0.8798198125070594,
          "R32": 0.8903699882999206,
          "R64": 0.900235983910645
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "update_distortion_ratio": {
    "improved_prompt_count": 0,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": false,
        "spearman_group_size_metric": -0.9999999999999998,
        "values": {
          "R128": 0.16819663661361292,
          "R16": 0.21969030151399557,
          "R32": 0.2013450292113844,
          "R64": 0.1842942622870021
        }
      }
    },
    "rate": 0.0,
    "total_prompt_count": 1
  },
  "within_group_dynamic_range_p95": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 5298.3161560902445,
          "R16": 237.91733224736745,
          "R32": 921.3794070856694,
          "R64": 2357.517658654774
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  }
}
```

## Table 7: Temporal Windows
```json
{
  "C128": {
    "1-128": {
      "KL": {
        "mean": 0.00041035968331197775,
        "median": 0.00041035968331197775,
        "p90": 0.00041035968331197775,
        "p95": 0.00041035968331197775
      },
      "delta_norm": {
        "mean": 2.9114629101173852,
        "median": 2.9114629101173852,
        "p90": 2.9114629101173852,
        "p95": 2.9114629101173852
      },
      "lost_update_fraction": {
        "mean": 0.03254107700908137,
        "median": 0.03254107700908137,
        "p90": 0.03254107700908137,
        "p95": 0.03254107700908137
      },
      "represented_update_norm_ratio": {
        "mean": 0.9644203256311418,
        "median": 0.9644203256311418,
        "p90": 0.9644203256311418,
        "p95": 0.9644203256311418
      },
      "residual_update_cosine": {
        "mean": -0.013810826557477732,
        "median": -0.013810826557477732,
        "p90": -0.013810826557477732,
        "p95": -0.013810826557477732
      },
      "residual_update_projection_coeff": {
        "mean": -0.0012346840792843024,
        "median": -0.0012346840792843024,
        "p90": -0.0012346840792843024,
        "p95": -0.0012346840792843024
      },
      "saturation_fraction": {
        "mean": 0.007918956711536314,
        "median": 0.007918956711536314,
        "p90": 0.007918956711536314,
        "p95": 0.007918956711536314
      },
      "scale_max": {
        "mean": 0.02658711456709903,
        "median": 0.02658711456709903,
        "p90": 0.02658711456709903,
        "p95": 0.02658711456709903
      },
      "scale_mean": {
        "mean": 0.00027243159021099803,
        "median": 0.00027243159021099803,
        "p90": 0.00027243159021099803,
        "p95": 0.00027243159021099803
      },
      "scale_median": {
        "mean": 0.00012585585450933004,
        "median": 0.00012585585450933004,
        "p90": 0.00012585585450933004,
        "p95": 0.00012585585450933004
      },
      "scale_p95_over_median": {
        "mean": 6.218201340796961,
        "median": 6.218201340796961,
        "p90": 6.218201340796961,
        "p95": 6.218201340796961
      },
      "state_change_error_E_Delta": {
        "mean": 0.21983151209116453,
        "median": 0.21983151209116453,
        "p90": 0.21983151209116453,
        "p95": 0.21983151209116453
      },
      "state_norm": {
        "mean": 18.065781771198033,
        "median": 18.065781771198033,
        "p90": 18.065781771198033,
        "p95": 18.065781771198033
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.009087469085957775,
        "median": 0.009087469085957775,
        "p90": 0.009087469085957775,
        "p95": 0.009087469085957775
      },
      "survival_fraction": {
        "mean": 0.6403506165101457,
        "median": 0.6403506165101457,
        "p90": 0.6403506165101457,
        "p95": 0.6403506165101457
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9705868956689133,
        "median": 0.9705868956689133,
        "p90": 0.9705868956689133,
        "p95": 0.9705868956689133
      },
      "update_distortion_ratio": {
        "mean": 0.057448757305449454,
        "median": 0.057448757305449454,
        "p90": 0.057448757305449454,
        "p95": 0.057448757305449454
      },
      "within_group_dynamic_range_p95": {
        "mean": 31.39799330819623,
        "median": 31.39799330819623,
        "p90": 31.39799330819623,
        "p95": 31.39799330819623
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.0012654675298511054,
        "median": 0.0012654675298511054,
        "p90": 0.0012654675298511054,
        "p95": 0.0012654675298511054
      },
      "delta_norm": {
        "mean": 2.802069958636327,
        "median": 2.802069958636327,
        "p90": 2.802069958636327,
        "p95": 2.802069958636327
      },
      "lost_update_fraction": {
        "mean": 0.03592951592330893,
        "median": 0.03592951592330893,
        "p90": 0.03592951592330893,
        "p95": 0.03592951592330893
      },
      "represented_update_norm_ratio": {
        "mean": 0.962801632125448,
        "median": 0.962801632125448,
        "p90": 0.962801632125448,
        "p95": 0.962801632125448
      },
      "residual_update_cosine": {
        "mean": -0.015992191947267052,
        "median": -0.015992191947267052,
        "p90": -0.015992191947267052,
        "p95": -0.015992191947267052
      },
      "residual_update_projection_coeff": {
        "mean": -0.0015026408400615652,
        "median": -0.0015026408400615652,
        "p90": -0.0015026408400615652,
        "p95": -0.0015026408400615652
      },
      "saturation_fraction": {
        "mean": 0.007919732481241226,
        "median": 0.007919732481241226,
        "p90": 0.007919732481241226,
        "p95": 0.007919732481241226
      },
      "scale_max": {
        "mean": 0.02847038176150818,
        "median": 0.02847038176150818,
        "p90": 0.02847038176150818,
        "p95": 0.02847038176150818
      },
      "scale_mean": {
        "mean": 0.000291271784214373,
        "median": 0.000291271784214373,
        "p90": 0.000291271784214373,
        "p95": 0.000291271784214373
      },
      "scale_median": {
        "mean": 0.00013274741922912148,
        "median": 0.00013274741922912148,
        "p90": 0.00013274741922912148,
        "p95": 0.00013274741922912148
      },
      "scale_p95_over_median": {
        "mean": 6.3507625410168,
        "median": 6.3507625410168,
        "p90": 6.3507625410168,
        "p95": 6.3507625410168
      },
      "state_change_error_E_Delta": {
        "mean": 0.2254077779482374,
        "median": 0.2254077779482374,
        "p90": 0.2254077779482374,
        "p95": 0.2254077779482374
      },
      "state_norm": {
        "mean": 19.609626544251416,
        "median": 19.609626544251416,
        "p90": 19.609626544251416,
        "p95": 19.609626544251416
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.008442768383687583,
        "median": 0.008442768383687583,
        "p90": 0.008442768383687583,
        "p95": 0.008442768383687583
      },
      "survival_fraction": {
        "mean": 0.5993677228689193,
        "median": 0.5993677228689193,
        "p90": 0.5993677228689193,
        "p95": 0.5993677228689193
      },
      "top1_agreement": {
        "mean": 0.9921875,
        "median": 0.9921875,
        "p90": 0.9921875,
        "p95": 0.9921875
      },
      "update_cosine": {
        "mean": 0.9687603861869623,
        "median": 0.9687603861869623,
        "p90": 0.9687603861869623,
        "p95": 0.9687603861869623
      },
      "update_distortion_ratio": {
        "mean": 0.060980853930270966,
        "median": 0.060980853930270966,
        "p90": 0.060980853930270966,
        "p95": 0.060980853930270966
      },
      "within_group_dynamic_range_p95": {
        "mean": 32.77498228824698,
        "median": 32.77498228824698,
        "p90": 32.77498228824698,
        "p95": 32.77498228824698
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.0009730391830097159,
        "median": 0.0009730391830097159,
        "p90": 0.0009730391830097159,
        "p95": 0.0009730391830097159
      },
      "delta_norm": {
        "mean": 2.3423013188769133,
        "median": 2.3423013188769133,
        "p90": 2.3423013188769133,
        "p95": 2.3423013188769133
      },
      "lost_update_fraction": {
        "mean": 0.037910855092732884,
        "median": 0.037910855092732884,
        "p90": 0.037910855092732884,
        "p95": 0.037910855092732884
      },
      "represented_update_norm_ratio": {
        "mean": 0.9630868072418162,
        "median": 0.9630868072418162,
        "p90": 0.9630868072418162,
        "p95": 0.9630868072418162
      },
      "residual_update_cosine": {
        "mean": -0.020319893128250523,
        "median": -0.020319893128250523,
        "p90": -0.020319893128250523,
        "p95": -0.020319893128250523
      },
      "residual_update_projection_coeff": {
        "mean": -0.002195920254653698,
        "median": -0.002195920254653698,
        "p90": -0.002195920254653698,
        "p95": -0.002195920254653698
      },
      "saturation_fraction": {
        "mean": 0.00792038347572088,
        "median": 0.00792038347572088,
        "p90": 0.00792038347572088,
        "p95": 0.00792038347572088
      },
      "scale_max": {
        "mean": 0.029419916438958662,
        "median": 0.029419916438958662,
        "p90": 0.029419916438958662,
        "p95": 0.029419916438958662
      },
      "scale_mean": {
        "mean": 0.00030016270559409794,
        "median": 0.00030016270559409794,
        "p90": 0.00030016270559409794,
        "p95": 0.00030016270559409794
      },
      "scale_median": {
        "mean": 0.00013562346410160114,
        "median": 0.00013562346410160114,
        "p90": 0.00013562346410160114,
        "p95": 0.00013562346410160114
      },
      "scale_p95_over_median": {
        "mean": 6.353558547474065,
        "median": 6.353558547474065,
        "p90": 6.353558547474065,
        "p95": 6.353558547474065
      },
      "state_change_error_E_Delta": {
        "mean": 0.2302460232397541,
        "median": 0.2302460232397541,
        "p90": 0.2302460232397541,
        "p95": 0.2302460232397541
      },
      "state_norm": {
        "mean": 20.053902845984936,
        "median": 20.053902845984936,
        "p90": 20.053902845984936,
        "p95": 20.053902845984936
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.007496113777959484,
        "median": 0.007496113777959484,
        "p90": 0.007496113777959484,
        "p95": 0.007496113777959484
      },
      "survival_fraction": {
        "mean": 0.5340550470476345,
        "median": 0.5340550470476345,
        "p90": 0.5340550470476345,
        "p95": 0.5340550470476345
      },
      "top1_agreement": {
        "mean": 0.99609375,
        "median": 0.99609375,
        "p90": 0.99609375,
        "p95": 0.99609375
      },
      "update_cosine": {
        "mean": 0.9675713743005565,
        "median": 0.9675713743005565,
        "p90": 0.9675713743005565,
        "p95": 0.9675713743005565
      },
      "update_distortion_ratio": {
        "mean": 0.06325125328094755,
        "median": 0.06325125328094755,
        "p90": 0.06325125328094755,
        "p95": 0.06325125328094755
      },
      "within_group_dynamic_range_p95": {
        "mean": 32.65563331831556,
        "median": 32.65563331831556,
        "p90": 32.65563331831556,
        "p95": 32.65563331831556
      }
    }
  },
  "C16": {
    "1-128": {
      "KL": {
        "mean": 0.0003500083066052096,
        "median": 0.0003500083066052096,
        "p90": 0.0003500083066052096,
        "p95": 0.0003500083066052096
      },
      "delta_norm": {
        "mean": 2.9115309847782815,
        "median": 2.9115309847782815,
        "p90": 2.9115309847782815,
        "p95": 2.9115309847782815
      },
      "lost_update_fraction": {
        "mean": 0.09260423376138242,
        "median": 0.09260423376138242,
        "p90": 0.09260423376138242,
        "p95": 0.09260423376138242
      },
      "represented_update_norm_ratio": {
        "mean": 0.9113532807390088,
        "median": 0.9113532807390088,
        "p90": 0.9113532807390088,
        "p95": 0.9113532807390088
      },
      "residual_update_cosine": {
        "mean": -0.005288439385555967,
        "median": -0.005288439385555967,
        "p90": -0.005288439385555967,
        "p95": -0.005288439385555967
      },
      "residual_update_projection_coeff": {
        "mean": -0.0002836116630478147,
        "median": -0.0002836116630478147,
        "p90": -0.0002836116630478147,
        "p95": -0.0002836116630478147
      },
      "saturation_fraction": {
        "mean": 0.06306675725721624,
        "median": 0.06306675725721624,
        "p90": 0.06306675725721624,
        "p95": 0.06306675725721624
      },
      "scale_max": {
        "mean": 0.026602014981447337,
        "median": 0.026602014981447337,
        "p90": 0.026602014981447337,
        "p95": 0.026602014981447337
      },
      "scale_mean": {
        "mean": 0.00013985115833195215,
        "median": 0.00013985115833195215,
        "p90": 0.00013985115833195215,
        "p95": 0.00013985115833195215
      },
      "scale_median": {
        "mean": 5.959881877739223e-05,
        "median": 5.959881877739223e-05,
        "p90": 5.959881877739223e-05,
        "p95": 5.959881877739223e-05
      },
      "scale_p95_over_median": {
        "mean": 6.861267878615402,
        "median": 6.861267878615402,
        "p90": 6.861267878615402,
        "p95": 6.861267878615402
      },
      "state_change_error_E_Delta": {
        "mean": 0.3423664350345129,
        "median": 0.3423664350345129,
        "p90": 0.3423664350345129,
        "p95": 0.3423664350345129
      },
      "state_norm": {
        "mean": 18.03836570113036,
        "median": 18.03836570113036,
        "p90": 18.03836570113036,
        "p95": 18.03836570113036
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.0051961636028009205,
        "median": 0.0051961636028009205,
        "p90": 0.0051961636028009205,
        "p95": 0.0051961636028009205
      },
      "survival_fraction": {
        "mean": 0.7182336698366899,
        "median": 0.7182336698366899,
        "p90": 0.7182336698366899,
        "p95": 0.7182336698366899
      },
      "top1_agreement": {
        "mean": 0.9921875,
        "median": 0.9921875,
        "p90": 0.9921875,
        "p95": 0.9921875
      },
      "update_cosine": {
        "mean": 0.9312729372244491,
        "median": 0.9312729372244491,
        "p90": 0.9312729372244491,
        "p95": 0.9312729372244491
      },
      "update_distortion_ratio": {
        "mean": 0.1307643509393404,
        "median": 0.1307643509393404,
        "p90": 0.1307643509393404,
        "p95": 0.1307643509393404
      },
      "within_group_dynamic_range_p95": {
        "mean": 20.262750696572738,
        "median": 20.262750696572738,
        "p90": 20.262750696572738,
        "p95": 20.262750696572738
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.0005066953230220613,
        "median": 0.0005066953230220613,
        "p90": 0.0005066953230220613,
        "p95": 0.0005066953230220613
      },
      "delta_norm": {
        "mean": 2.7977152919520885,
        "median": 2.7977152919520885,
        "p90": 2.7977152919520885,
        "p95": 2.7977152919520885
      },
      "lost_update_fraction": {
        "mean": 0.099934037637933,
        "median": 0.099934037637933,
        "p90": 0.099934037637933,
        "p95": 0.099934037637933
      },
      "represented_update_norm_ratio": {
        "mean": 0.9066985574364641,
        "median": 0.9066985574364641,
        "p90": 0.9066985574364641,
        "p95": 0.9066985574364641
      },
      "residual_update_cosine": {
        "mean": -0.00664234401064334,
        "median": -0.00664234401064334,
        "p90": -0.00664234401064334,
        "p95": -0.00664234401064334
      },
      "residual_update_projection_coeff": {
        "mean": -0.00037086179074638075,
        "median": -0.00037086179074638075,
        "p90": -0.00037086179074638075,
        "p95": -0.00037086179074638075
      },
      "saturation_fraction": {
        "mean": 0.06307402501503628,
        "median": 0.06307402501503628,
        "p90": 0.06307402501503628,
        "p95": 0.06307402501503628
      },
      "scale_max": {
        "mean": 0.028530084240249682,
        "median": 0.028530084240249682,
        "p90": 0.028530084240249682,
        "p95": 0.028530084240249682
      },
      "scale_mean": {
        "mean": 0.00014884296767093621,
        "median": 0.00014884296767093621,
        "p90": 0.00014884296767093621,
        "p95": 0.00014884296767093621
      },
      "scale_median": {
        "mean": 6.165041742630292e-05,
        "median": 6.165041742630292e-05,
        "p90": 6.165041742630292e-05,
        "p95": 6.165041742630292e-05
      },
      "scale_p95_over_median": {
        "mean": 7.05382933404638,
        "median": 7.05382933404638,
        "p90": 7.05382933404638,
        "p95": 7.05382933404638
      },
      "state_change_error_E_Delta": {
        "mean": 0.35193088428062047,
        "median": 0.35193088428062047,
        "p90": 0.35193088428062047,
        "p95": 0.35193088428062047
      },
      "state_norm": {
        "mean": 19.540950587484986,
        "median": 19.540950587484986,
        "p90": 19.540950587484986,
        "p95": 19.540950587484986
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004908772384751065,
        "median": 0.004908772384751065,
        "p90": 0.004908772384751065,
        "p95": 0.004908772384751065
      },
      "survival_fraction": {
        "mean": 0.6796245531489451,
        "median": 0.6796245531489451,
        "p90": 0.6796245531489451,
        "p95": 0.6796245531489451
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9267737908327637,
        "median": 0.9267737908327637,
        "p90": 0.9267737908327637,
        "p95": 0.9267737908327637
      },
      "update_distortion_ratio": {
        "mean": 0.13877930526145116,
        "median": 0.13877930526145116,
        "p90": 0.13877930526145116,
        "p95": 0.13877930526145116
      },
      "within_group_dynamic_range_p95": {
        "mean": 21.145953890122456,
        "median": 21.145953890122456,
        "p90": 21.145953890122456,
        "p95": 21.145953890122456
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.0005562677434759602,
        "median": 0.0005562677434759602,
        "p90": 0.0005562677434759602,
        "p95": 0.0005562677434759602
      },
      "delta_norm": {
        "mean": 2.3403663321771013,
        "median": 2.3403663321771013,
        "p90": 2.3403663321771013,
        "p95": 2.3403663321771013
      },
      "lost_update_fraction": {
        "mean": 0.1038999722950432,
        "median": 0.1038999722950432,
        "p90": 0.1038999722950432,
        "p95": 0.1038999722950432
      },
      "represented_update_norm_ratio": {
        "mean": 0.9058806483488906,
        "median": 0.9058806483488906,
        "p90": 0.9058806483488906,
        "p95": 0.9058806483488906
      },
      "residual_update_cosine": {
        "mean": -0.009027377882006249,
        "median": -0.009027377882006249,
        "p90": -0.009027377882006249,
        "p95": -0.009027377882006249
      },
      "residual_update_projection_coeff": {
        "mean": -0.0006244083750307822,
        "median": -0.0006244083750307822,
        "p90": -0.0006244083750307822,
        "p95": -0.0006244083750307822
      },
      "saturation_fraction": {
        "mean": 0.06305554074545701,
        "median": 0.06305554074545701,
        "p90": 0.06305554074545701,
        "p95": 0.06305554074545701
      },
      "scale_max": {
        "mean": 0.029554392154977908,
        "median": 0.029554392154977908,
        "p90": 0.029554392154977908,
        "p95": 0.029554392154977908
      },
      "scale_mean": {
        "mean": 0.00015343200771029063,
        "median": 0.00015343200771029063,
        "p90": 0.00015343200771029063,
        "p95": 0.00015343200771029063
      },
      "scale_median": {
        "mean": 6.346335059772186e-05,
        "median": 6.346335059772186e-05,
        "p90": 6.346335059772186e-05,
        "p95": 6.346335059772186e-05
      },
      "scale_p95_over_median": {
        "mean": 6.990885507817528,
        "median": 6.990885507817528,
        "p90": 6.990885507817528,
        "p95": 6.990885507817528
      },
      "state_change_error_E_Delta": {
        "mean": 0.357519720058913,
        "median": 0.357519720058913,
        "p90": 0.357519720058913,
        "p95": 0.357519720058913
      },
      "state_norm": {
        "mean": 19.91959942136114,
        "median": 19.91959942136114,
        "p90": 19.91959942136114,
        "p95": 19.91959942136114
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004459729794183367,
        "median": 0.004459729794183367,
        "p90": 0.004459729794183367,
        "p95": 0.004459729794183367
      },
      "survival_fraction": {
        "mean": 0.6161157405003904,
        "median": 0.6161157405003904,
        "p90": 0.6161157405003904,
        "p95": 0.6161157405003904
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9247612998490998,
        "median": 0.9247612998490998,
        "p90": 0.9247612998490998,
        "p95": 0.9247612998490998
      },
      "update_distortion_ratio": {
        "mean": 0.14247171143004775,
        "median": 0.14247171143004775,
        "p90": 0.14247171143004775,
        "p95": 0.14247171143004775
      },
      "within_group_dynamic_range_p95": {
        "mean": 21.06161637487821,
        "median": 21.06161637487821,
        "p90": 21.06161637487821,
        "p95": 21.06161637487821
      }
    }
  },
  "C32": {
    "1-128": {
      "KL": {
        "mean": 0.00035422713724353764,
        "median": 0.00035422713724353764,
        "p90": 0.00035422713724353764,
        "p95": 0.00035422713724353764
      },
      "delta_norm": {
        "mean": 2.911687783500938,
        "median": 2.911687783500938,
        "p90": 2.911687783500938,
        "p95": 2.911687783500938
      },
      "lost_update_fraction": {
        "mean": 0.06740968304665021,
        "median": 0.06740968304665021,
        "p90": 0.06740968304665021,
        "p95": 0.06740968304665021
      },
      "represented_update_norm_ratio": {
        "mean": 0.9327574375329789,
        "median": 0.9327574375329789,
        "p90": 0.9327574375329789,
        "p95": 0.9327574375329789
      },
      "residual_update_cosine": {
        "mean": -0.007928244273729112,
        "median": -0.007928244273729112,
        "p90": -0.007928244273729112,
        "p95": -0.007928244273729112
      },
      "residual_update_projection_coeff": {
        "mean": -0.0005028304289545478,
        "median": -0.0005028304289545478,
        "p90": -0.0005028304289545478,
        "p95": -0.0005028304289545478
      },
      "saturation_fraction": {
        "mean": 0.031574803074513845,
        "median": 0.031574803074513845,
        "p90": 0.031574803074513845,
        "p95": 0.031574803074513845
      },
      "scale_max": {
        "mean": 0.026594309365511878,
        "median": 0.026594309365511878,
        "p90": 0.026594309365511878,
        "p95": 0.026594309365511878
      },
      "scale_mean": {
        "mean": 0.0001800559091182187,
        "median": 0.0001800559091182187,
        "p90": 0.0001800559091182187,
        "p95": 0.0001800559091182187
      },
      "scale_median": {
        "mean": 7.970760139604557e-05,
        "median": 7.970760139604557e-05,
        "p90": 7.970760139604557e-05,
        "p95": 7.970760139604557e-05
      },
      "scale_p95_over_median": {
        "mean": 6.525382838772164,
        "median": 6.525382838772164,
        "p90": 6.525382838772164,
        "p95": 6.525382838772164
      },
      "state_change_error_E_Delta": {
        "mean": 0.29686670097852913,
        "median": 0.29686670097852913,
        "p90": 0.29686670097852913,
        "p95": 0.29686670097852913
      },
      "state_norm": {
        "mean": 18.046067065610654,
        "median": 18.046067065610654,
        "p90": 18.046067065610654,
        "p95": 18.046067065610654
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006463570604132626,
        "median": 0.006463570604132626,
        "p90": 0.006463570604132626,
        "p95": 0.006463570604132626
      },
      "survival_fraction": {
        "mean": 0.6948418585960003,
        "median": 0.6948418585960003,
        "p90": 0.6948418585960003,
        "p95": 0.6948418585960003
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9476277413252457,
        "median": 0.9476277413252457,
        "p90": 0.9476277413252457,
        "p95": 0.9476277413252457
      },
      "update_distortion_ratio": {
        "mean": 0.10065318846077344,
        "median": 0.10065318846077344,
        "p90": 0.10065318846077344,
        "p95": 0.10065318846077344
      },
      "within_group_dynamic_range_p95": {
        "mean": 23.22442630923638,
        "median": 23.22442630923638,
        "p90": 23.22442630923638,
        "p95": 23.22442630923638
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.0007283072205908725,
        "median": 0.0007283072205908725,
        "p90": 0.0007283072205908725,
        "p95": 0.0007283072205908725
      },
      "delta_norm": {
        "mean": 2.799646097304739,
        "median": 2.799646097304739,
        "p90": 2.799646097304739,
        "p95": 2.799646097304739
      },
      "lost_update_fraction": {
        "mean": 0.07287037138765953,
        "median": 0.07287037138765953,
        "p90": 0.07287037138765953,
        "p95": 0.07287037138765953
      },
      "represented_update_norm_ratio": {
        "mean": 0.929264146102682,
        "median": 0.929264146102682,
        "p90": 0.929264146102682,
        "p95": 0.929264146102682
      },
      "residual_update_cosine": {
        "mean": -0.00927228319141212,
        "median": -0.00927228319141212,
        "p90": -0.00927228319141212,
        "p95": -0.00927228319141212
      },
      "residual_update_projection_coeff": {
        "mean": -0.0006273008221476092,
        "median": -0.0006273008221476092,
        "p90": -0.0006273008221476092,
        "p95": -0.0006273008221476092
      },
      "saturation_fraction": {
        "mean": 0.0315734725445509,
        "median": 0.0315734725445509,
        "p90": 0.0315734725445509,
        "p95": 0.0315734725445509
      },
      "scale_max": {
        "mean": 0.02853022999299053,
        "median": 0.02853022999299053,
        "p90": 0.02853022999299053,
        "p95": 0.02853022999299053
      },
      "scale_mean": {
        "mean": 0.00019169126343204012,
        "median": 0.00019169126343204012,
        "p90": 0.00019169126343204012,
        "p95": 0.00019169126343204012
      },
      "scale_median": {
        "mean": 8.311180648294207e-05,
        "median": 8.311180648294207e-05,
        "p90": 8.311180648294207e-05,
        "p95": 8.311180648294207e-05
      },
      "scale_p95_over_median": {
        "mean": 6.672580242116893,
        "median": 6.672580242116893,
        "p90": 6.672580242116893,
        "p95": 6.672580242116893
      },
      "state_change_error_E_Delta": {
        "mean": 0.3049515012821404,
        "median": 0.3049515012821404,
        "p90": 0.3049515012821404,
        "p95": 0.3049515012821404
      },
      "state_norm": {
        "mean": 19.566608674669023,
        "median": 19.566608674669023,
        "p90": 19.566608674669023,
        "p95": 19.566608674669023
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006075173216168705,
        "median": 0.006075173216168705,
        "p90": 0.006075173216168705,
        "p95": 0.006075173216168705
      },
      "survival_fraction": {
        "mean": 0.6545501165091991,
        "median": 0.6545501165091991,
        "p90": 0.6545501165091991,
        "p95": 0.6545501165091991
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9443071402880984,
        "median": 0.9443071402880984,
        "p90": 0.9443071402880984,
        "p95": 0.9443071402880984
      },
      "update_distortion_ratio": {
        "mean": 0.10673584560515208,
        "median": 0.10673584560515208,
        "p90": 0.10673584560515208,
        "p95": 0.10673584560515208
      },
      "within_group_dynamic_range_p95": {
        "mean": 24.26590097397564,
        "median": 24.26590097397564,
        "p90": 24.26590097397564,
        "p95": 24.26590097397564
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.0005612359871005647,
        "median": 0.0005612359871005647,
        "p90": 0.0005612359871005647,
        "p95": 0.0005612359871005647
      },
      "delta_norm": {
        "mean": 2.341168425534348,
        "median": 2.341168425534348,
        "p90": 2.341168425534348,
        "p95": 2.341168425534348
      },
      "lost_update_fraction": {
        "mean": 0.07637643186908463,
        "median": 0.07637643186908463,
        "p90": 0.07637643186908463,
        "p95": 0.07637643186908463
      },
      "represented_update_norm_ratio": {
        "mean": 0.9289751942808464,
        "median": 0.9289751942808464,
        "p90": 0.9289751942808464,
        "p95": 0.9289751942808464
      },
      "residual_update_cosine": {
        "mean": -0.012597134082630157,
        "median": -0.012597134082630157,
        "p90": -0.012597134082630157,
        "p95": -0.012597134082630157
      },
      "residual_update_projection_coeff": {
        "mean": -0.0010382708606262714,
        "median": -0.0010382708606262714,
        "p90": -0.0010382708606262714,
        "p95": -0.0010382708606262714
      },
      "saturation_fraction": {
        "mean": 0.03156628894309203,
        "median": 0.03156628894309203,
        "p90": 0.03156628894309203,
        "p95": 0.03156628894309203
      },
      "scale_max": {
        "mean": 0.029542301470807327,
        "median": 0.029542301470807327,
        "p90": 0.029542301470807327,
        "p95": 0.029542301470807327
      },
      "scale_mean": {
        "mean": 0.00019729877111671163,
        "median": 0.00019729877111671163,
        "p90": 0.00019729877111671163,
        "p95": 0.00019729877111671163
      },
      "scale_median": {
        "mean": 8.525725640125734e-05,
        "median": 8.525725640125734e-05,
        "p90": 8.525725640125734e-05,
        "p95": 8.525725640125734e-05
      },
      "scale_p95_over_median": {
        "mean": 6.63059486511184,
        "median": 6.63059486511184,
        "p90": 6.63059486511184,
        "p95": 6.63059486511184
      },
      "state_change_error_E_Delta": {
        "mean": 0.31048336421989214,
        "median": 0.31048336421989214,
        "p90": 0.31048336421989214,
        "p95": 0.31048336421989214
      },
      "state_norm": {
        "mean": 19.977400775766007,
        "median": 19.977400775766007,
        "p90": 19.977400775766007,
        "p95": 19.977400775766007
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.005468650574568246,
        "median": 0.005468650574568246,
        "p90": 0.005468650574568246,
        "p95": 0.005468650574568246
      },
      "survival_fraction": {
        "mean": 0.5894387913867829,
        "median": 0.5894387913867829,
        "p90": 0.5894387913867829,
        "p95": 0.5894387913867829
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9426431791507638,
        "median": 0.9426431791507638,
        "p90": 0.9426431791507638,
        "p95": 0.9426431791507638
      },
      "update_distortion_ratio": {
        "mean": 0.10989039585982724,
        "median": 0.10989039585982724,
        "p90": 0.10989039585982724,
        "p95": 0.10989039585982724
      },
      "within_group_dynamic_range_p95": {
        "mean": 24.105522312774944,
        "median": 24.105522312774944,
        "p90": 24.105522312774944,
        "p95": 24.105522312774944
      }
    }
  },
  "C64": {
    "1-128": {
      "KL": {
        "mean": 0.00041928417270093777,
        "median": 0.00041928417270093777,
        "p90": 0.00041928417270093777,
        "p95": 0.00041928417270093777
      },
      "delta_norm": {
        "mean": 2.910771419677117,
        "median": 2.910771419677117,
        "p90": 2.910771419677117,
        "p95": 2.910771419677117
      },
      "lost_update_fraction": {
        "mean": 0.04763839867112859,
        "median": 0.04763839867112859,
        "p90": 0.04763839867112859,
        "p95": 0.04763839867112859
      },
      "represented_update_norm_ratio": {
        "mean": 0.950626479055145,
        "median": 0.950626479055145,
        "p90": 0.950626479055145,
        "p95": 0.950626479055145
      },
      "residual_update_cosine": {
        "mean": -0.009881917643767566,
        "median": -0.009881917643767566,
        "p90": -0.009881917643767566,
        "p95": -0.009881917643767566
      },
      "residual_update_projection_coeff": {
        "mean": -0.0007703799216505182,
        "median": -0.0007703799216505182,
        "p90": -0.0007703799216505182,
        "p95": -0.0007703799216505182
      },
      "saturation_fraction": {
        "mean": 0.015810837582966163,
        "median": 0.015810837582966163,
        "p90": 0.015810837582966163,
        "p95": 0.015810837582966163
      },
      "scale_max": {
        "mean": 0.026627953488393762,
        "median": 0.026627953488393762,
        "p90": 0.026627953488393762,
        "p95": 0.026627953488393762
      },
      "scale_mean": {
        "mean": 0.00022376499139623444,
        "median": 0.00022376499139623444,
        "p90": 0.00022376499139623444,
        "p95": 0.00022376499139623444
      },
      "scale_median": {
        "mean": 0.00010317063613005844,
        "median": 0.00010317063613005844,
        "p90": 0.00010317063613005844,
        "p95": 0.00010317063613005844
      },
      "scale_p95_over_median": {
        "mean": 6.2828135031682875,
        "median": 6.2828135031682875,
        "p90": 6.2828135031682875,
        "p95": 6.2828135031682875
      },
      "state_change_error_E_Delta": {
        "mean": 0.2547870544401365,
        "median": 0.2547870544401365,
        "p90": 0.2547870544401365,
        "p95": 0.2547870544401365
      },
      "state_norm": {
        "mean": 18.058283185943218,
        "median": 18.058283185943218,
        "p90": 18.058283185943218,
        "p95": 18.058283185943218
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.0077176027338929,
        "median": 0.0077176027338929,
        "p90": 0.0077176027338929,
        "p95": 0.0077176027338929
      },
      "survival_fraction": {
        "mean": 0.6669584002707576,
        "median": 0.6669584002707576,
        "p90": 0.6669584002707576,
        "p95": 0.6669584002707576
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9609245695662619,
        "median": 0.9609245695662619,
        "p90": 0.9609245695662619,
        "p95": 0.9609245695662619
      },
      "update_distortion_ratio": {
        "mean": 0.07579162001880344,
        "median": 0.07579162001880344,
        "p90": 0.07579162001880344,
        "p95": 0.07579162001880344
      },
      "within_group_dynamic_range_p95": {
        "mean": 27.01410881651668,
        "median": 27.01410881651668,
        "p90": 27.01410881651668,
        "p95": 27.01410881651668
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.0008064976120401808,
        "median": 0.0008064976120401808,
        "p90": 0.0008064976120401808,
        "p95": 0.0008064976120401808
      },
      "delta_norm": {
        "mean": 2.8008666637343804,
        "median": 2.8008666637343804,
        "p90": 2.8008666637343804,
        "p95": 2.8008666637343804
      },
      "lost_update_fraction": {
        "mean": 0.051706183146306185,
        "median": 0.051706183146306185,
        "p90": 0.051706183146306185,
        "p95": 0.051706183146306185
      },
      "represented_update_norm_ratio": {
        "mean": 0.9482081645309237,
        "median": 0.9482081645309237,
        "p90": 0.9482081645309237,
        "p95": 0.9482081645309237
      },
      "residual_update_cosine": {
        "mean": -0.011920502796243682,
        "median": -0.011920502796243682,
        "p90": -0.011920502796243682,
        "p95": -0.011920502796243682
      },
      "residual_update_projection_coeff": {
        "mean": -0.000963679547649488,
        "median": -0.000963679547649488,
        "p90": -0.000963679547649488,
        "p95": -0.000963679547649488
      },
      "saturation_fraction": {
        "mean": 0.015810413286089897,
        "median": 0.015810413286089897,
        "p90": 0.015810413286089897,
        "p95": 0.015810413286089897
      },
      "scale_max": {
        "mean": 0.028596748308094305,
        "median": 0.028596748308094305,
        "p90": 0.028596748308094305,
        "p95": 0.028596748308094305
      },
      "scale_mean": {
        "mean": 0.00023882829193048736,
        "median": 0.00023882829193048736,
        "p90": 0.00023882829193048736,
        "p95": 0.00023882829193048736
      },
      "scale_median": {
        "mean": 0.00010849807088142431,
        "median": 0.00010849807088142431,
        "p90": 0.00010849807088142431,
        "p95": 0.00010849807088142431
      },
      "scale_p95_over_median": {
        "mean": 6.4173763941229245,
        "median": 6.4173763941229245,
        "p90": 6.4173763941229245,
        "p95": 6.4173763941229245
      },
      "state_change_error_E_Delta": {
        "mean": 0.26161718006673435,
        "median": 0.26161718006673435,
        "p90": 0.26161718006673435,
        "p95": 0.26161718006673435
      },
      "state_norm": {
        "mean": 19.600526856180906,
        "median": 19.600526856180906,
        "p90": 19.600526856180906,
        "p95": 19.600526856180906
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.00722686029684508,
        "median": 0.00722686029684508,
        "p90": 0.00722686029684508,
        "p95": 0.00722686029684508
      },
      "survival_fraction": {
        "mean": 0.6259127166122195,
        "median": 0.6259127166122195,
        "p90": 0.6259127166122195,
        "p95": 0.6259127166122195
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9583585181389935,
        "median": 0.9583585181389935,
        "p90": 0.9583585181389935,
        "p95": 0.9583585181389935
      },
      "update_distortion_ratio": {
        "mean": 0.08058730541429618,
        "median": 0.08058730541429618,
        "p90": 0.08058730541429618,
        "p95": 0.08058730541429618
      },
      "within_group_dynamic_range_p95": {
        "mean": 28.257236163209495,
        "median": 28.257236163209495,
        "p90": 28.257236163209495,
        "p95": 28.257236163209495
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.0006387008372198764,
        "median": 0.0006387008372198764,
        "p90": 0.0006387008372198764,
        "p95": 0.0006387008372198764
      },
      "delta_norm": {
        "mean": 2.3429059253271047,
        "median": 2.3429059253271047,
        "p90": 2.3429059253271047,
        "p95": 2.3429059253271047
      },
      "lost_update_fraction": {
        "mean": 0.05378047989393971,
        "median": 0.05378047989393971,
        "p90": 0.05378047989393971,
        "p95": 0.05378047989393971
      },
      "represented_update_norm_ratio": {
        "mean": 0.9483240586546166,
        "median": 0.9483240586546166,
        "p90": 0.9483240586546166,
        "p95": 0.9483240586546166
      },
      "residual_update_cosine": {
        "mean": -0.015668402476317546,
        "median": -0.015668402476317546,
        "p90": -0.015668402476317546,
        "p95": -0.015668402476317546
      },
      "residual_update_projection_coeff": {
        "mean": -0.0015249023563919891,
        "median": -0.0015249023563919891,
        "p90": -0.0015249023563919891,
        "p95": -0.0015249023563919891
      },
      "saturation_fraction": {
        "mean": 0.015810300596058365,
        "median": 0.015810300596058365,
        "p90": 0.015810300596058365,
        "p95": 0.015810300596058365
      },
      "scale_max": {
        "mean": 0.029632978675939135,
        "median": 0.029632978675939135,
        "p90": 0.029632978675939135,
        "p95": 0.029632978675939135
      },
      "scale_mean": {
        "mean": 0.0002457192586034581,
        "median": 0.0002457192586034581,
        "p90": 0.0002457192586034581,
        "p95": 0.0002457192586034581
      },
      "scale_median": {
        "mean": 0.00011077768216623724,
        "median": 0.00011077768216623724,
        "p90": 0.00011077768216623724,
        "p95": 0.00011077768216623724
      },
      "scale_p95_over_median": {
        "mean": 6.392837064231269,
        "median": 6.392837064231269,
        "p90": 6.392837064231269,
        "p95": 6.392837064231269
      },
      "state_change_error_E_Delta": {
        "mean": 0.2658741480580112,
        "median": 0.2658741480580112,
        "p90": 0.2658741480580112,
        "p95": 0.2658741480580112
      },
      "state_norm": {
        "mean": 20.023301674324713,
        "median": 20.023301674324713,
        "p90": 20.023301674324713,
        "p95": 20.023301674324713
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.0064767362268502975,
        "median": 0.0064767362268502975,
        "p90": 0.0064767362268502975,
        "p95": 0.0064767362268502975
      },
      "survival_fraction": {
        "mean": 0.5607890235260127,
        "median": 0.5607890235260127,
        "p90": 0.5607890235260127,
        "p95": 0.5607890235260127
      },
      "top1_agreement": {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0
      },
      "update_cosine": {
        "mean": 0.9572510783036705,
        "median": 0.9572510783036705,
        "p90": 0.9572510783036705,
        "p95": 0.9572510783036705
      },
      "update_distortion_ratio": {
        "mean": 0.08271146669394917,
        "median": 0.08271146669394917,
        "p90": 0.08271146669394917,
        "p95": 0.08271146669394917
      },
      "within_group_dynamic_range_p95": {
        "mean": 28.220997902813043,
        "median": 28.220997902813043,
        "p90": 28.220997902813043,
        "p95": 28.220997902813043
      }
    }
  },
  "R128": {
    "1-128": {
      "KL": {
        "mean": 0.05782031027243924,
        "median": 0.05782031027243924,
        "p90": 0.05782031027243924,
        "p95": 0.05782031027243924
      },
      "delta_norm": {
        "mean": 2.9229261624744645,
        "median": 2.9229261624744645,
        "p90": 2.9229261624744645,
        "p95": 2.9229261624744645
      },
      "lost_update_fraction": {
        "mean": 0.12249155777285241,
        "median": 0.12249155777285241,
        "p90": 0.12249155777285241,
        "p95": 0.12249155777285241
      },
      "represented_update_norm_ratio": {
        "mean": 0.9024301431203966,
        "median": 0.9024301431203966,
        "p90": 0.9024301431203966,
        "p95": 0.9024301431203966
      },
      "residual_update_cosine": {
        "mean": -0.01827481251829939,
        "median": -0.01827481251829939,
        "p90": -0.01827481251829939,
        "p95": -0.01827481251829939
      },
      "residual_update_projection_coeff": {
        "mean": -0.0018656707083409094,
        "median": -0.0018656707083409094,
        "p90": -0.0018656707083409094,
        "p95": -0.0018656707083409094
      },
      "saturation_fraction": {
        "mean": 0.00785659742480501,
        "median": 0.00785659742480501,
        "p90": 0.00785659742480501,
        "p95": 0.00785659742480501
      },
      "scale_max": {
        "mean": 0.02653789057359992,
        "median": 0.02653789057359992,
        "p90": 0.02653789057359992,
        "p95": 0.02653789057359992
      },
      "scale_mean": {
        "mean": 0.0009965538973667498,
        "median": 0.0009965538973667498,
        "p90": 0.0009965538973667498,
        "p95": 0.0009965538973667498
      },
      "scale_median": {
        "mean": 0.00047708660811806203,
        "median": 0.00047708660811806203,
        "p90": 0.00047708660811806203,
        "p95": 0.00047708660811806203
      },
      "scale_p95_over_median": {
        "mean": 8.406016416681124,
        "median": 8.406016416681124,
        "p90": 8.406016416681124,
        "p95": 8.406016416681124
      },
      "state_change_error_E_Delta": {
        "mean": 0.37989876241260934,
        "median": 0.37989876241260934,
        "p90": 0.37989876241260934,
        "p95": 0.37989876241260934
      },
      "state_norm": {
        "mean": 17.98520183805718,
        "median": 17.98520183805718,
        "p90": 17.98520183805718,
        "p95": 17.98520183805718
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.011625390044275687,
        "median": 0.011625390044275687,
        "p90": 0.011625390044275687,
        "p95": 0.011625390044275687
      },
      "survival_fraction": {
        "mean": 0.5131133422451069,
        "median": 0.5131133422451069,
        "p90": 0.5131133422451069,
        "p95": 0.5131133422451069
      },
      "top1_agreement": {
        "mean": 0.9765625,
        "median": 0.9765625,
        "p90": 0.9765625,
        "p95": 0.9765625
      },
      "update_cosine": {
        "mean": 0.9127704368762457,
        "median": 0.9127704368762457,
        "p90": 0.9127704368762457,
        "p95": 0.9127704368762457
      },
      "update_distortion_ratio": {
        "mean": 0.16291185439542424,
        "median": 0.16291185439542424,
        "p90": 0.16291185439542424,
        "p95": 0.16291185439542424
      },
      "within_group_dynamic_range_p95": {
        "mean": 3689.303976305514,
        "median": 3689.303976305514,
        "p90": 3689.303976305514,
        "p95": 3689.303976305514
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.042163560400272214,
        "median": 0.042163560400272214,
        "p90": 0.042163560400272214,
        "p95": 0.042163560400272214
      },
      "delta_norm": {
        "mean": 2.8220911663230446,
        "median": 2.8220911663230446,
        "p90": 2.8220911663230446,
        "p95": 2.8220911663230446
      },
      "lost_update_fraction": {
        "mean": 0.13102271893758027,
        "median": 0.13102271893758027,
        "p90": 0.13102271893758027,
        "p95": 0.13102271893758027
      },
      "represented_update_norm_ratio": {
        "mean": 0.897330088637549,
        "median": 0.897330088637549,
        "p90": 0.897330088637549,
        "p95": 0.897330088637549
      },
      "residual_update_cosine": {
        "mean": -0.020168941226756956,
        "median": -0.020168941226756956,
        "p90": -0.020168941226756956,
        "p95": -0.020168941226756956
      },
      "residual_update_projection_coeff": {
        "mean": -0.0021632386776528844,
        "median": -0.0021632386776528844,
        "p90": -0.0021632386776528844,
        "p95": -0.0021632386776528844
      },
      "saturation_fraction": {
        "mean": 0.007854346806804337,
        "median": 0.007854346806804337,
        "p90": 0.007854346806804337,
        "p95": 0.007854346806804337
      },
      "scale_max": {
        "mean": 0.028499232671038054,
        "median": 0.028499232671038054,
        "p90": 0.028499232671038054,
        "p95": 0.028499232671038054
      },
      "scale_mean": {
        "mean": 0.0010809205586313872,
        "median": 0.0010809205586313872,
        "p90": 0.0010809205586313872,
        "p95": 0.0010809205586313872
      },
      "scale_median": {
        "mean": 0.0005122400032983402,
        "median": 0.0005122400032983402,
        "p90": 0.0005122400032983402,
        "p95": 0.0005122400032983402
      },
      "scale_p95_over_median": {
        "mean": 8.407109900497494,
        "median": 8.407109900497494,
        "p90": 8.407109900497494,
        "p95": 8.407109900497494
      },
      "state_change_error_E_Delta": {
        "mean": 0.3916562144209176,
        "median": 0.3916562144209176,
        "p90": 0.3916562144209176,
        "p95": 0.3916562144209176
      },
      "state_norm": {
        "mean": 19.515205747059856,
        "median": 19.515205747059856,
        "p90": 19.515205747059856,
        "p95": 19.515205747059856
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.010892626101520982,
        "median": 0.010892626101520982,
        "p90": 0.010892626101520982,
        "p95": 0.010892626101520982
      },
      "survival_fraction": {
        "mean": 0.47624248638749117,
        "median": 0.47624248638749117,
        "p90": 0.47624248638749117,
        "p95": 0.47624248638749117
      },
      "top1_agreement": {
        "mean": 0.9453125,
        "median": 0.9453125,
        "p90": 0.9453125,
        "p95": 0.9453125
      },
      "update_cosine": {
        "mean": 0.9071434984992567,
        "median": 0.9071434984992567,
        "p90": 0.9071434984992567,
        "p95": 0.9071434984992567
      },
      "update_distortion_ratio": {
        "mean": 0.17257676644863018,
        "median": 0.17257676644863018,
        "p90": 0.17257676644863018,
        "p95": 0.17257676644863018
      },
      "within_group_dynamic_range_p95": {
        "mean": 4195.001010424147,
        "median": 4195.001010424147,
        "p90": 4195.001010424147,
        "p95": 4195.001010424147
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.04098586948015295,
        "median": 0.04098586948015295,
        "p90": 0.04098586948015295,
        "p95": 0.04098586948015295
      },
      "delta_norm": {
        "mean": 2.366202690472533,
        "median": 2.366202690472533,
        "p90": 2.366202690472533,
        "p95": 2.366202690472533
      },
      "lost_update_fraction": {
        "mean": 0.12903923117440252,
        "median": 0.12903923117440252,
        "p90": 0.12903923117440252,
        "p95": 0.12903923117440252
      },
      "represented_update_norm_ratio": {
        "mean": 0.902351097035346,
        "median": 0.902351097035346,
        "p90": 0.902351097035346,
        "p95": 0.902351097035346
      },
      "residual_update_cosine": {
        "mean": -0.02422063209309993,
        "median": -0.02422063209309993,
        "p90": -0.02422063209309993,
        "p95": -0.02422063209309993
      },
      "residual_update_projection_coeff": {
        "mean": -0.002921027981836478,
        "median": -0.002921027981836478,
        "p90": -0.002921027981836478,
        "p95": -0.002921027981836478
      },
      "saturation_fraction": {
        "mean": 0.007856795874734717,
        "median": 0.007856795874734717,
        "p90": 0.007856795874734717,
        "p95": 0.007856795874734717
      },
      "scale_max": {
        "mean": 0.0295273446461124,
        "median": 0.0295273446461124,
        "p90": 0.0295273446461124,
        "p95": 0.0295273446461124
      },
      "scale_mean": {
        "mean": 0.001099403132080467,
        "median": 0.001099403132080467,
        "p90": 0.001099403132080467,
        "p95": 0.001099403132080467
      },
      "scale_median": {
        "mean": 0.0005224147044686764,
        "median": 0.0005224147044686764,
        "p90": 0.0005224147044686764,
        "p95": 0.0005224147044686764
      },
      "scale_p95_over_median": {
        "mean": 8.33508076846807,
        "median": 8.33508076846807,
        "p90": 8.33508076846807,
        "p95": 8.33508076846807
      },
      "state_change_error_E_Delta": {
        "mean": 0.3848883175866188,
        "median": 0.3848883175866188,
        "p90": 0.3848883175866188,
        "p95": 0.3848883175866188
      },
      "state_norm": {
        "mean": 19.911678709244985,
        "median": 19.911678709244985,
        "p90": 19.911678709244985,
        "p95": 19.911678709244985
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.009815534507173028,
        "median": 0.009815534507173028,
        "p90": 0.009815534507173028,
        "p95": 0.009815534507173028
      },
      "survival_fraction": {
        "mean": 0.4255667971447113,
        "median": 0.4255667971447113,
        "p90": 0.4255667971447113,
        "p95": 0.4255667971447113
      },
      "top1_agreement": {
        "mean": 0.94921875,
        "median": 0.94921875,
        "p90": 0.94921875,
        "p95": 0.94921875
      },
      "update_cosine": {
        "mean": 0.9089882971456975,
        "median": 0.9089882971456975,
        "p90": 0.9089882971456975,
        "p95": 0.9089882971456975
      },
      "update_distortion_ratio": {
        "mean": 0.16862831912465875,
        "median": 0.16862831912465875,
        "p90": 0.16862831912465875,
        "p95": 0.16862831912465875
      },
      "within_group_dynamic_range_p95": {
        "mean": 6648.19461498836,
        "median": 6648.19461498836,
        "p90": 6648.19461498836,
        "p95": 6648.19461498836
      }
    }
  },
  "R16": {
    "1-128": {
      "KL": {
        "mean": 0.0011671016387425062,
        "median": 0.0011671016387425062,
        "p90": 0.0011671016387425062,
        "p95": 0.0011671016387425062
      },
      "delta_norm": {
        "mean": 2.91110355215905,
        "median": 2.91110355215905,
        "p90": 2.91110355215905,
        "p95": 2.91110355215905
      },
      "lost_update_fraction": {
        "mean": 0.17235494555775066,
        "median": 0.17235494555775066,
        "p90": 0.17235494555775066,
        "p95": 0.17235494555775066
      },
      "represented_update_norm_ratio": {
        "mean": 0.8604399945718335,
        "median": 0.8604399945718335,
        "p90": 0.8604399945718335,
        "p95": 0.8604399945718335
      },
      "residual_update_cosine": {
        "mean": -0.0066019979431004,
        "median": -0.0066019979431004,
        "p90": -0.0066019979431004,
        "p95": -0.0066019979431004
      },
      "residual_update_projection_coeff": {
        "mean": -0.0003151011548514211,
        "median": -0.0003151011548514211,
        "p90": -0.0003151011548514211,
        "p95": -0.0003151011548514211
      },
      "saturation_fraction": {
        "mean": 0.06296616026109911,
        "median": 0.06296616026109911,
        "p90": 0.06296616026109911,
        "p95": 0.06296616026109911
      },
      "scale_max": {
        "mean": 0.026591151738322376,
        "median": 0.026591151738322376,
        "p90": 0.026591151738322376,
        "p95": 0.026591151738322376
      },
      "scale_mean": {
        "mean": 0.00022020136840338655,
        "median": 0.00022020136840338655,
        "p90": 0.00022020136840338655,
        "p95": 0.00022020136840338655
      },
      "scale_median": {
        "mean": 6.22988857658111e-05,
        "median": 6.22988857658111e-05,
        "p90": 6.22988857658111e-05,
        "p95": 6.22988857658111e-05
      },
      "scale_p95_over_median": {
        "mean": 15.91959692030447,
        "median": 15.91959692030447,
        "p90": 15.91959692030447,
        "p95": 15.91959692030447
      },
      "state_change_error_E_Delta": {
        "mean": 0.44347678907171556,
        "median": 0.44347678907171556,
        "p90": 0.44347678907171556,
        "p95": 0.44347678907171556
      },
      "state_norm": {
        "mean": 18.02437713969098,
        "median": 18.02437713969098,
        "p90": 18.02437713969098,
        "p95": 18.02437713969098
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.0049477350593744,
        "median": 0.0049477350593744,
        "p90": 0.0049477350593744,
        "p95": 0.0049477350593744
      },
      "survival_fraction": {
        "mean": 0.698907653177817,
        "median": 0.698907653177817,
        "p90": 0.698907653177817,
        "p95": 0.698907653177817
      },
      "top1_agreement": {
        "mean": 0.9921875,
        "median": 0.9921875,
        "p90": 0.9921875,
        "p95": 0.9921875
      },
      "update_cosine": {
        "mean": 0.8832839903497475,
        "median": 0.8832839903497475,
        "p90": 0.8832839903497475,
        "p95": 0.8832839903497475
      },
      "update_distortion_ratio": {
        "mean": 0.21438901795928555,
        "median": 0.21438901795928555,
        "p90": 0.21438901795928555,
        "p95": 0.21438901795928555
      },
      "within_group_dynamic_range_p95": {
        "mean": 222.99172766641954,
        "median": 222.99172766641954,
        "p90": 222.99172766641954,
        "p95": 222.99172766641954
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.003507990020373739,
        "median": 0.003507990020373739,
        "p90": 0.003507990020373739,
        "p95": 0.003507990020373739
      },
      "delta_norm": {
        "mean": 2.801861660831306,
        "median": 2.801861660831306,
        "p90": 2.801861660831306,
        "p95": 2.801861660831306
      },
      "lost_update_fraction": {
        "mean": 0.18234827458946865,
        "median": 0.18234827458946865,
        "p90": 0.18234827458946865,
        "p95": 0.18234827458946865
      },
      "represented_update_norm_ratio": {
        "mean": 0.8542648431414122,
        "median": 0.8542648431414122,
        "p90": 0.8542648431414122,
        "p95": 0.8542648431414122
      },
      "residual_update_cosine": {
        "mean": -0.007609917598344397,
        "median": -0.007609917598344397,
        "p90": -0.007609917598344397,
        "p95": -0.007609917598344397
      },
      "residual_update_projection_coeff": {
        "mean": -0.00037963618312025543,
        "median": -0.00037963618312025543,
        "p90": -0.00037963618312025543,
        "p95": -0.00037963618312025543
      },
      "saturation_fraction": {
        "mean": 0.06296083579460778,
        "median": 0.06296083579460778,
        "p90": 0.06296083579460778,
        "p95": 0.06296083579460778
      },
      "scale_max": {
        "mean": 0.028583771963212712,
        "median": 0.028583771963212712,
        "p90": 0.028583771963212712,
        "p95": 0.028583771963212712
      },
      "scale_mean": {
        "mean": 0.00023576610159617694,
        "median": 0.00023576610159617694,
        "p90": 0.00023576610159617694,
        "p95": 0.00023576610159617694
      },
      "scale_median": {
        "mean": 6.348446048624427e-05,
        "median": 6.348446048624427e-05,
        "p90": 6.348446048624427e-05,
        "p95": 6.348446048624427e-05
      },
      "scale_p95_over_median": {
        "mean": 16.03881762369961,
        "median": 16.03881762369961,
        "p90": 16.03881762369961,
        "p95": 16.03881762369961
      },
      "state_change_error_E_Delta": {
        "mean": 0.45395836521561844,
        "median": 0.45395836521561844,
        "p90": 0.45395836521561844,
        "p95": 0.45395836521561844
      },
      "state_norm": {
        "mean": 19.522762116588037,
        "median": 19.522762116588037,
        "p90": 19.522762116588037,
        "p95": 19.522762116588037
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004639411326138806,
        "median": 0.004639411326138806,
        "p90": 0.004639411326138806,
        "p95": 0.004639411326138806
      },
      "survival_fraction": {
        "mean": 0.6620654047777257,
        "median": 0.6620654047777257,
        "p90": 0.6620654047777257,
        "p95": 0.6620654047777257
      },
      "top1_agreement": {
        "mean": 0.984375,
        "median": 0.984375,
        "p90": 0.984375,
        "p95": 0.984375
      },
      "update_cosine": {
        "mean": 0.8766104655126884,
        "median": 0.8766104655126884,
        "p90": 0.8766104655126884,
        "p95": 0.8766104655126884
      },
      "update_distortion_ratio": {
        "mean": 0.22512634569476844,
        "median": 0.22512634569476844,
        "p90": 0.22512634569476844,
        "p95": 0.22512634569476844
      },
      "within_group_dynamic_range_p95": {
        "mean": 216.56011614312044,
        "median": 216.56011614312044,
        "p90": 216.56011614312044,
        "p95": 216.56011614312044
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.0049178909510329445,
        "median": 0.0049178909510329445,
        "p90": 0.0049178909510329445,
        "p95": 0.0049178909510329445
      },
      "delta_norm": {
        "mean": 2.3443943771736047,
        "median": 2.3443943771736047,
        "p90": 2.3443943771736047,
        "p95": 2.3443943771736047
      },
      "lost_update_fraction": {
        "mean": 0.17882797317997642,
        "median": 0.17882797317997642,
        "p90": 0.17882797317997642,
        "p95": 0.17882797317997642
      },
      "represented_update_norm_ratio": {
        "mean": 0.8590500994787297,
        "median": 0.8590500994787297,
        "p90": 0.8590500994787297,
        "p95": 0.8590500994787297
      },
      "residual_update_cosine": {
        "mean": -0.009317075777464844,
        "median": -0.009317075777464844,
        "p90": -0.009317075777464844,
        "p95": -0.009317075777464844
      },
      "residual_update_projection_coeff": {
        "mean": -0.0005513368935113665,
        "median": -0.0005513368935113665,
        "p90": -0.0005513368935113665,
        "p95": -0.0005513368935113665
      },
      "saturation_fraction": {
        "mean": 0.0629602111876011,
        "median": 0.0629602111876011,
        "p90": 0.0629602111876011,
        "p95": 0.0629602111876011
      },
      "scale_max": {
        "mean": 0.029664670331915954,
        "median": 0.029664670331915954,
        "p90": 0.029664670331915954,
        "p95": 0.029664670331915954
      },
      "scale_mean": {
        "mean": 0.00024161256446469573,
        "median": 0.00024161256446469573,
        "p90": 0.00024161256446469573,
        "p95": 0.00024161256446469573
      },
      "scale_median": {
        "mean": 6.525212258913962e-05,
        "median": 6.525212258913962e-05,
        "p90": 6.525212258913962e-05,
        "p95": 6.525212258913962e-05
      },
      "scale_p95_over_median": {
        "mean": 15.820286632224313,
        "median": 15.820286632224313,
        "p90": 15.820286632224313,
        "p95": 15.820286632224313
      },
      "state_change_error_E_Delta": {
        "mean": 0.4474413601586115,
        "median": 0.4474413601586115,
        "p90": 0.4474413601586115,
        "p95": 0.4474413601586115
      },
      "state_norm": {
        "mean": 19.87820385152008,
        "median": 19.87820385152008,
        "p90": 19.87820385152008,
        "p95": 19.87820385152008
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.004227528352555537,
        "median": 0.004227528352555537,
        "p90": 0.004227528352555537,
        "p95": 0.004227528352555537
      },
      "survival_fraction": {
        "mean": 0.6022155275568364,
        "median": 0.6022155275568364,
        "p90": 0.6022155275568364,
        "p95": 0.6022155275568364
      },
      "top1_agreement": {
        "mean": 0.984375,
        "median": 0.984375,
        "p90": 0.984375,
        "p95": 0.984375
      },
      "update_cosine": {
        "mean": 0.8797059290275988,
        "median": 0.8797059290275988,
        "p90": 0.8797059290275988,
        "p95": 0.8797059290275988
      },
      "update_distortion_ratio": {
        "mean": 0.21960221306207856,
        "median": 0.21960221306207856,
        "p90": 0.21960221306207856,
        "p95": 0.21960221306207856
      },
      "within_group_dynamic_range_p95": {
        "mean": 256.00043944707045,
        "median": 256.00043944707045,
        "p90": 256.00043944707045,
        "p95": 256.00043944707045
      }
    }
  },
  "R32": {
    "1-128": {
      "KL": {
        "mean": 0.0024112943783864502,
        "median": 0.0024112943783864502,
        "p90": 0.0024112943783864502,
        "p95": 0.0024112943783864502
      },
      "delta_norm": {
        "mean": 2.912495628213567,
        "median": 2.912495628213567,
        "p90": 2.912495628213567,
        "p95": 2.912495628213567
      },
      "lost_update_fraction": {
        "mean": 0.15518935949421905,
        "median": 0.15518935949421905,
        "p90": 0.15518935949421905,
        "p95": 0.15518935949421905
      },
      "represented_update_norm_ratio": {
        "mean": 0.874439372638055,
        "median": 0.874439372638055,
        "p90": 0.874439372638055,
        "p95": 0.874439372638055
      },
      "residual_update_cosine": {
        "mean": -0.009559223975523926,
        "median": -0.009559223975523926,
        "p90": -0.009559223975523926,
        "p95": -0.009559223975523926
      },
      "residual_update_projection_coeff": {
        "mean": -0.0005982527445686348,
        "median": -0.0005982527445686348,
        "p90": -0.0005982527445686348,
        "p95": -0.0005982527445686348
      },
      "saturation_fraction": {
        "mean": 0.03149423574212343,
        "median": 0.03149423574212343,
        "p90": 0.03149423574212343,
        "p95": 0.03149423574212343
      },
      "scale_max": {
        "mean": 0.026567618183628464,
        "median": 0.026567618183628464,
        "p90": 0.026567618183628464,
        "p95": 0.026567618183628464
      },
      "scale_mean": {
        "mean": 0.0003557840758768586,
        "median": 0.0003557840758768586,
        "p90": 0.0003557840758768586,
        "p95": 0.0003557840758768586
      },
      "scale_median": {
        "mean": 9.122955572735165e-05,
        "median": 9.122955572735165e-05,
        "p90": 9.122955572735165e-05,
        "p95": 9.122955572735165e-05
      },
      "scale_p95_over_median": {
        "mean": 18.61474480161007,
        "median": 18.61474480161007,
        "p90": 18.61474480161007,
        "p95": 18.61474480161007
      },
      "state_change_error_E_Delta": {
        "mean": 0.4211343938965472,
        "median": 0.4211343938965472,
        "p90": 0.4211343938965472,
        "p95": 0.4211343938965472
      },
      "state_norm": {
        "mean": 18.010965444753797,
        "median": 18.010965444753797,
        "p90": 18.010965444753797,
        "p95": 18.010965444753797
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006671804377306441,
        "median": 0.006671804377306441,
        "p90": 0.006671804377306441,
        "p95": 0.006671804377306441
      },
      "survival_fraction": {
        "mean": 0.6524768901935085,
        "median": 0.6524768901935085,
        "p90": 0.6524768901935085,
        "p95": 0.6524768901935085
      },
      "top1_agreement": {
        "mean": 0.984375,
        "median": 0.984375,
        "p90": 0.984375,
        "p95": 0.984375
      },
      "update_cosine": {
        "mean": 0.8938297236685362,
        "median": 0.8938297236685362,
        "p90": 0.8938297236685362,
        "p95": 0.8938297236685362
      },
      "update_distortion_ratio": {
        "mean": 0.1959739561688758,
        "median": 0.1959739561688758,
        "p90": 0.1959739561688758,
        "p95": 0.1959739561688758
      },
      "within_group_dynamic_range_p95": {
        "mean": 720.5944236709202,
        "median": 720.5944236709202,
        "p90": 720.5944236709202,
        "p95": 720.5944236709202
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.0078974717254864,
        "median": 0.0078974717254864,
        "p90": 0.0078974717254864,
        "p95": 0.0078974717254864
      },
      "delta_norm": {
        "mean": 2.807018066224219,
        "median": 2.807018066224219,
        "p90": 2.807018066224219,
        "p95": 2.807018066224219
      },
      "lost_update_fraction": {
        "mean": 0.16517955668409237,
        "median": 0.16517955668409237,
        "p90": 0.16517955668409237,
        "p95": 0.16517955668409237
      },
      "represented_update_norm_ratio": {
        "mean": 0.8682763289654245,
        "median": 0.8682763289654245,
        "p90": 0.8682763289654245,
        "p95": 0.8682763289654245
      },
      "residual_update_cosine": {
        "mean": -0.01062374383484628,
        "median": -0.01062374383484628,
        "p90": -0.01062374383484628,
        "p95": -0.01062374383484628
      },
      "residual_update_projection_coeff": {
        "mean": -0.0006944318137182919,
        "median": -0.0006944318137182919,
        "p90": -0.0006944318137182919,
        "p95": -0.0006944318137182919
      },
      "saturation_fraction": {
        "mean": 0.03148949022094408,
        "median": 0.03148949022094408,
        "p90": 0.03148949022094408,
        "p95": 0.03148949022094408
      },
      "scale_max": {
        "mean": 0.028562324750510005,
        "median": 0.028562324750510005,
        "p90": 0.028562324750510005,
        "p95": 0.028562324750510005
      },
      "scale_mean": {
        "mean": 0.0003827911945124115,
        "median": 0.0003827911945124115,
        "p90": 0.0003827911945124115,
        "p95": 0.0003827911945124115
      },
      "scale_median": {
        "mean": 9.346492018476778e-05,
        "median": 9.346492018476778e-05,
        "p90": 9.346492018476778e-05,
        "p95": 9.346492018476778e-05
      },
      "scale_p95_over_median": {
        "mean": 18.712192118870828,
        "median": 18.712192118870828,
        "p90": 18.712192118870828,
        "p95": 18.712192118870828
      },
      "state_change_error_E_Delta": {
        "mean": 0.4322271292991611,
        "median": 0.4322271292991611,
        "p90": 0.4322271292991611,
        "p95": 0.4322271292991611
      },
      "state_norm": {
        "mean": 19.521503292179357,
        "median": 19.521503292179357,
        "p90": 19.521503292179357,
        "p95": 19.521503292179357
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.006242220649452228,
        "median": 0.006242220649452228,
        "p90": 0.006242220649452228,
        "p95": 0.006242220649452228
      },
      "survival_fraction": {
        "mean": 0.614367333551248,
        "median": 0.614367333551248,
        "p90": 0.614367333551248,
        "p95": 0.614367333551248
      },
      "top1_agreement": {
        "mean": 0.96875,
        "median": 0.96875,
        "p90": 0.96875,
        "p95": 0.96875
      },
      "update_cosine": {
        "mean": 0.8872756551621322,
        "median": 0.8872756551621322,
        "p90": 0.8872756551621322,
        "p95": 0.8872756551621322
      },
      "update_distortion_ratio": {
        "mean": 0.20670187276919685,
        "median": 0.20670187276919685,
        "p90": 0.20670187276919685,
        "p95": 0.20670187276919685
      },
      "within_group_dynamic_range_p95": {
        "mean": 795.0421616323115,
        "median": 795.0421616323115,
        "p90": 795.0421616323115,
        "p95": 795.0421616323115
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.007700431272483044,
        "median": 0.007700431272483044,
        "p90": 0.007700431272483044,
        "p95": 0.007700431272483044
      },
      "delta_norm": {
        "mean": 2.3465602791252196,
        "median": 2.3465602791252196,
        "p90": 2.3465602791252196,
        "p95": 2.3465602791252196
      },
      "lost_update_fraction": {
        "mean": 0.161833772510014,
        "median": 0.161833772510014,
        "p90": 0.161833772510014,
        "p95": 0.161833772510014
      },
      "represented_update_norm_ratio": {
        "mean": 0.8734437904435304,
        "median": 0.8734437904435304,
        "p90": 0.8734437904435304,
        "p95": 0.8734437904435304
      },
      "residual_update_cosine": {
        "mean": -0.013109719856619977,
        "median": -0.013109719856619977,
        "p90": -0.013109719856619977,
        "p95": -0.013109719856619977
      },
      "residual_update_projection_coeff": {
        "mean": -0.0010023612751435028,
        "median": -0.0010023612751435028,
        "p90": -0.0010023612751435028,
        "p95": -0.0010023612751435028
      },
      "saturation_fraction": {
        "mean": 0.03149063264330227,
        "median": 0.03149063264330227,
        "p90": 0.03149063264330227,
        "p95": 0.03149063264330227
      },
      "scale_max": {
        "mean": 0.029640536893869755,
        "median": 0.029640536893869755,
        "p90": 0.029640536893869755,
        "p95": 0.029640536893869755
      },
      "scale_mean": {
        "mean": 0.0003912860424796855,
        "median": 0.0003912860424796855,
        "p90": 0.0003912860424796855,
        "p95": 0.0003912860424796855
      },
      "scale_median": {
        "mean": 9.58543628090854e-05,
        "median": 9.58543628090854e-05,
        "p90": 9.58543628090854e-05,
        "p95": 9.58543628090854e-05
      },
      "scale_p95_over_median": {
        "mean": 18.291631183038493,
        "median": 18.291631183038493,
        "p90": 18.291631183038493,
        "p95": 18.291631183038493
      },
      "state_change_error_E_Delta": {
        "mean": 0.4253472996327374,
        "median": 0.4253472996327374,
        "p90": 0.4253472996327374,
        "p95": 0.4253472996327374
      },
      "state_norm": {
        "mean": 19.883701129312,
        "median": 19.883701129312,
        "p90": 19.883701129312,
        "p95": 19.883701129312
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.005662450289670114,
        "median": 0.005662450289670114,
        "p90": 0.005662450289670114,
        "p95": 0.005662450289670114
      },
      "survival_fraction": {
        "mean": 0.5550650159517926,
        "median": 0.5550650159517926,
        "p90": 0.5550650159517926,
        "p95": 0.5550650159517926
      },
      "top1_agreement": {
        "mean": 0.98046875,
        "median": 0.98046875,
        "p90": 0.98046875,
        "p95": 0.98046875
      },
      "update_cosine": {
        "mean": 0.8902008017757906,
        "median": 0.8902008017757906,
        "p90": 0.8902008017757906,
        "p95": 0.8902008017757906
      },
      "update_distortion_ratio": {
        "mean": 0.20133116319965996,
        "median": 0.20133116319965996,
        "p90": 0.20133116319965996,
        "p95": 0.20133116319965996
      },
      "within_group_dynamic_range_p95": {
        "mean": 1084.156205178259,
        "median": 1084.156205178259,
        "p90": 1084.156205178259,
        "p95": 1084.156205178259
      }
    }
  },
  "R64": {
    "1-128": {
      "KL": {
        "mean": 0.004594278225245417,
        "median": 0.004594278225245417,
        "p90": 0.004594278225245417,
        "p95": 0.004594278225245417
      },
      "delta_norm": {
        "mean": 2.9133562950859444,
        "median": 2.9133562950859444,
        "p90": 2.9133562950859444,
        "p95": 2.9133562950859444
      },
      "lost_update_fraction": {
        "mean": 0.13799867668861177,
        "median": 0.13799867668861177,
        "p90": 0.13799867668861177,
        "p95": 0.13799867668861177
      },
      "represented_update_norm_ratio": {
        "mean": 0.8881387744745796,
        "median": 0.8881387744745796,
        "p90": 0.8881387744745796,
        "p95": 0.8881387744745796
      },
      "residual_update_cosine": {
        "mean": -0.013386206511869699,
        "median": -0.013386206511869699,
        "p90": -0.013386206511869699,
        "p95": -0.013386206511869699
      },
      "residual_update_projection_coeff": {
        "mean": -0.0010893808863936987,
        "median": -0.0010893808863936987,
        "p90": -0.0010893808863936987,
        "p95": -0.0010893808863936987
      },
      "saturation_fraction": {
        "mean": 0.015739750048619865,
        "median": 0.015739750048619865,
        "p90": 0.015739750048619865,
        "p95": 0.015739750048619865
      },
      "scale_max": {
        "mean": 0.026560712283204644,
        "median": 0.026560712283204644,
        "p90": 0.026560712283204644,
        "p95": 0.026560712283204644
      },
      "scale_mean": {
        "mean": 0.0005874421234277689,
        "median": 0.0005874421234277689,
        "p90": 0.0005874421234277689,
        "p95": 0.0005874421234277689
      },
      "scale_median": {
        "mean": 0.00014363471672916613,
        "median": 0.00014363471672916613,
        "p90": 0.00014363471672916613,
        "p95": 0.00014363471672916613
      },
      "scale_p95_over_median": {
        "mean": 14.150153264959467,
        "median": 14.150153264959467,
        "p90": 14.150153264959467,
        "p95": 14.150153264959467
      },
      "state_change_error_E_Delta": {
        "mean": 0.3995299837834897,
        "median": 0.3995299837834897,
        "p90": 0.3995299837834897,
        "p95": 0.3995299837834897
      },
      "state_norm": {
        "mean": 18.008421923231886,
        "median": 18.008421923231886,
        "p90": 18.008421923231886,
        "p95": 18.008421923231886
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.008782678258673697,
        "median": 0.008782678258673697,
        "p90": 0.008782678258673697,
        "p95": 0.008782678258673697
      },
      "survival_fraction": {
        "mean": 0.590912124303382,
        "median": 0.590912124303382,
        "p90": 0.590912124303382,
        "p95": 0.590912124303382
      },
      "top1_agreement": {
        "mean": 0.984375,
        "median": 0.984375,
        "p90": 0.984375,
        "p95": 0.984375
      },
      "update_cosine": {
        "mean": 0.9037183437721309,
        "median": 0.9037183437721309,
        "p90": 0.9037183437721309,
        "p95": 0.9037183437721309
      },
      "update_distortion_ratio": {
        "mean": 0.178738764764658,
        "median": 0.178738764764658,
        "p90": 0.178738764764658,
        "p95": 0.178738764764658
      },
      "within_group_dynamic_range_p95": {
        "mean": 1719.2949354353052,
        "median": 1719.2949354353052,
        "p90": 1719.2949354353052,
        "p95": 1719.2949354353052
      }
    },
    "129-256": {
      "KL": {
        "mean": 0.017937367968783934,
        "median": 0.017937367968783934,
        "p90": 0.017937367968783934,
        "p95": 0.017937367968783934
      },
      "delta_norm": {
        "mean": 2.817156061282738,
        "median": 2.817156061282738,
        "p90": 2.817156061282738,
        "p95": 2.817156061282738
      },
      "lost_update_fraction": {
        "mean": 0.1476656004108094,
        "median": 0.1476656004108094,
        "p90": 0.1476656004108094,
        "p95": 0.1476656004108094
      },
      "represented_update_norm_ratio": {
        "mean": 0.8819345787213344,
        "median": 0.8819345787213344,
        "p90": 0.8819345787213344,
        "p95": 0.8819345787213344
      },
      "residual_update_cosine": {
        "mean": -0.014816518409550856,
        "median": -0.014816518409550856,
        "p90": -0.014816518409550856,
        "p95": -0.014816518409550856
      },
      "residual_update_projection_coeff": {
        "mean": -0.001247959147619756,
        "median": -0.001247959147619756,
        "p90": -0.001247959147619756,
        "p95": -0.001247959147619756
      },
      "saturation_fraction": {
        "mean": 0.01573817059397697,
        "median": 0.01573817059397697,
        "p90": 0.01573817059397697,
        "p95": 0.01573817059397697
      },
      "scale_max": {
        "mean": 0.02854399662714968,
        "median": 0.02854399662714968,
        "p90": 0.02854399662714968,
        "p95": 0.02854399662714968
      },
      "scale_mean": {
        "mean": 0.0006346893198643608,
        "median": 0.0006346893198643608,
        "p90": 0.0006346893198643608,
        "p95": 0.0006346893198643608
      },
      "scale_median": {
        "mean": 0.00014740925035579738,
        "median": 0.00014740925035579738,
        "p90": 0.00014740925035579738,
        "p95": 0.00014740925035579738
      },
      "scale_p95_over_median": {
        "mean": 14.843489920948429,
        "median": 14.843489920948429,
        "p90": 14.843489920948429,
        "p95": 14.843489920948429
      },
      "state_change_error_E_Delta": {
        "mean": 0.4117315121966744,
        "median": 0.4117315121966744,
        "p90": 0.4117315121966744,
        "p95": 0.4117315121966744
      },
      "state_norm": {
        "mean": 19.526885460441314,
        "median": 19.526885460441314,
        "p90": 19.526885460441314,
        "p95": 19.526885460441314
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.008228437901461138,
        "median": 0.008228437901461138,
        "p90": 0.008228437901461138,
        "p95": 0.008228437901461138
      },
      "survival_fraction": {
        "mean": 0.55413678723077,
        "median": 0.55413678723077,
        "p90": 0.55413678723077,
        "p95": 0.55413678723077
      },
      "top1_agreement": {
        "mean": 0.96875,
        "median": 0.96875,
        "p90": 0.96875,
        "p95": 0.96875
      },
      "update_cosine": {
        "mean": 0.897291416108298,
        "median": 0.897291416108298,
        "p90": 0.897291416108298,
        "p95": 0.897291416108298
      },
      "update_distortion_ratio": {
        "mean": 0.18959530495378307,
        "median": 0.18959530495378307,
        "p90": 0.18959530495378307,
        "p95": 0.18959530495378307
      },
      "within_group_dynamic_range_p95": {
        "mean": 1965.277270066582,
        "median": 1965.277270066582,
        "p90": 1965.277270066582,
        "p95": 1965.277270066582
      }
    },
    "257-512": {
      "KL": {
        "mean": 0.02062880626660734,
        "median": 0.02062880626660734,
        "p90": 0.02062880626660734,
        "p95": 0.02062880626660734
      },
      "delta_norm": {
        "mean": 2.3523307323619105,
        "median": 2.3523307323619105,
        "p90": 2.3523307323619105,
        "p95": 2.3523307323619105
      },
      "lost_update_fraction": {
        "mean": 0.14504922397100437,
        "median": 0.14504922397100437,
        "p90": 0.14504922397100437,
        "p95": 0.14504922397100437
      },
      "represented_update_norm_ratio": {
        "mean": 0.8874044577051107,
        "median": 0.8874044577051107,
        "p90": 0.8874044577051107,
        "p95": 0.8874044577051107
      },
      "residual_update_cosine": {
        "mean": -0.018052157070722313,
        "median": -0.018052157070722313,
        "p90": -0.018052157070722313,
        "p95": -0.018052157070722313
      },
      "residual_update_projection_coeff": {
        "mean": -0.0017546099373583082,
        "median": -0.0017546099373583082,
        "p90": -0.0017546099373583082,
        "p95": -0.0017546099373583082
      },
      "saturation_fraction": {
        "mean": 0.015738921239972108,
        "median": 0.015738921239972108,
        "p90": 0.015738921239972108,
        "p95": 0.015738921239972108
      },
      "scale_max": {
        "mean": 0.029604672274217595,
        "median": 0.029604672274217595,
        "p90": 0.029604672274217595,
        "p95": 0.029604672274217595
      },
      "scale_mean": {
        "mean": 0.0006467432886588733,
        "median": 0.0006467432886588733,
        "p90": 0.0006467432886588733,
        "p95": 0.0006467432886588733
      },
      "scale_median": {
        "mean": 0.00014905438780806374,
        "median": 0.00014905438780806374,
        "p90": 0.00014905438780806374,
        "p95": 0.00014905438780806374
      },
      "scale_p95_over_median": {
        "mean": 14.63982771914594,
        "median": 14.63982771914594,
        "p90": 14.63982771914594,
        "p95": 14.63982771914594
      },
      "state_change_error_E_Delta": {
        "mean": 0.40443073606588986,
        "median": 0.40443073606588986,
        "p90": 0.40443073606588986,
        "p95": 0.40443073606588986
      },
      "state_norm": {
        "mean": 19.89401603063259,
        "median": 19.89401603063259,
        "p90": 19.89401603063259,
        "p95": 19.89401603063259
      },
      "state_reconstruction_relative_error_E_S": {
        "mean": 0.007439583408879956,
        "median": 0.007439583408879956,
        "p90": 0.007439583408879956,
        "p95": 0.007439583408879956
      },
      "survival_fraction": {
        "mean": 0.49771310916791367,
        "median": 0.49771310916791367,
        "p90": 0.49771310916791367,
        "p95": 0.49771310916791367
      },
      "top1_agreement": {
        "mean": 0.9609375,
        "median": 0.9609375,
        "p90": 0.9609375,
        "p95": 0.9609375
      },
      "update_cosine": {
        "mean": 0.8999806908492854,
        "median": 0.8999806908492854,
        "p90": 0.8999806908492854,
        "p95": 0.8999806908492854
      },
      "update_distortion_ratio": {
        "mean": 0.1843997885525871,
        "median": 0.1843997885525871,
        "p90": 0.1843997885525871,
        "p95": 0.1843997885525871
      },
      "within_group_dynamic_range_p95": {
        "mean": 2870.2561570460243,
        "median": 2870.2561570460243,
        "p90": 2870.2561570460243,
        "p95": 2870.2561570460243
      }
    }
  }
}
```

## Table 8: Snapshot Rescue vs State-Change Rescue
```json
{
  "snapshot": {
    "improved_prompt_count": 1,
    "per_prompt": {
      "test/algebra/1332.json": {
        "rescued_R128_to_16": true,
        "spearman_group_size_metric": 0.9999999999999998,
        "values": {
          "R128": 0.010535141899127196,
          "R16": 0.004509695226106723,
          "R32": 0.006058533607051571,
          "R64": 0.007970985406872515
        }
      }
    },
    "rate": 1.0,
    "total_prompt_count": 1
  },
  "state_change": {
    "lost_update_fraction": {
      "improved_prompt_count": 0,
      "per_prompt": {
        "test/algebra/1332.json": {
          "rescued_R128_to_16": false,
          "spearman_group_size_metric": -0.9999999999999998,
          "values": {
            "R128": 0.12790876524815967,
            "R16": 0.17810101441753484,
            "R32": 0.16102050425419426,
            "R64": 0.14395230944934342
          }
        }
      },
      "rate": 0.0,
      "total_prompt_count": 1
    },
    "state_change_error_E_Delta": {
      "improved_prompt_count": 0,
      "per_prompt": {
        "test/algebra/1332.json": {
          "rescued_R128_to_16": false,
          "spearman_group_size_metric": -0.9999999999999998,
          "values": {
            "R128": 0.385343537327697,
            "R16": 0.44808847585188144,
            "R32": 0.4260235798065262,
            "R64": 0.40504150672122385
          }
        }
      },
      "rate": 0.0,
      "total_prompt_count": 1
    },
    "update_distortion_ratio": {
      "improved_prompt_count": 0,
      "per_prompt": {
        "test/algebra/1332.json": {
          "rescued_R128_to_16": false,
          "spearman_group_size_metric": -0.9999999999999998,
          "values": {
            "R128": 0.16819663661361292,
            "R16": 0.21969030151399557,
            "R32": 0.2013450292113844,
            "R64": 0.1842942622870021
          }
        }
      },
      "rate": 0.0,
      "total_prompt_count": 1
    }
  }
}
```

## Table 9: Layer/Head Strongest Rescue Locations
```json
{
  "lost_update_fraction": [
    {
      "R128": 0.6374921632866642,
      "R16": 0.2939271960666819,
      "head": 0,
      "layer": 13,
      "rescue_magnitude": 0.34356496721998236
    },
    {
      "R128": 0.5174961590507519,
      "R16": 0.21733061310369903,
      "head": 16,
      "layer": 4,
      "rescue_magnitude": 0.3001655459470528
    },
    {
      "R128": 0.7281869081439647,
      "R16": 0.440476863039132,
      "head": 9,
      "layer": 8,
      "rescue_magnitude": 0.28771004510483267
    },
    {
      "R128": 0.6277260015092316,
      "R16": 0.3518994537249774,
      "head": 22,
      "layer": 30,
      "rescue_magnitude": 0.27582654778425425
    },
    {
      "R128": 0.40171125574303657,
      "R16": 0.14039761898476769,
      "head": 26,
      "layer": 26,
      "rescue_magnitude": 0.2613136367582689
    },
    {
      "R128": 0.6232620275769161,
      "R16": 0.36526070621041423,
      "head": 9,
      "layer": 6,
      "rescue_magnitude": 0.25800132136650183
    },
    {
      "R128": 0.5299182484592586,
      "R16": 0.27558389697874036,
      "head": 27,
      "layer": 9,
      "rescue_magnitude": 0.25433435148051825
    },
    {
      "R128": 0.44104945671904455,
      "R16": 0.19502816127634892,
      "head": 14,
      "layer": 8,
      "rescue_magnitude": 0.24602129544269563
    },
    {
      "R128": 0.47356950026171196,
      "R16": 0.24696580086069586,
      "head": 23,
      "layer": 30,
      "rescue_magnitude": 0.2266036994010161
    },
    {
      "R128": 0.6803029811196071,
      "R16": 0.4569792614644485,
      "head": 20,
      "layer": 9,
      "rescue_magnitude": 0.22332371965515857
    },
    {
      "R128": 0.4718495756488904,
      "R16": 0.24999101579445684,
      "head": 18,
      "layer": 29,
      "rescue_magnitude": 0.22185855985443356
    },
    {
      "R128": 0.5497021040120882,
      "R16": 0.33408392093287526,
      "head": 31,
      "layer": 9,
      "rescue_magnitude": 0.21561818307921293
    },
    {
      "R128": 0.5316613035250922,
      "R16": 0.32101552495196584,
      "head": 23,
      "layer": 5,
      "rescue_magnitude": 0.21064577857312639
    },
    {
      "R128": 0.5913916422929991,
      "R16": 0.3819229026871292,
      "head": 16,
      "layer": 10,
      "rescue_magnitude": 0.20946873960586992
    },
    {
      "R128": 0.5880004629766461,
      "R16": 0.38028286528474003,
      "head": 21,
      "layer": 9,
      "rescue_magnitude": 0.20771759769190606
    }
  ],
  "state_change_error_E_Delta": [
    {
      "R128": 0.6801600535950911,
      "R16": 0.413239585973291,
      "head": 26,
      "layer": 26,
      "rescue_magnitude": 0.2669204676218001
    },
    {
      "R128": 0.752888631962979,
      "R16": 0.5027605101876971,
      "head": 16,
      "layer": 4,
      "rescue_magnitude": 0.2501281217752819
    },
    {
      "R128": 0.8427600070892052,
      "R16": 0.6057843425195573,
      "head": 0,
      "layer": 13,
      "rescue_magnitude": 0.23697566456964791
    },
    {
      "R128": 0.7617819136609648,
      "R16": 0.5318473582149135,
      "head": 23,
      "layer": 30,
      "rescue_magnitude": 0.22993455544605135
    },
    {
      "R128": 0.6784490161936108,
      "R16": 0.4624715110027317,
      "head": 20,
      "layer": 29,
      "rescue_magnitude": 0.21597750519087905
    },
    {
      "R128": 0.7192948828563429,
      "R16": 0.5089307953394169,
      "head": 14,
      "layer": 8,
      "rescue_magnitude": 0.21036408751692603
    },
    {
      "R128": 0.8636961815804923,
      "R16": 0.6571879531500706,
      "head": 22,
      "layer": 30,
      "rescue_magnitude": 0.20650822843042171
    },
    {
      "R128": 0.7889128267461155,
      "R16": 0.5833744465521866,
      "head": 31,
      "layer": 9,
      "rescue_magnitude": 0.2055383801939289
    },
    {
      "R128": 0.7767622049852247,
      "R16": 0.5772781801923361,
      "head": 27,
      "layer": 9,
      "rescue_magnitude": 0.19948402479288863
    },
    {
      "R128": 0.7441642413246379,
      "R16": 0.5472848745795119,
      "head": 18,
      "layer": 29,
      "rescue_magnitude": 0.19687936674512596
    },
    {
      "R128": 0.8314782146337987,
      "R16": 0.6405542107532033,
      "head": 9,
      "layer": 6,
      "rescue_magnitude": 0.19092400388059538
    },
    {
      "R128": 0.8721583472064534,
      "R16": 0.7014117174329341,
      "head": 9,
      "layer": 8,
      "rescue_magnitude": 0.1707466297735193
    },
    {
      "R128": 0.6422190587476564,
      "R16": 0.4722107776457283,
      "head": 12,
      "layer": 29,
      "rescue_magnitude": 0.17000828110192812
    },
    {
      "R128": 0.8374572785547386,
      "R16": 0.669147901510618,
      "head": 24,
      "layer": 6,
      "rescue_magnitude": 0.16830937704412063
    },
    {
      "R128": 0.7777285203896643,
      "R16": 0.6100361618714933,
      "head": 31,
      "layer": 8,
      "rescue_magnitude": 0.167692358518171
    }
  ],
  "update_cosine": [
    {
      "R128": 0.43639581544058664,
      "R16": 0.6999440533774239,
      "head": 9,
      "layer": 8,
      "rescue_magnitude": 0.2635482379368373
    },
    {
      "R128": 0.5597806913512093,
      "R16": 0.8071008324623108,
      "head": 0,
      "layer": 13,
      "rescue_magnitude": 0.24732014111110145
    },
    {
      "R128": 0.6454450786113739,
      "R16": 0.8630162392343793,
      "head": 16,
      "layer": 4,
      "rescue_magnitude": 0.2175711606230054
    },
    {
      "R128": 0.5409710279532841,
      "R16": 0.7437054472310203,
      "head": 22,
      "layer": 30,
      "rescue_magnitude": 0.20273441927773617
    },
    {
      "R128": 0.5627425611019135,
      "R16": 0.762964631829943,
      "head": 9,
      "layer": 6,
      "rescue_magnitude": 0.20022207072802956
    },
    {
      "R128": 0.3214980214834213,
      "R16": 0.5116433203220367,
      "head": 25,
      "layer": 14,
      "rescue_magnitude": 0.19014529883861542
    },
    {
      "R128": 0.7200783150536674,
      "R16": 0.9004750592367989,
      "head": 26,
      "layer": 26,
      "rescue_magnitude": 0.18039674418313156
    },
    {
      "R128": 0.6347053732190814,
      "R16": 0.8142336436680385,
      "head": 27,
      "layer": 9,
      "rescue_magnitude": 0.17952827044895714
    },
    {
      "R128": 0.5129676801817757,
      "R16": 0.6914199590682983,
      "head": 20,
      "layer": 9,
      "rescue_magnitude": 0.1784522788865226
    },
    {
      "R128": 0.6092314209256854,
      "R16": 0.7854688252721514,
      "head": 31,
      "layer": 9,
      "rescue_magnitude": 0.17623740434646606
    },
    {
      "R128": 0.45509079311575207,
      "R16": 0.6273076747144971,
      "head": 20,
      "layer": 16,
      "rescue_magnitude": 0.17221688159874504
    },
    {
      "R128": 0.5316519524369921,
      "R16": 0.6976107358932495,
      "head": 6,
      "layer": 8,
      "rescue_magnitude": 0.16595878345625736
    },
    {
      "R128": 0.5617181786469051,
      "R16": 0.7268055421965463,
      "head": 24,
      "layer": 6,
      "rescue_magnitude": 0.16508736354964115
    },
    {
      "R128": 0.6642435533659798,
      "R16": 0.828113683632442,
      "head": 23,
      "layer": 30,
      "rescue_magnitude": 0.16387013026646213
    },
    {
      "R128": 0.6009070064340319,
      "R16": 0.7631615826061794,
      "head": 18,
      "layer": 6,
      "rescue_magnitude": 0.16225457617214745
    }
  ],
  "update_distortion_ratio": [
    {
      "R128": 0.7180567491034863,
      "R16": 0.37560617619686176,
      "head": 0,
      "layer": 13,
      "rescue_magnitude": 0.3424505729066245
    },
    {
      "R128": 0.5875349897564895,
      "R16": 0.2608437515591781,
      "head": 16,
      "layer": 4,
      "rescue_magnitude": 0.32669123819731144
    },
    {
      "R128": 0.4952999255120337,
      "R16": 0.19849100147270704,
      "head": 26,
      "layer": 26,
      "rescue_magnitude": 0.2968089240393267
    },
    {
      "R128": 0.7693071506110861,
      "R16": 0.4753404689289124,
      "head": 22,
      "layer": 30,
      "rescue_magnitude": 0.2939666816821737
    },
    {
      "R128": 0.600711124834849,
      "R16": 0.31498161566703475,
      "head": 23,
      "layer": 30,
      "rescue_magnitude": 0.28572950916781426
    },
    {
      "R128": 0.6230100642005167,
      "R16": 0.34839798178244435,
      "head": 27,
      "layer": 9,
      "rescue_magnitude": 0.2746120824180724
    },
    {
      "R128": 0.7006789565262352,
      "R16": 0.42715832953726796,
      "head": 9,
      "layer": 6,
      "rescue_magnitude": 0.2735206269889673
    },
    {
      "R128": 0.7830596916725535,
      "R16": 0.5146803142718561,
      "head": 9,
      "layer": 8,
      "rescue_magnitude": 0.2683793774006974
    },
    {
      "R128": 0.634681369491806,
      "R16": 0.36925597418577966,
      "head": 31,
      "layer": 9,
      "rescue_magnitude": 0.2654253953060263
    },
    {
      "R128": 0.5230860327088654,
      "R16": 0.2663682951131046,
      "head": 14,
      "layer": 8,
      "rescue_magnitude": 0.25671773759576083
    },
    {
      "R128": 0.5602862004533329,
      "R16": 0.3076371017248817,
      "head": 18,
      "layer": 29,
      "rescue_magnitude": 0.2526490987284512
    },
    {
      "R128": 0.7637783269843992,
      "R16": 0.5216270642974955,
      "head": 20,
      "layer": 9,
      "rescue_magnitude": 0.24215126268690368
    },
    {
      "R128": 0.7067805844372851,
      "R16": 0.46770546677517727,
      "head": 24,
      "layer": 6,
      "rescue_magnitude": 0.23907511766210782
    },
    {
      "R128": 0.4905795853872344,
      "R16": 0.25192078945782737,
      "head": 20,
      "layer": 29,
      "rescue_magnitude": 0.23865879592940703
    },
    {
      "R128": 0.6173502894866115,
      "R16": 0.3837198184809648,
      "head": 23,
      "layer": 5,
      "rescue_magnitude": 0.2336304710056467
    }
  ],
  "within_group_dynamic_range_p95": [
    {
      "R128": 3913754.8497767793,
      "R16": 253127.5801199772,
      "head": 11,
      "layer": 0,
      "rescue_magnitude": 3660627.269656802
    },
    {
      "R128": 396879.18275669543,
      "R16": 23172.522181919543,
      "head": 23,
      "layer": 0,
      "rescue_magnitude": 373706.6605747759
    },
    {
      "R128": 383780.0968191955,
      "R16": 27212.541706194115,
      "head": 22,
      "layer": 0,
      "rescue_magnitude": 356567.5551130014
    },
    {
      "R128": 368217.0320870533,
      "R16": 33295.53810337601,
      "head": 19,
      "layer": 0,
      "rescue_magnitude": 334921.4939836773
    },
    {
      "R128": 353018.0036830356,
      "R16": 23471.412538364882,
      "head": 29,
      "layer": 0,
      "rescue_magnitude": 329546.59114467073
    },
    {
      "R128": 343466.6425223207,
      "R16": 25521.24883510033,
      "head": 10,
      "layer": 0,
      "rescue_magnitude": 317945.39368722035
    },
    {
      "R128": 299668.7442243303,
      "R16": 19478.05029645641,
      "head": 18,
      "layer": 0,
      "rescue_magnitude": 280190.69392787386
    },
    {
      "R128": 254535.97366071414,
      "R16": 19619.79234095974,
      "head": 27,
      "layer": 0,
      "rescue_magnitude": 234916.1813197544
    },
    {
      "R128": 239819.5826729907,
      "R16": 13466.284692382726,
      "head": 5,
      "layer": 0,
      "rescue_magnitude": 226353.29798060798
    },
    {
      "R128": 220980.81456473196,
      "R16": 15766.243310546794,
      "head": 26,
      "layer": 0,
      "rescue_magnitude": 205214.57125418517
    },
    {
      "R128": 161312.2994280132,
      "R16": 13313.00945870532,
      "head": 9,
      "layer": 0,
      "rescue_magnitude": 147999.2899693079
    },
    {
      "R128": 138104.96263950862,
      "R16": 10454.800735909555,
      "head": 28,
      "layer": 0,
      "rescue_magnitude": 127650.16190359907
    },
    {
      "R128": 101723.76403459812,
      "R16": 8029.367177036789,
      "head": 8,
      "layer": 0,
      "rescue_magnitude": 93694.39685756133
    },
    {
      "R128": 94920.86718749988,
      "R16": 6387.937393624407,
      "head": 3,
      "layer": 0,
      "rescue_magnitude": 88532.92979387548
    },
    {
      "R128": 80499.42299107133,
      "R16": 4928.439099121062,
      "head": 2,
      "layer": 0,
      "rescue_magnitude": 75570.98389195027
    }
  ]
}
```

## Table 10: Final Causal Classification
```json
{
  "AXIS_GEOMETRY_CAUSAL_SUPPORT": "NO",
  "FINAL_CLASSIFICATION": "AXIS_GEOMETRY_CAUSAL_HYPOTHESIS_NOT_SUPPORTED",
  "METHOD_DESIGN_READY": "NO",
  "METHOD_DESIGN_READY_CANDIDATE": "NO",
  "table": {
    "AXIS_GEOMETRY_CAUSAL_SUPPORT": "NO",
    "MATCHED_GROUP_ROW_COLUMN_GAP_SHRINKS": "YES",
    "RESCUE_IS_CROSS_PROMPT_CONSISTENT": "NO",
    "ROW_GROUP_REDUCTION_LOWERS_DYNAMIC_RANGE": "YES",
    "ROW_GROUP_REDUCTION_RESCUES_LOGIT_FIDELITY": "YES",
    "ROW_GROUP_REDUCTION_RESCUES_STATE_CHANGE_PRESERVATION": "NO",
    "ROW_RESCUE_SHOWS_DOSE_RESPONSE": "NO",
    "STATE_CHANGE_RESCUE_EXCEEDS_SNAPSHOT_RESCUE": "NO"
  }
}
```

## Limitations
- R128->R16 changes both group size and scale count; matched R/C at the same group size controls scale count and isolates orientation/membership geometry more closely.
- These quantizers are CAUSAL / MECHANISM DIAGNOSTIC CONTROLS, not a proposed method.
