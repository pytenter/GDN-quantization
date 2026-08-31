# GDN INT8 Residual Direction Sensitivity Panel V1

## Gates
- FORMAL_STATUS: `COMPLETE`
- PROTOCOL_GATE: `PASS`
- PANEL_IMPLEMENTATION_GATE: `PASS`
- NORM_MATCH_GATE: `PASS`; ANGLE_GATE: `PASS`
- DIRECTION_DIVERSITY_GATE: `PASS`
- PILOT_CLASSIFICATION: `N/A`
- FINAL_CLASSIFICATION: `RECURRENT_DIRECTIONAL_SENSITIVITY_DISTRIBUTION_SUPPORTED`
- PROPAGATION_FIDELITY_LINK: `INCONCLUSIVE`
- NATURAL_RESIDUAL_DYNAMIC_ALIGNMENT: `LOW_GAIN_BIASED`
- MECHANISM_CLOSURE_CANDIDATE: `NO`
- METHOD_DESIGN_READY_CANDIDATE: `NO`
- METHOD_DESIGN_READY: `NO`

## Analysis
```json
{
  "aggregate_by_config": {
    "FP_STATE": {
      "mean_future_KL_score": 0.0,
      "mean_max_future_KL": 0.0,
      "mean_max_propagation_gain": 0.0,
      "mean_persistence_score": 0.0
    },
    "ORTHOGONAL_REALDIR_PULSE": {
      "mean_future_KL_score": 0.0009317822221343938,
      "mean_max_future_KL": 0.003595623867163845,
      "mean_max_propagation_gain": 0.9917711841852799,
      "mean_persistence_score": 0.8171134533147376
    },
    "ORTHO_RAND_00": {
      "mean_future_KL_score": 0.003824700552185502,
      "mean_max_future_KL": 0.5926015647350545,
      "mean_max_propagation_gain": 1.5572507505832973,
      "mean_persistence_score": 1.2397591011292661
    },
    "ORTHO_RAND_01": {
      "mean_future_KL_score": 0.0030655915271226583,
      "mean_max_future_KL": 0.04331495716340012,
      "mean_max_propagation_gain": 1.5287465021951916,
      "mean_persistence_score": 1.2397574729145762
    },
    "ORTHO_RAND_02": {
      "mean_future_KL_score": 0.003660423063321711,
      "mean_max_future_KL": 0.1131765757551572,
      "mean_max_propagation_gain": 1.5773564622462162,
      "mean_persistence_score": 1.2698200447943349
    },
    "ORTHO_RAND_03": {
      "mean_future_KL_score": 0.0017183483148375351,
      "mean_max_future_KL": 0.024589092726172466,
      "mean_max_propagation_gain": 1.4682853372154254,
      "mean_persistence_score": 1.2158767708970921
    },
    "ORTHO_RAND_04": {
      "mean_future_KL_score": 0.002309651760238119,
      "mean_max_future_KL": 0.40876948908731314,
      "mean_max_propagation_gain": 1.5532174131316456,
      "mean_persistence_score": 1.2344614976851374
    },
    "ORTHO_RAND_05": {
      "mean_future_KL_score": 0.0035790624677399974,
      "mean_max_future_KL": 0.3162360780422912,
      "mean_max_propagation_gain": 1.64274373519663,
      "mean_persistence_score": 1.3171664759864579
    },
    "ORTHO_RAND_06": {
      "mean_future_KL_score": 0.0023832871826228266,
      "mean_max_future_KL": 0.15163992645749305,
      "mean_max_propagation_gain": 1.472258369369737,
      "mean_persistence_score": 1.1585797024202613
    },
    "ORTHO_RAND_07": {
      "mean_future_KL_score": 0.0018902483350448597,
      "mean_max_future_KL": 0.044175445622790396,
      "mean_max_propagation_gain": 1.4939907070755352,
      "mean_persistence_score": 1.2124636282841814
    },
    "ORTHO_RAND_08": {
      "mean_future_KL_score": 0.002113616552707867,
      "mean_max_future_KL": 0.3020689349925508,
      "mean_max_propagation_gain": 1.39583695615056,
      "mean_persistence_score": 1.1628932697540095
    },
    "ORTHO_RAND_09": {
      "mean_future_KL_score": 0.0034778649521477844,
      "mean_max_future_KL": 0.1672347257521728,
      "mean_max_propagation_gain": 1.7392785189875177,
      "mean_persistence_score": 1.3206503598854
    },
    "ORTHO_RAND_10": {
      "mean_future_KL_score": 0.00330795642248429,
      "mean_max_future_KL": 0.05685971490392047,
      "mean_max_propagation_gain": 1.5782271075519796,
      "mean_persistence_score": 1.2636553140188669
    },
    "ORTHO_RAND_11": {
      "mean_future_KL_score": 0.003656608535401335,
      "mean_max_future_KL": 0.02634613515692763,
      "mean_max_propagation_gain": 1.5190323344452814,
      "mean_persistence_score": 1.2448914851534378
    },
    "ORTHO_RAND_12": {
      "mean_future_KL_score": 0.002164221124158079,
      "mean_max_future_KL": 0.08231000874366146,
      "mean_max_propagation_gain": 1.3997232420679093,
      "mean_persistence_score": 1.1597314732066453
    },
    "ORTHO_RAND_13": {
      "mean_future_KL_score": 0.018866932080878886,
      "mean_max_future_KL": 0.4766485862556793,
      "mean_max_propagation_gain": 1.6146561521449563,
      "mean_persistence_score": 1.2476466382541929
    },
    "ORTHO_RAND_14": {
      "mean_future_KL_score": 0.002554183821884681,
      "mean_max_future_KL": 0.11549164223510565,
      "mean_max_propagation_gain": 1.5258903251991756,
      "mean_persistence_score": 1.22327823428073
    },
    "ORTHO_RAND_15": {
      "mean_future_KL_score": 0.0021284578107165146,
      "mean_max_future_KL": 0.025640323978020914,
      "mean_max_propagation_gain": 1.4135939178089103,
      "mean_persistence_score": 1.1619770703114576
    },
    "PARALLEL_PLUS_PULSE": {
      "mean_future_KL_score": 0.00014220601686061346,
      "mean_max_future_KL": 0.000953943334025098,
      "mean_max_propagation_gain": 0.9610830094536948,
      "mean_persistence_score": 0.6362780354297021
    },
    "REAL_R128_PULSE": {
      "mean_future_KL_score": 0.0011327364248182024,
      "mean_max_future_KL": 0.0042057513617490055,
      "mean_max_propagation_gain": 0.9916334175050034,
      "mean_persistence_score": 0.8170382409068151
    }
  },
  "clear_future_KL_separation_count": 15,
  "clear_persistence_separation_count": 16,
  "median_future_KL_spread": 0.005726796960341762,
  "median_persistence_spread": 0.4620849972970387,
  "median_random_amplification_fraction": 1.0,
  "median_random_persistent_amplification_fraction": 1.0,
  "median_rho_persistence_future_KL": 0.19117647058823473,
  "natural_realdir_position_summary": {
    "future_KL": {
      "count_bottom_quartile": 11,
      "count_middle_50": 6,
      "count_top_quartile": 1,
      "median_percentile": 0.0625,
      "note": "N=16 random directions per unit; percentile is coarse descriptive positioning only."
    },
    "max_gain": {
      "count_bottom_quartile": 17,
      "count_middle_50": 1,
      "count_top_quartile": 0,
      "median_percentile": 0.0,
      "note": "N=16 random directions per unit; percentile is coarse descriptive positioning only."
    },
    "persistence": {
      "count_bottom_quartile": 18,
      "count_middle_50": 0,
      "count_top_quartile": 0,
      "median_percentile": 0.0,
      "note": "N=16 random directions per unit; percentile is coarse descriptive positioning only."
    }
  },
  "per_unit": [
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "7/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.3125,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "6/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.1559503868500014,
        "max": 0.0017622315724125314,
        "mean": 0.00036891088368361523,
        "median": 0.0002526210711882193,
        "min": 3.695142625801395e-05,
        "p10": 4.602196621146248e-05,
        "p25": 7.625912991606754e-05,
        "p75": 0.00043126088891725094,
        "p90": 0.0007493573700429578,
        "std": 0.0004264426798632013
      },
      "future_KL_spread": 0.0017252801461545174,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.27968394782767,
        "max": 0.12053059786558151,
        "mean": 0.026953515869536204,
        "median": 0.010164111386984587,
        "min": 0.0006543867639265954,
        "p10": 0.000890892930328846,
        "p25": 0.0028727243770845234,
        "p75": 0.030024335719645023,
        "p90": 0.07817502692341805,
        "std": 0.034491981597043524
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.1856941044826029,
        "max": 2.212794527948698,
        "mean": 1.3706003311766024,
        "median": 1.316383775332811,
        "min": 1.1282868480430819,
        "p10": 1.1566554987733875,
        "p25": 1.2309102946670352,
        "p75": 1.3816165305820423,
        "p90": 1.5923045797521076,
        "std": 0.25451240110158385
      },
      "max_gain_spread": 1.084507679905616,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.00014915694234787224,
          "max_future_KL": 0.0006050142110325396,
          "max_propagation_gain": 0.9922102300415815,
          "mean_G_state": 0.88560559681479,
          "mean_KL_future": 0.00015576534604511139,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8261739924203976,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 3.695142625801395e-05,
          "max_future_KL": 0.010125797241926193,
          "max_propagation_gain": 1.3573860744345485,
          "mean_G_state": 1.1348705276853561,
          "mean_KL_future": 0.00135392139545365,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1912387106713638,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.000391895829451272,
          "max_future_KL": 0.01013909000903368,
          "max_propagation_gain": 1.2779061368794087,
          "mean_G_state": 1.1440497924729933,
          "mean_KL_future": 0.0015778847553682418,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1087417221981486,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0003272269044011011,
          "max_future_KL": 0.0010894077131524682,
          "max_propagation_gain": 1.243620634597579,
          "mean_G_state": 1.083523510062707,
          "mean_KL_future": 0.0002783153730150367,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.059345556885995,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 5.174789108046696e-05,
          "max_future_KL": 0.011669671162962914,
          "max_propagation_gain": 1.192779274875403,
          "mean_G_state": 1.07309905092213,
          "mean_KL_future": 0.0016901379193574595,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0530294300331753,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.00020590066076255908,
          "max_future_KL": 0.04662877693772316,
          "max_propagation_gain": 1.306876376098014,
          "mean_G_state": 1.1652005370168774,
          "mean_KL_future": 0.006042198781571639,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.117174467617413,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0006671677839541701,
          "max_future_KL": 0.024489521980285645,
          "max_propagation_gain": 1.609843728543543,
          "mean_G_state": 1.3771877124776446,
          "mean_KL_future": 0.003836022171489706,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3339884936095696,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 8.11120080484784e-05,
          "max_future_KL": 0.0006923781475052238,
          "max_propagation_gain": 1.374589174419625,
          "mean_G_state": 1.1816124220803068,
          "mean_KL_future": 0.00020016348952456298,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2450323312951206,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.00015375055719673013,
          "max_future_KL": 0.07813093066215515,
          "max_propagation_gain": 1.4026985990692935,
          "mean_G_state": 1.2147599780766496,
          "mean_KL_future": 0.010029371327712422,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2196246463650706,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0005493560673151876,
          "max_future_KL": 0.0026959062088280916,
          "max_propagation_gain": 1.3539943183765624,
          "mean_G_state": 1.164344097209558,
          "mean_KL_future": 0.0006412718420860664,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1889391837292025,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 4.029604134245801e-05,
          "max_future_KL": 0.002931663766503334,
          "max_propagation_gain": 1.5747654309606725,
          "mean_G_state": 1.2631599132696616,
          "mean_KL_future": 0.0004680407960844235,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3863440343294715,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0008315469561317457,
          "max_future_KL": 0.12053059786558151,
          "max_propagation_gain": 1.3258911745676076,
          "mean_G_state": 1.1265882064641803,
          "mean_KL_future": 0.016669159545732892,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.13312058479816,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0017622315724125314,
          "max_future_KL": 0.008640468120574951,
          "max_propagation_gain": 2.212794527948698,
          "mean_G_state": 1.5250850841567862,
          "mean_KL_future": 0.0018591588968215333,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.7849479796152274,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0003022715588897995,
          "max_future_KL": 0.010189132764935493,
          "max_propagation_gain": 1.1748839274302627,
          "mean_G_state": 1.0412593808671067,
          "mean_KL_future": 0.0014646426023121606,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9983723595327441,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.00029934148161387953,
          "max_future_KL": 0.07821912318468094,
          "max_propagation_gain": 1.1282868480430819,
          "mean_G_state": 1.0409535452089405,
          "mean_KL_future": 0.00996536620067312,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0156448688110578,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 6.170049551883494e-05,
          "max_future_KL": 0.024429401382803917,
          "max_propagation_gain": 1.1384270701165122,
          "mean_G_state": 1.0556883497826617,
          "mean_KL_future": 0.0033059143258142853,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.025786364652723,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0001400769045606154,
          "max_future_KL": 0.0006543867639265954,
          "max_propagation_gain": 1.2548620024648267,
          "mean_G_state": 1.1008659368156155,
          "mean_KL_future": 0.00017103353527958554,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.112310715324735,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 2.4685581552930102e-05,
          "max_future_KL": 0.00012195564340800047,
          "max_propagation_gain": 0.9581869021034912,
          "mean_G_state": 0.7123213275976246,
          "mean_KL_future": 1.7496274185679894e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.5738063861984999,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.00013045991168354477,
          "max_future_KL": 0.0015356484800577164,
          "max_propagation_gain": 0.992428238791364,
          "mean_G_state": 0.8875190226097004,
          "mean_KL_future": 0.00034468573406609693,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8291963470601997,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.15888188688131988,
        "max": 1.7849479796152274,
        "mean": 1.1858525905918238,
        "median": 1.1251475262077864,
        "min": 0.9983723595327441,
        "p10": 1.0207156167318905,
        "p25": 1.0577665251727901,
        "p75": 1.2259765675975831,
        "p90": 1.3601662639695205,
        "std": 0.18841049715648917
      },
      "persistence_separation_ratio": 1.7878579695944576,
      "persistence_spread": 0.7865756200824834,
      "problem_id": "test/algebra/1332.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0003022715588897995,
          "max_future_KL": 0.010189132764935493,
          "max_propagation_gain": 1.1748839274302627,
          "persistence_score": 0.9983723595327441
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.00029934148161387953,
          "max_future_KL": 0.07821912318468094,
          "max_propagation_gain": 1.1282868480430819,
          "persistence_score": 1.0156448688110578
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 6.170049551883494e-05,
          "max_future_KL": 0.024429401382803917,
          "max_propagation_gain": 1.1384270701165122,
          "persistence_score": 1.025786364652723
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 5.174789108046696e-05,
          "max_future_KL": 0.011669671162962914,
          "max_propagation_gain": 1.192779274875403,
          "persistence_score": 1.0530294300331753
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0003272269044011011,
          "max_future_KL": 0.0010894077131524682,
          "max_propagation_gain": 1.243620634597579,
          "persistence_score": 1.059345556885995
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.000391895829451272,
          "max_future_KL": 0.01013909000903368,
          "max_propagation_gain": 1.2779061368794087,
          "persistence_score": 1.1087417221981486
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0001400769045606154,
          "max_future_KL": 0.0006543867639265954,
          "max_propagation_gain": 1.2548620024648267,
          "persistence_score": 1.112310715324735
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.00020590066076255908,
          "max_future_KL": 0.04662877693772316,
          "max_propagation_gain": 1.306876376098014,
          "persistence_score": 1.117174467617413
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0008315469561317457,
          "max_future_KL": 0.12053059786558151,
          "max_propagation_gain": 1.3258911745676076,
          "persistence_score": 1.13312058479816
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0005493560673151876,
          "max_future_KL": 0.0026959062088280916,
          "max_propagation_gain": 1.3539943183765624,
          "persistence_score": 1.1889391837292025
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 3.695142625801395e-05,
          "max_future_KL": 0.010125797241926193,
          "max_propagation_gain": 1.3573860744345485,
          "persistence_score": 1.1912387106713638
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.00015375055719673013,
          "max_future_KL": 0.07813093066215515,
          "max_propagation_gain": 1.4026985990692935,
          "persistence_score": 1.2196246463650706
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 8.11120080484784e-05,
          "max_future_KL": 0.0006923781475052238,
          "max_propagation_gain": 1.374589174419625,
          "persistence_score": 1.2450323312951206
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0006671677839541701,
          "max_future_KL": 0.024489521980285645,
          "max_propagation_gain": 1.609843728543543,
          "persistence_score": 1.3339884936095696
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 4.029604134245801e-05,
          "max_future_KL": 0.002931663766503334,
          "max_propagation_gain": 1.5747654309606725,
          "persistence_score": 1.3863440343294715
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0017622315724125314,
          "max_future_KL": 0.008640468120574951,
          "max_propagation_gain": 2.212794527948698,
          "persistence_score": 1.7849479796152274
        }
      ],
      "random_persistent_amplification_fraction": 0.9375,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.18529411764705828,
      "spearman_persistence_future_KL": 0.10588235294117615,
      "t0": 64,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.125,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "3/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": false,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.8486094263566142,
        "max": 0.000309669329830875,
        "mean": 7.96666226343834e-05,
        "median": 6.664136358196515e-05,
        "min": 1.39346858816225e-05,
        "p10": 2.1020799631976673e-05,
        "p25": 4.6538277088470354e-05,
        "p75": 8.886003517076803e-05,
        "p90": 0.00011902937693299976,
        "std": 6.760584778214238e-05
      },
      "future_KL_spread": 0.0002957346439492525,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.746440948432887,
        "max": 0.06238464266061783,
        "mean": 0.00886644810452708,
        "median": 0.0020744826178997755,
        "min": 0.0002364303800277412,
        "p10": 0.0007268158660735935,
        "p25": 0.0010319197463104501,
        "p75": 0.0076185776852071285,
        "p90": 0.022194736637175083,
        "std": 0.01548472803864769
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.12274039311940212,
        "max": 1.6046856699772478,
        "mean": 1.229040070157635,
        "median": 1.212808351816429,
        "min": 1.0424727003576841,
        "p10": 1.0664156748591584,
        "p25": 1.097953203320882,
        "p75": 1.2902397520670323,
        "p90": 1.4223819531082542,
        "std": 0.15085286137076842
      },
      "max_gain_spread": 0.5622129696195637,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 2.754293459261703e-06,
          "max_future_KL": 0.0004719505086541176,
          "max_propagation_gain": 0.9588012113516992,
          "mean_G_state": 0.8234947918363085,
          "mean_KL_future": 9.611047394363081e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7601577536757198,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 8.711187983365676e-05,
          "max_future_KL": 0.001567187486216426,
          "max_propagation_gain": 1.0921836704683463,
          "mean_G_state": 0.984299140072965,
          "mean_KL_future": 0.0002741530192461594,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9364458121473088,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 4.67568109236538e-05,
          "max_future_KL": 0.0010688259499147534,
          "max_propagation_gain": 1.0499817604244206,
          "mean_G_state": 1.013058159404618,
          "mean_KL_future": 0.00017608637033772467,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9988885937680305,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.00014234697248056706,
          "max_future_KL": 0.02128773182630539,
          "max_propagation_gain": 1.144223906402074,
          "mean_G_state": 1.027037502567119,
          "mean_KL_future": 0.00290017726612668,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9812096893632931,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 6.941769938677567e-05,
          "max_future_KL": 0.0002364303800277412,
          "max_propagation_gain": 1.099876380938394,
          "mean_G_state": 1.0132720482429904,
          "mean_KL_future": 7.696397298723179e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9770799244647295,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 4.588267558292003e-05,
          "max_future_KL": 0.001249631866812706,
          "max_propagation_gain": 1.2832433934400957,
          "mean_G_state": 1.1192727234267983,
          "mean_KL_future": 0.0001947754491298248,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.065210694777856,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 4.869358577757055e-05,
          "max_future_KL": 0.004042140208184719,
          "max_propagation_gain": 1.2022069698451012,
          "mean_G_state": 1.0398585997241814,
          "mean_KL_future": 0.0008483789832703925,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9778998721143799,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 1.39346858816225e-05,
          "max_future_KL": 0.000813323596958071,
          "max_propagation_gain": 1.082849589293896,
          "mean_G_state": 0.9915916169374406,
          "mean_KL_future": 0.00012830406057801058,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9529232510687958,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 8.584468471539708e-05,
          "max_future_KL": 0.011690949089825153,
          "max_propagation_gain": 1.160558047513325,
          "mean_G_state": 1.0675308598636095,
          "mean_KL_future": 0.0015908768700249598,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0589228357256144,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 16
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 1.724052337941373e-05,
          "max_future_KL": 0.002863778732717037,
          "max_propagation_gain": 1.0424727003576841,
          "mean_G_state": 0.9827764129381433,
          "mean_KL_future": 0.0004492781263578105,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9546355281760366,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 9.571178138543246e-05,
          "max_future_KL": 0.002581777749583125,
          "max_propagation_gain": 1.2234097337877567,
          "mean_G_state": 1.0579627681006756,
          "mean_KL_future": 0.0005642350441095456,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9939058619275165,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 2.4801075884539614e-05,
          "max_future_KL": 0.0011523788562044501,
          "max_propagation_gain": 1.414381979233063,
          "mean_G_state": 1.2207047132083892,
          "mean_KL_future": 0.00021770915104529776,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2392578246710337,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 4.8496494331118355e-05,
          "max_future_KL": 0.06238464266061783,
          "max_propagation_gain": 1.6046856699772478,
          "mean_G_state": 1.303822814432993,
          "mean_KL_future": 0.010015958273668524,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1927413592408735,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 8.078823379733535e-05,
          "max_future_KL": 0.006261120550334454,
          "max_propagation_gain": 1.3112288279478417,
          "mean_G_state": 1.1373610773684324,
          "mean_KL_future": 0.0008553572164289869,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0751755403233685,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 9.410450118210178e-05,
          "max_future_KL": 0.000640308135189116,
          "max_propagation_gain": 1.2583972422732064,
          "mean_G_state": 1.0994388858305053,
          "mean_KL_future": 0.00018109205550020935,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0443643432984528,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 6.386502777715464e-05,
          "max_future_KL": 0.023101741448044777,
          "max_propagation_gain": 1.4303819269834455,
          "mean_G_state": 1.1779291754960917,
          "mean_KL_future": 0.00445048256868541,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0937917225460667,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.000309669329830875,
          "max_future_KL": 0.0009212011354975402,
          "max_propagation_gain": 1.264559323636261,
          "mean_G_state": 1.1485153019875076,
          "mean_KL_future": 0.0002893703459844743,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1319314647580598,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 6.223858441956054e-08,
          "max_future_KL": 0.0002657131408341229,
          "max_propagation_gain": 0.9704337003840069,
          "mean_G_state": 0.7864823621039121,
          "mean_KL_future": 3.323644897103295e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6918323816690193,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 1.866439828788202e-05,
          "max_future_KL": 0.00047202425776049495,
          "max_propagation_gain": 0.9588491064579427,
          "mean_G_state": 0.8223104914353745,
          "mean_KL_future": 7.973113660836306e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7582969347832593,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.08213667484102874,
        "max": 1.2392578246710337,
        "mean": 1.0421490198982135,
        "median": 1.0216264685332417,
        "min": 0.9364458121473088,
        "p10": 0.9537793896224163,
        "p25": 0.9776948852019673,
        "p75": 1.079829585879043,
        "p90": 1.1623364119994668,
        "std": 0.08559865518335849
      },
      "persistence_separation_ratio": 1.3233630911627883,
      "persistence_spread": 0.3028120125237249,
      "problem_id": "test/algebra/1214.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 8.711187983365676e-05,
          "max_future_KL": 0.001567187486216426,
          "max_propagation_gain": 1.0921836704683463,
          "persistence_score": 0.9364458121473088
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 1.39346858816225e-05,
          "max_future_KL": 0.000813323596958071,
          "max_propagation_gain": 1.082849589293896,
          "persistence_score": 0.9529232510687958
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 1.724052337941373e-05,
          "max_future_KL": 0.002863778732717037,
          "max_propagation_gain": 1.0424727003576841,
          "persistence_score": 0.9546355281760366
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 6.941769938677567e-05,
          "max_future_KL": 0.0002364303800277412,
          "max_propagation_gain": 1.099876380938394,
          "persistence_score": 0.9770799244647295
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 4.869358577757055e-05,
          "max_future_KL": 0.004042140208184719,
          "max_propagation_gain": 1.2022069698451012,
          "persistence_score": 0.9778998721143799
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.00014234697248056706,
          "max_future_KL": 0.02128773182630539,
          "max_propagation_gain": 1.144223906402074,
          "persistence_score": 0.9812096893632931
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 9.571178138543246e-05,
          "max_future_KL": 0.002581777749583125,
          "max_propagation_gain": 1.2234097337877567,
          "persistence_score": 0.9939058619275165
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 4.67568109236538e-05,
          "max_future_KL": 0.0010688259499147534,
          "max_propagation_gain": 1.0499817604244206,
          "persistence_score": 0.9988885937680305
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 9.410450118210178e-05,
          "max_future_KL": 0.000640308135189116,
          "max_propagation_gain": 1.2583972422732064,
          "persistence_score": 1.0443643432984528
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 8.584468471539708e-05,
          "max_future_KL": 0.011690949089825153,
          "max_propagation_gain": 1.160558047513325,
          "persistence_score": 1.0589228357256144
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 4.588267558292003e-05,
          "max_future_KL": 0.001249631866812706,
          "max_propagation_gain": 1.2832433934400957,
          "persistence_score": 1.065210694777856
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 8.078823379733535e-05,
          "max_future_KL": 0.006261120550334454,
          "max_propagation_gain": 1.3112288279478417,
          "persistence_score": 1.0751755403233685
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 6.386502777715464e-05,
          "max_future_KL": 0.023101741448044777,
          "max_propagation_gain": 1.4303819269834455,
          "persistence_score": 1.0937917225460667
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.000309669329830875,
          "max_future_KL": 0.0009212011354975402,
          "max_propagation_gain": 1.264559323636261,
          "persistence_score": 1.1319314647580598
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 4.8496494331118355e-05,
          "max_future_KL": 0.06238464266061783,
          "max_propagation_gain": 1.6046856699772478,
          "persistence_score": 1.1927413592408735
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 2.4801075884539614e-05,
          "max_future_KL": 0.0011523788562044501,
          "max_propagation_gain": 1.414381979233063,
          "persistence_score": 1.2392578246710337
        }
      ],
      "random_persistent_amplification_fraction": 0.5,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": 0.12647058823529372,
      "spearman_persistence_future_KL": 0.05882352941176453,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "11/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.6875,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "12/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.0168332974157095,
        "max": 0.0006886290733278067,
        "mean": 0.00015179494546655738,
        "median": 0.00013795990432790005,
        "min": 1.7765131063640637e-05,
        "p10": 3.740550777528995e-05,
        "p25": 5.572242534540584e-05,
        "p75": 0.0001825374921880396,
        "p90": 0.0002246302456057947,
        "std": 0.00015435015594663065
      },
      "future_KL_spread": 0.0006708639422641661,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.072921329556827,
        "max": 0.6261552572250366,
        "mean": 0.13814844700391404,
        "median": 0.10345117375254631,
        "min": 0.0065497723408043385,
        "p10": 0.014856878202408552,
        "p25": 0.027988586574792862,
        "p75": 0.19574951753020287,
        "p90": 0.22492486238479614,
        "std": 0.14822241543672324
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.11176135135772125,
        "max": 1.6012554103021623,
        "mean": 1.3282683528989634,
        "median": 1.3414500430132863,
        "min": 1.0802467688321213,
        "p10": 1.167518257287329,
        "p25": 1.214390201163585,
        "p75": 1.4372958776573443,
        "p90": 1.5203443733860906,
        "std": 0.1484490660857945
      },
      "max_gain_spread": 0.521008641470041,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0001595619014215899,
          "max_future_KL": 0.0016227897722274065,
          "max_propagation_gain": 0.9785885699560902,
          "mean_G_state": 0.8773022723998447,
          "mean_KL_future": 0.00031519222748277453,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.828880440386396,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.0006886290733278067,
          "max_future_KL": 0.10939425975084305,
          "max_propagation_gain": 1.3621473492226701,
          "mean_G_state": 1.1387377673288526,
          "mean_KL_future": 0.01505445036962444,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0540491840672725,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.00013673519526022915,
          "max_future_KL": 0.07362489402294159,
          "max_propagation_gain": 1.39378204322267,
          "mean_G_state": 1.212366102232577,
          "mean_KL_future": 0.009465499863333005,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1421621528467094,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 3.663328478324957e-05,
          "max_future_KL": 0.21565453708171844,
          "max_propagation_gain": 1.6012554103021623,
          "mean_G_state": 1.37652992488677,
          "mean_KL_future": 0.027567152950600082,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2982729069957248,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.00018147353564970103,
          "max_future_KL": 0.02980867214500904,
          "max_propagation_gain": 1.14912045380132,
          "mean_G_state": 1.088241368452982,
          "mean_KL_future": 0.004331703222976824,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0637471992175098,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 4.174908030005753e-05,
          "max_future_KL": 0.022528329864144325,
          "max_propagation_gain": 1.1859160607733379,
          "mean_G_state": 1.1030767949431999,
          "mean_KL_future": 0.0043978666737637395,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0817567670561528,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 6.879314563352778e-05,
          "max_future_KL": 0.03896123170852661,
          "max_propagation_gain": 1.212856153245965,
          "mean_G_state": 1.0989152461865874,
          "mean_KL_future": 0.004918097022786783,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0576827634869899,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 6.0380207027188606e-05,
          "max_future_KL": 0.011463095434010029,
          "max_propagation_gain": 1.0802467688321213,
          "mean_G_state": 1.0351694213027463,
          "mean_KL_future": 0.0018491832792912055,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0109971875918222,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 6.16814365088203e-05,
          "max_future_KL": 0.20338714122772217,
          "max_propagation_gain": 1.5764199198261633,
          "mean_G_state": 1.367654258123682,
          "mean_KL_future": 0.027979849268808366,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2889513060342477,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.00018572936180305533,
          "max_future_KL": 0.23419518768787384,
          "max_propagation_gain": 1.436312051515536,
          "mean_G_state": 1.2296587391152387,
          "mean_KL_future": 0.029746702681569204,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1422099012965563,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 3.817773076733033e-05,
          "max_future_KL": 0.09750808775424957,
          "max_propagation_gain": 1.3824870588217022,
          "mean_G_state": 1.194503608674411,
          "mean_KL_future": 0.014640390306113193,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1255666035005345,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.00022663535542086777,
          "max_future_KL": 0.1717197149991989,
          "max_propagation_gain": 1.2153535391987231,
          "mean_G_state": 1.1226610169794733,
          "mean_KL_future": 0.021886898301623825,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0750059677796695,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0001521639190916524,
          "max_future_KL": 0.0065497723408043385,
          "max_propagation_gain": 1.2162263673188922,
          "mean_G_state": 1.0920661896476824,
          "mean_KL_future": 0.0012599041541785927,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0378004988797207,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.00022262513579072163,
          "max_future_KL": 0.15797066688537598,
          "max_propagation_gain": 1.4402473560827689,
          "mean_G_state": 1.2181221439647159,
          "mean_KL_future": 0.021810623338090585,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1273436349880228,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 1.7765131063640637e-05,
          "max_future_KL": 0.6261552572250366,
          "max_propagation_gain": 1.464268826946018,
          "mean_G_state": 1.2444149163957097,
          "mean_KL_future": 0.07837218542947166,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.1549120465686271,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.00013918461339557098,
          "max_future_KL": 0.018250660970807076,
          "max_propagation_gain": 1.2149015504694585,
          "mean_G_state": 1.1242415317766283,
          "mean_KL_future": 0.0034919283324574835,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0954576886498022,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 16
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.00017036292164149814,
          "max_future_KL": 0.1932036429643631,
          "max_propagation_gain": 1.3207527368039025,
          "mean_G_state": 1.1567486831018363,
          "mean_KL_future": 0.02462801896245237,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.088428923476613,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 3.4876952064166745e-07,
          "max_future_KL": 0.0005388450808823109,
          "max_propagation_gain": 0.969200089580428,
          "mean_G_state": 0.806675387350583,
          "mean_KL_future": 6.765696775512531e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7246433953605033,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0001809568309376175,
          "max_future_KL": 0.001622024574317038,
          "max_propagation_gain": 0.9785635773055199,
          "mean_G_state": 0.8737077365347264,
          "mean_KL_future": 0.00032122398163858534,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8229449674739462,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.06995034106631424,
        "max": 1.2982729069957248,
        "mean": 1.1152715457772484,
        "median": 1.0919433060632076,
        "min": 1.0109971875918222,
        "p10": 1.0459248414734965,
        "p25": 1.0622310902848797,
        "p75": 1.142174089959171,
        "p90": 1.2219316763014374,
        "std": 0.07801362500874398
      },
      "persistence_separation_ratio": 1.2841508591007105,
      "persistence_spread": 0.2872757194039026,
      "problem_id": "test/counting_and_probability/119.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 6.0380207027188606e-05,
          "max_future_KL": 0.011463095434010029,
          "max_propagation_gain": 1.0802467688321213,
          "persistence_score": 1.0109971875918222
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0001521639190916524,
          "max_future_KL": 0.0065497723408043385,
          "max_propagation_gain": 1.2162263673188922,
          "persistence_score": 1.0378004988797207
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.0006886290733278067,
          "max_future_KL": 0.10939425975084305,
          "max_propagation_gain": 1.3621473492226701,
          "persistence_score": 1.0540491840672725
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 6.879314563352778e-05,
          "max_future_KL": 0.03896123170852661,
          "max_propagation_gain": 1.212856153245965,
          "persistence_score": 1.0576827634869899
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.00018147353564970103,
          "max_future_KL": 0.02980867214500904,
          "max_propagation_gain": 1.14912045380132,
          "persistence_score": 1.0637471992175098
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.00022663535542086777,
          "max_future_KL": 0.1717197149991989,
          "max_propagation_gain": 1.2153535391987231,
          "persistence_score": 1.0750059677796695
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 4.174908030005753e-05,
          "max_future_KL": 0.022528329864144325,
          "max_propagation_gain": 1.1859160607733379,
          "persistence_score": 1.0817567670561528
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.00017036292164149814,
          "max_future_KL": 0.1932036429643631,
          "max_propagation_gain": 1.3207527368039025,
          "persistence_score": 1.088428923476613
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.00013918461339557098,
          "max_future_KL": 0.018250660970807076,
          "max_propagation_gain": 1.2149015504694585,
          "persistence_score": 1.0954576886498022
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 3.817773076733033e-05,
          "max_future_KL": 0.09750808775424957,
          "max_propagation_gain": 1.3824870588217022,
          "persistence_score": 1.1255666035005345
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.00022262513579072163,
          "max_future_KL": 0.15797066688537598,
          "max_propagation_gain": 1.4402473560827689,
          "persistence_score": 1.1273436349880228
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.00013673519526022915,
          "max_future_KL": 0.07362489402294159,
          "max_propagation_gain": 1.39378204322267,
          "persistence_score": 1.1421621528467094
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.00018572936180305533,
          "max_future_KL": 0.23419518768787384,
          "max_propagation_gain": 1.436312051515536,
          "persistence_score": 1.1422099012965563
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 1.7765131063640637e-05,
          "max_future_KL": 0.6261552572250366,
          "max_propagation_gain": 1.464268826946018,
          "persistence_score": 1.1549120465686271
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 6.16814365088203e-05,
          "max_future_KL": 0.20338714122772217,
          "max_propagation_gain": 1.5764199198261633,
          "persistence_score": 1.2889513060342477
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 3.663328478324957e-05,
          "max_future_KL": 0.21565453708171844,
          "max_propagation_gain": 1.6012554103021623,
          "persistence_score": 1.2982729069957248
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": -0.17058823529411712,
      "spearman_persistence_future_KL": -0.3617647058823519,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "11/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.1875,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "4/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.9195228997241671,
        "max": 0.01572555040320367,
        "mean": 0.0036362851373879625,
        "median": 0.002875042446418519,
        "min": 0.0010702292902124099,
        "p10": 0.0012905770272084284,
        "p25": 0.0019314365590884287,
        "p75": 0.004107718960075691,
        "p90": 0.0048262217304561265,
        "std": 0.0033436474546743936
      },
      "future_KL_spread": 0.01465532111299126,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.4795794049897573,
        "max": 1.8659604787826538,
        "mean": 0.31839919718913734,
        "median": 0.10223522037267685,
        "min": 0.014664996415376663,
        "p10": 0.017386112362146378,
        "p25": 0.020623987540602684,
        "p75": 0.4012874588370323,
        "p90": 0.7494736909866333,
        "std": 0.47109689472779975
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.20804773098786034,
        "max": 3.23228035324504,
        "mean": 2.058887773942477,
        "median": 1.9550284340263633,
        "min": 1.5904835403174085,
        "p10": 1.6224806845968938,
        "p25": 1.6958871901367474,
        "p75": 2.272103107425975,
        "p90": 2.5028834329550356,
        "std": 0.42834692972758714
      },
      "max_gain_spread": 1.6417968129276317,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.003192264875991313,
          "max_future_KL": 0.01575818844139576,
          "max_propagation_gain": 0.9677808009676502,
          "mean_G_state": 0.8367594343775717,
          "mean_KL_future": 0.0022386828337024323,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7662797470317082,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.002953072769987841,
          "max_future_KL": 0.014664996415376663,
          "max_propagation_gain": 1.9080001820262271,
          "mean_G_state": 1.5166333404714127,
          "mean_KL_future": 0.0025279622815732594,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.458903870353707,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.0029773964759215233,
          "max_future_KL": 0.37190699577331543,
          "max_propagation_gain": 2.294219798696432,
          "mean_G_state": 1.7647530770677324,
          "mean_KL_future": 0.04840695152880414,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.7445012044220527,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0016477662694274642,
          "max_future_KL": 0.01596008986234665,
          "max_propagation_gain": 1.9665365598819686,
          "mean_G_state": 1.697758944155944,
          "mean_KL_future": 0.004331915397581598,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.6804386963469107,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.004167996692723363,
          "max_future_KL": 0.020827725529670715,
          "max_propagation_gain": 2.097013983563244,
          "mean_G_state": 1.615248569441711,
          "mean_KL_future": 0.0049361618417727016,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5546423356793482,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.005198446398274825,
          "max_future_KL": 0.025566041469573975,
          "max_propagation_gain": 2.2647308770024894,
          "mean_G_state": 1.8095527262371705,
          "mean_KL_future": 0.004320710540283379,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.718327631991659,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.004453997062637427,
          "max_future_KL": 0.489428848028183,
          "max_propagation_gain": 2.6009060421597816,
          "mean_G_state": 1.9740804480424339,
          "mean_KL_future": 0.06398750175983992,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.9883391889279582,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0027970121228491963,
          "max_future_KL": 1.8659604787826538,
          "max_propagation_gain": 1.9435203081707582,
          "mean_G_state": 1.4256735743582374,
          "mean_KL_future": 0.23663268082515465,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2233957571363694,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.001305420233586574,
          "max_future_KL": 0.11568175256252289,
          "max_propagation_gain": 1.6516417246583384,
          "mean_G_state": 1.3970519989144852,
          "mean_KL_future": 0.015380846247104785,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3227620613515647,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0038016881540464454,
          "max_future_KL": 0.018812134861946106,
          "max_propagation_gain": 1.6332766118292645,
          "mean_G_state": 1.3985429173700252,
          "mean_KL_future": 0.0026694903677095816,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3440808964561415,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.01572555040320367,
          "max_future_KL": 0.6936202645301819,
          "max_propagation_gain": 3.23228035324504,
          "mean_G_state": 2.224194878815392,
          "mean_KL_future": 0.16112223941212278,
          "mean_Top1_future": 0.875,
          "persistence_score": 2.2002197170800524,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0020259933223087502,
          "max_future_KL": 0.3702811002731323,
          "max_propagation_gain": 2.4048608237502895,
          "mean_G_state": 1.8967133702863836,
          "mean_KL_future": 0.04883556551958179,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.8395362703752316,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.004087626382526466,
          "max_future_KL": 0.02001277357339859,
          "max_propagation_gain": 1.611684757364523,
          "mean_G_state": 1.3154316753646944,
          "mean_KL_future": 0.0028640031732674043,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3636644417021668,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0010702292902124099,
          "max_future_KL": 0.8053271174430847,
          "max_propagation_gain": 2.2509877419937214,
          "mean_G_state": 1.6633426834741964,
          "mean_KL_future": 0.10162536219296214,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.6083060894001253,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.002211819813635074,
          "max_future_KL": 0.02880118042230606,
          "max_propagation_gain": 1.7814253997905953,
          "mean_G_state": 1.4740187099727136,
          "mean_KL_future": 0.006135649013479627,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4291334205440986,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.002480812986036085,
          "max_future_KL": 0.14874696731567383,
          "max_propagation_gain": 1.5904835403174085,
          "mean_G_state": 1.2930714251284126,
          "mean_KL_future": 0.020217733267420712,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2009557538020827,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0012757338208302827,
          "max_future_KL": 0.08878868818283081,
          "max_propagation_gain": 1.7106356786295505,
          "mean_G_state": 1.4617837170213863,
          "mean_KL_future": 0.012653751813758163,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.4116284975771654,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 2.577166794743846e-05,
          "max_future_KL": 0.0001286092883674428,
          "max_propagation_gain": 0.9665958756133578,
          "mean_G_state": 0.7472834222386104,
          "mean_KL_future": 2.7637465948027184e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.640266806038273,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0013175597987863164,
          "max_future_KL": 0.006384940817952156,
          "max_propagation_gain": 0.9675335530117347,
          "mean_G_state": 0.8385363797935037,
          "mean_KL_future": 0.0010369548064407819,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7692586030096074,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.1731495262099635,
        "max": 2.2002197170800524,
        "mean": 1.5680522395716647,
        "median": 1.5067731030165277,
        "min": 1.2009557538020827,
        "p10": 1.2730789092439672,
        "p25": 1.3587685553906605,
        "p75": 1.7248710250992572,
        "p90": 1.913937729651595,
        "std": 0.27150750235447907
      },
      "persistence_separation_ratio": 1.8320572678157265,
      "persistence_spread": 0.9992639632779696,
      "problem_id": "test/geometry/477.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.002480812986036085,
          "max_future_KL": 0.14874696731567383,
          "max_propagation_gain": 1.5904835403174085,
          "persistence_score": 1.2009557538020827
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0027970121228491963,
          "max_future_KL": 1.8659604787826538,
          "max_propagation_gain": 1.9435203081707582,
          "persistence_score": 1.2233957571363694
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.001305420233586574,
          "max_future_KL": 0.11568175256252289,
          "max_propagation_gain": 1.6516417246583384,
          "persistence_score": 1.3227620613515647
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0038016881540464454,
          "max_future_KL": 0.018812134861946106,
          "max_propagation_gain": 1.6332766118292645,
          "persistence_score": 1.3440808964561415
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.004087626382526466,
          "max_future_KL": 0.02001277357339859,
          "max_propagation_gain": 1.611684757364523,
          "persistence_score": 1.3636644417021668
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0012757338208302827,
          "max_future_KL": 0.08878868818283081,
          "max_propagation_gain": 1.7106356786295505,
          "persistence_score": 1.4116284975771654
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.002211819813635074,
          "max_future_KL": 0.02880118042230606,
          "max_propagation_gain": 1.7814253997905953,
          "persistence_score": 1.4291334205440986
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.002953072769987841,
          "max_future_KL": 0.014664996415376663,
          "max_propagation_gain": 1.9080001820262271,
          "persistence_score": 1.458903870353707
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.004167996692723363,
          "max_future_KL": 0.020827725529670715,
          "max_propagation_gain": 2.097013983563244,
          "persistence_score": 1.5546423356793482
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0010702292902124099,
          "max_future_KL": 0.8053271174430847,
          "max_propagation_gain": 2.2509877419937214,
          "persistence_score": 1.6083060894001253
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0016477662694274642,
          "max_future_KL": 0.01596008986234665,
          "max_propagation_gain": 1.9665365598819686,
          "persistence_score": 1.6804386963469107
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.005198446398274825,
          "max_future_KL": 0.025566041469573975,
          "max_propagation_gain": 2.2647308770024894,
          "persistence_score": 1.718327631991659
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.0029773964759215233,
          "max_future_KL": 0.37190699577331543,
          "max_propagation_gain": 2.294219798696432,
          "persistence_score": 1.7445012044220527
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0020259933223087502,
          "max_future_KL": 0.3702811002731323,
          "max_propagation_gain": 2.4048608237502895,
          "persistence_score": 1.8395362703752316
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.004453997062637427,
          "max_future_KL": 0.489428848028183,
          "max_propagation_gain": 2.6009060421597816,
          "persistence_score": 1.9883391889279582
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.01572555040320367,
          "max_future_KL": 0.6936202645301819,
          "max_propagation_gain": 3.23228035324504,
          "persistence_score": 2.2002197170800524
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.34117647058823425,
      "spearman_persistence_future_KL": 0.3617647058823519,
      "t0": 64,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.4375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "8/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.25,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "5/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": false,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.3177969115773704,
        "max": 0.00013884562536645718,
        "mean": 2.5500552190543815e-05,
        "median": 1.131886456882114e-05,
        "min": 1.0369486076911016e-06,
        "p10": 2.004358353246971e-06,
        "p25": 6.018969005694429e-06,
        "p75": 2.9081814494347217e-05,
        "p90": 5.4179597760173334e-05,
        "std": 3.36045502380131e-05
      },
      "future_KL_spread": 0.00013780867675876609,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.4333787908090698,
        "max": 0.6713191866874695,
        "mean": 0.11270619818242267,
        "median": 0.056666815653443336,
        "min": 0.0030729183927178383,
        "p10": 0.005404564552009106,
        "p25": 0.012921121902763844,
        "p75": 0.1496400646865368,
        "p90": 0.2215234413743019,
        "std": 0.16155067406884177
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.1284035398671826,
        "max": 1.793142135815948,
        "mean": 1.3915268506154297,
        "median": 1.3306220195891512,
        "min": 1.181048225236257,
        "p10": 1.2361140625003697,
        "p25": 1.2647403677013616,
        "p75": 1.4741093214620808,
        "p90": 1.6711593179704427,
        "std": 0.17867697343938177
      },
      "max_gain_spread": 0.612093910579691,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 1.065092542011925e-05,
          "max_future_KL": 0.0007527763955295086,
          "max_propagation_gain": 0.9661972326718371,
          "mean_G_state": 0.8888249318873698,
          "mean_KL_future": 0.00010087279022513718,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8504917907379761,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 2.418081521682325e-06,
          "max_future_KL": 0.01043968740850687,
          "max_propagation_gain": 1.324574839954913,
          "mean_G_state": 1.2175922293960089,
          "mean_KL_future": 0.0014680918451688285,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.211324005382147,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.00013884562536645718,
          "max_future_KL": 0.032171670347452164,
          "max_propagation_gain": 1.3971382592712875,
          "mean_G_state": 1.1820919961642429,
          "mean_KL_future": 0.006880335378102842,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2006790315403673,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 128
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 3.35806619318646e-05,
          "max_future_KL": 0.065340556204319,
          "max_propagation_gain": 1.3366691992233894,
          "mean_G_state": 1.1879105663960914,
          "mean_KL_future": 0.008410544840138812,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1910690765087846,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 128
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 2.758219868184142e-05,
          "max_future_KL": 0.22793082892894745,
          "max_propagation_gain": 1.2512805336568498,
          "mean_G_state": 1.1339630470706683,
          "mean_KL_future": 0.028673290783996386,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.111411416497537,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 5.5015180967643575e-05,
          "max_future_KL": 0.0182485468685627,
          "max_propagation_gain": 1.2725038036746377,
          "mean_G_state": 1.168568581416644,
          "mean_KL_future": 0.0036270964970572805,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1534860727126648,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 1.8351305519104245e-05,
          "max_future_KL": 0.09585068374872208,
          "max_propagation_gain": 1.727712035082297,
          "mean_G_state": 1.4800369858805418,
          "mean_KL_future": 0.01214744117851474,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5437204460389689,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 1.100662748925174e-05,
          "max_future_KL": 0.0030729183927178383,
          "max_propagation_gain": 1.2654203618432103,
          "mean_G_state": 1.1691642605625725,
          "mean_KL_future": 0.00039141947467102867,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1588309607345277,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 1.0369486076911016e-06,
          "max_future_KL": 0.08375506103038788,
          "max_propagation_gain": 1.6146066008585882,
          "mean_G_state": 1.3490192593561965,
          "mean_KL_future": 0.010631594284661539,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2853420788934453,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 2.4072907060457282e-05,
          "max_future_KL": 0.21511605381965637,
          "max_propagation_gain": 1.2747351746636104,
          "mean_G_state": 1.1395604106132184,
          "mean_KL_future": 0.03339641399823412,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1140828214672067,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 2.741938517658582e-06,
          "max_future_KL": 0.1455283761024475,
          "max_propagation_gain": 1.2627003852758154,
          "mean_G_state": 1.136204626777642,
          "mean_KL_future": 0.018853963853577405,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1188920860526759,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 5.33440145527031e-05,
          "max_future_KL": 0.00602137204259634,
          "max_propagation_gain": 1.181048225236257,
          "mean_G_state": 1.108238954907412,
          "mean_KL_future": 0.0007864423721481439,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1051436724272699,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 128
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 1.5906351848116174e-06,
          "max_future_KL": 0.013748266734182835,
          "max_propagation_gain": 1.4539330505302595,
          "mean_G_state": 1.330303077643008,
          "mean_KL_future": 0.001878788206500559,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3269153511280638,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 1.163110164839054e-05,
          "max_future_KL": 0.16197513043880463,
          "max_propagation_gain": 1.35337927915838,
          "mean_G_state": 1.1560299292618237,
          "mean_KL_future": 0.020417214743876144,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1092518450226398,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 9.81847769878641e-06,
          "max_future_KL": 0.6713191866874695,
          "max_propagation_gain": 1.793142135815948,
          "mean_G_state": 1.5077962021761822,
          "mean_KL_future": 0.09294633726471879,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.449925376768883,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 9.861817798650917e-06,
          "max_future_KL": 0.04799307510256767,
          "max_propagation_gain": 1.534638134257545,
          "mean_G_state": 1.3648595620393245,
          "mean_KL_future": 0.00877364189394747,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.335374119200774,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 7.111312501706379e-06,
          "max_future_KL": 0.004787757061421871,
          "max_propagation_gain": 1.2209475913438896,
          "mean_G_state": 1.1123462095033865,
          "mean_KL_future": 0.0011981809527608078,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0834436175982791,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 2.589558193477615e-06,
          "max_future_KL": 0.0012898680288344622,
          "max_propagation_gain": 0.9589956319746189,
          "mean_G_state": 0.8084344947862183,
          "mean_KL_future": 0.0001728428144738281,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7279343844159107,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 3.417423443607959e-06,
          "max_future_KL": 0.0007352608372457325,
          "max_propagation_gain": 0.966225697901978,
          "mean_G_state": 0.8903703954953135,
          "mean_KL_future": 9.410652341346193e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.852936653314203,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.10726937072637081,
        "max": 1.5437204460389689,
        "mean": 1.2186807486233895,
        "median": 1.1749500186216562,
        "min": 1.0834436175982791,
        "p10": 1.1071977587249549,
        "p25": 1.1134149702247893,
        "p75": 1.2957353969521,
        "p90": 1.3926497479848283,
        "std": 0.13072711702128076
      },
      "persistence_separation_ratio": 1.4248276707371097,
      "persistence_spread": 0.46027682844068973,
      "problem_id": "test/geometry/702.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 7.111312501706379e-06,
          "max_future_KL": 0.004787757061421871,
          "max_propagation_gain": 1.2209475913438896,
          "persistence_score": 1.0834436175982791
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 5.33440145527031e-05,
          "max_future_KL": 0.00602137204259634,
          "max_propagation_gain": 1.181048225236257,
          "persistence_score": 1.1051436724272699
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 1.163110164839054e-05,
          "max_future_KL": 0.16197513043880463,
          "max_propagation_gain": 1.35337927915838,
          "persistence_score": 1.1092518450226398
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 2.758219868184142e-05,
          "max_future_KL": 0.22793082892894745,
          "max_propagation_gain": 1.2512805336568498,
          "persistence_score": 1.111411416497537
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 2.4072907060457282e-05,
          "max_future_KL": 0.21511605381965637,
          "max_propagation_gain": 1.2747351746636104,
          "persistence_score": 1.1140828214672067
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 2.741938517658582e-06,
          "max_future_KL": 0.1455283761024475,
          "max_propagation_gain": 1.2627003852758154,
          "persistence_score": 1.1188920860526759
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 5.5015180967643575e-05,
          "max_future_KL": 0.0182485468685627,
          "max_propagation_gain": 1.2725038036746377,
          "persistence_score": 1.1534860727126648
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 1.100662748925174e-05,
          "max_future_KL": 0.0030729183927178383,
          "max_propagation_gain": 1.2654203618432103,
          "persistence_score": 1.1588309607345277
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 3.35806619318646e-05,
          "max_future_KL": 0.065340556204319,
          "max_propagation_gain": 1.3366691992233894,
          "persistence_score": 1.1910690765087846
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.00013884562536645718,
          "max_future_KL": 0.032171670347452164,
          "max_propagation_gain": 1.3971382592712875,
          "persistence_score": 1.2006790315403673
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 2.418081521682325e-06,
          "max_future_KL": 0.01043968740850687,
          "max_propagation_gain": 1.324574839954913,
          "persistence_score": 1.211324005382147
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 1.0369486076911016e-06,
          "max_future_KL": 0.08375506103038788,
          "max_propagation_gain": 1.6146066008585882,
          "persistence_score": 1.2853420788934453
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 1.5906351848116174e-06,
          "max_future_KL": 0.013748266734182835,
          "max_propagation_gain": 1.4539330505302595,
          "persistence_score": 1.3269153511280638
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 9.861817798650917e-06,
          "max_future_KL": 0.04799307510256767,
          "max_propagation_gain": 1.534638134257545,
          "persistence_score": 1.335374119200774
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 9.81847769878641e-06,
          "max_future_KL": 0.6713191866874695,
          "max_propagation_gain": 1.793142135815948,
          "persistence_score": 1.449925376768883
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 1.8351305519104245e-05,
          "max_future_KL": 0.09585068374872208,
          "max_propagation_gain": 1.727712035082297,
          "persistence_score": 1.5437204460389689
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": -0.2676470588235286,
      "spearman_persistence_future_KL": -0.30588235294117555,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.25,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "5/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.25,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "5/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.9510931519563367,
        "max": 0.012930163662167616,
        "mean": 0.0036461487344294327,
        "median": 0.0023786588324583137,
        "min": 0.000501159242667626,
        "p10": 0.0005428393650376506,
        "p25": 0.001421196412957215,
        "p75": 0.004941687246218862,
        "p90": 0.008262855690350079,
        "std": 0.00346782709328119
      },
      "future_KL_spread": 0.012429004419499989,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 3.1418357697599446,
        "max": 1.8153104782104492,
        "mean": 0.13803117766656214,
        "median": 0.018900899216532707,
        "min": 0.0011969051556661725,
        "p10": 0.007867417065426707,
        "p25": 0.012753261718899012,
        "p75": 0.03373693209141493,
        "p90": 0.0784870870411396,
        "std": 0.4336712913380368
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.21750561052810538,
        "max": 2.2583274347399187,
        "mean": 1.4457555178219905,
        "median": 1.2990220397937877,
        "min": 1.1023606323689397,
        "p10": 1.1488781288325116,
        "p25": 1.2120972302327928,
        "p75": 1.638220140310504,
        "p90": 1.8037900127514568,
        "std": 0.3144599365784667
      },
      "max_gain_spread": 1.155966802370979,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0014875333321015204,
          "max_future_KL": 0.006671047769486904,
          "max_propagation_gain": 0.979928892402044,
          "mean_G_state": 0.8318053630308962,
          "mean_KL_future": 0.001172287689091167,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7582417196218808,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.004595186438200472,
          "max_future_KL": 0.020730823278427124,
          "max_propagation_gain": 1.2360440319225308,
          "mean_G_state": 1.1110428223896864,
          "mean_KL_future": 0.0034219269050610457,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0922728243095634,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.009765280905367036,
          "max_future_KL": 0.04636763036251068,
          "max_propagation_gain": 1.3066179405609664,
          "mean_G_state": 1.1191264716138571,
          "mean_KL_future": 0.009539789143701682,
          "mean_Top1_future": 0.75,
          "persistence_score": 1.0569828565810928,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0022415776649722828,
          "max_future_KL": 0.017959604039788246,
          "max_propagation_gain": 1.5400204253441987,
          "mean_G_state": 1.2668095339024235,
          "mean_KL_future": 0.0036466786982529698,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2206908141652635,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0016853395689395256,
          "max_future_KL": 0.007810977753251791,
          "max_propagation_gain": 1.1590196552836676,
          "mean_G_state": 1.0396192226569283,
          "mean_KL_future": 0.0018616731414393684,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.0001198289639568,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.005981189670274034,
          "max_future_KL": 1.8153104782104492,
          "max_propagation_gain": 2.2583274347399187,
          "mean_G_state": 1.562897269728652,
          "mean_KL_future": 0.2946182991763635,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2854309473735264,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.000628766945010284,
          "max_future_KL": 0.02710537426173687,
          "max_propagation_gain": 1.1681721398015985,
          "mean_G_state": 1.0458094015675528,
          "mean_KL_future": 0.0037832458004798397,
          "mean_Top1_future": 0.875,
          "persistence_score": 0.989297842000761,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.012930163662167616,
          "max_future_KL": 0.06395771354436874,
          "max_propagation_gain": 1.291426139026609,
          "mean_G_state": 1.1021195726267032,
          "mean_KL_future": 0.012031012393454166,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.0580262861417555,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.000525198905961588,
          "max_future_KL": 0.0011969051556661725,
          "max_propagation_gain": 1.7608464737212242,
          "mean_G_state": 1.2195931605889723,
          "mean_KL_future": 0.0005022803855325719,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0705173392579737,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0018127788638139463,
          "max_future_KL": 0.017404716461896896,
          "max_propagation_gain": 1.1387366023813557,
          "mean_G_state": 1.0146080988668265,
          "mean_KL_future": 0.0033086794800571973,
          "mean_Top1_future": 0.875,
          "persistence_score": 0.9745127230793035,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0005604798241137133,
          "max_future_KL": 0.013030234724283218,
          "max_propagation_gain": 1.2267389270431908,
          "mean_G_state": 1.074899392297306,
          "mean_KL_future": 0.0019792202205920484,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.0217585037287245,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.006760430475333123,
          "max_future_KL": 0.02952669933438301,
          "max_propagation_gain": 1.5973446958402637,
          "mean_G_state": 1.176513353574918,
          "mean_KL_future": 0.006281253477595783,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0923513153288877,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0027522511459210364,
          "max_future_KL": 0.01984219439327717,
          "max_propagation_gain": 1.4587581138306034,
          "mean_G_state": 1.2327379577897113,
          "mean_KL_future": 0.004200440919354342,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1938777883957847,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0031622399226762356,
          "max_future_KL": 0.015392831526696682,
          "max_propagation_gain": 1.1023606323689397,
          "mean_G_state": 1.0150415794698626,
          "mean_KL_future": 0.0022188882046587644,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9679256956178566,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.000501159242667626,
          "max_future_KL": 0.09301646053791046,
          "max_propagation_gain": 1.7944160652399053,
          "mean_G_state": 1.379999938679737,
          "mean_KL_future": 0.011942973691457137,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3082096057261123,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.0019205965155080662,
          "max_future_KL": 0.007923856377601624,
          "max_propagation_gain": 1.2800950477838708,
          "mean_G_state": 1.1015479401399797,
          "mean_KL_future": 0.002190855334601949,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0910206217922984,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0025157399999443442,
          "max_future_KL": 0.011922342702746391,
          "max_propagation_gain": 1.8131639602630083,
          "mean_G_state": 1.3882472515079711,
          "mean_KL_future": 0.004426496116593626,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.1745328590488193,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 5.403283310343454e-05,
          "max_future_KL": 0.0001435734739061445,
          "max_propagation_gain": 0.9743730781399036,
          "mean_G_state": 0.708486002929191,
          "mean_KL_future": 3.3858418623466946e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6028230599193151,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0013946515249699588,
          "max_future_KL": 0.006671106442809105,
          "max_propagation_gain": 0.9799345349444568,
          "mean_G_state": 0.8314762029238179,
          "mean_KL_future": 0.0011142915502667123,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.757705908930363,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.09451438947540713,
        "max": 1.3082096057261123,
        "mean": 1.0998454907194801,
        "median": 1.0807689805251361,
        "min": 0.9679256956178566,
        "p10": 0.9819052825400323,
        "p25": 1.0163488350375325,
        "p75": 1.1793690913855608,
        "p90": 1.2530608807693948,
        "std": 0.10395122507272575
      },
      "persistence_separation_ratio": 1.3515599509833145,
      "persistence_spread": 0.34028391010825576,
      "problem_id": "test/algebra/1332.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0031622399226762356,
          "max_future_KL": 0.015392831526696682,
          "max_propagation_gain": 1.1023606323689397,
          "persistence_score": 0.9679256956178566
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0018127788638139463,
          "max_future_KL": 0.017404716461896896,
          "max_propagation_gain": 1.1387366023813557,
          "persistence_score": 0.9745127230793035
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.000628766945010284,
          "max_future_KL": 0.02710537426173687,
          "max_propagation_gain": 1.1681721398015985,
          "persistence_score": 0.989297842000761
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0016853395689395256,
          "max_future_KL": 0.007810977753251791,
          "max_propagation_gain": 1.1590196552836676,
          "persistence_score": 1.0001198289639568
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0005604798241137133,
          "max_future_KL": 0.013030234724283218,
          "max_propagation_gain": 1.2267389270431908,
          "persistence_score": 1.0217585037287245
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.009765280905367036,
          "max_future_KL": 0.04636763036251068,
          "max_propagation_gain": 1.3066179405609664,
          "persistence_score": 1.0569828565810928
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.012930163662167616,
          "max_future_KL": 0.06395771354436874,
          "max_propagation_gain": 1.291426139026609,
          "persistence_score": 1.0580262861417555
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.000525198905961588,
          "max_future_KL": 0.0011969051556661725,
          "max_propagation_gain": 1.7608464737212242,
          "persistence_score": 1.0705173392579737
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.0019205965155080662,
          "max_future_KL": 0.007923856377601624,
          "max_propagation_gain": 1.2800950477838708,
          "persistence_score": 1.0910206217922984
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.004595186438200472,
          "max_future_KL": 0.020730823278427124,
          "max_propagation_gain": 1.2360440319225308,
          "persistence_score": 1.0922728243095634
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.006760430475333123,
          "max_future_KL": 0.02952669933438301,
          "max_propagation_gain": 1.5973446958402637,
          "persistence_score": 1.0923513153288877
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0025157399999443442,
          "max_future_KL": 0.011922342702746391,
          "max_propagation_gain": 1.8131639602630083,
          "persistence_score": 1.1745328590488193
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0027522511459210364,
          "max_future_KL": 0.01984219439327717,
          "max_propagation_gain": 1.4587581138306034,
          "persistence_score": 1.1938777883957847
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0022415776649722828,
          "max_future_KL": 0.017959604039788246,
          "max_propagation_gain": 1.5400204253441987,
          "persistence_score": 1.2206908141652635
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.005981189670274034,
          "max_future_KL": 1.8153104782104492,
          "max_propagation_gain": 2.2583274347399187,
          "persistence_score": 1.2854309473735264
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.000501159242667626,
          "max_future_KL": 0.09301646053791046,
          "max_propagation_gain": 1.7944160652399053,
          "persistence_score": 1.3082096057261123
        }
      ],
      "random_persistent_amplification_fraction": 0.8125,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.10588235294117615,
      "spearman_persistence_future_KL": 0.08823529411764679,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "2/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.1875,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "4/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": false,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.9419067775619377,
        "max": 0.004315043834083277,
        "mean": 0.0010497949980955614,
        "median": 0.0008605352128276067,
        "min": 5.632984321692902e-05,
        "p10": 0.000212013475420747,
        "p25": 0.0005009865326620399,
        "p75": 0.0011345879392110848,
        "p90": 0.001878120574367348,
        "std": 0.0009888090246987376
      },
      "future_KL_spread": 0.004258713990866347,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 0.9694473888702625,
        "max": 0.02155190333724022,
        "mean": 0.005118656110425945,
        "median": 0.0041788737289607525,
        "min": 0.00027904100716114044,
        "p10": 0.0010033997823484242,
        "p25": 0.002492287429049611,
        "p75": 0.005649765837006271,
        "p90": 0.00931405695155263,
        "std": 0.0049622678017466944
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.055965705933333755,
        "max": 1.2552209387926143,
        "mean": 1.0547244218072473,
        "median": 1.031937463458764,
        "min": 0.9981247552069542,
        "p10": 1.015109108441567,
        "p25": 1.0199410272859635,
        "p75": 1.0640067333255017,
        "p90": 1.096346293782807,
        "std": 0.05902839683162585
      },
      "max_gain_spread": 0.2570961835856601,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.00012112968191395624,
          "max_future_KL": 0.00034857814898714423,
          "max_propagation_gain": 0.9725394991744363,
          "mean_G_state": 0.830592664624526,
          "mean_KL_future": 7.57292458197778e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7606771577315022,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 5.632984321692902e-05,
          "max_future_KL": 0.00027904100716114044,
          "max_propagation_gain": 1.0259180192870958,
          "mean_G_state": 0.9687983271889096,
          "mean_KL_future": 3.5217854033042784e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9447175352728043,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.002050569741740027,
          "max_future_KL": 0.010176138952374458,
          "max_propagation_gain": 1.1099612768620633,
          "mean_G_state": 1.0186993808504778,
          "mean_KL_future": 0.001281708390132752,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9843506286107162,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0006587091329457451,
          "max_future_KL": 0.0027049502823501825,
          "max_propagation_gain": 1.0202863732043044,
          "mean_G_state": 0.9686898895340251,
          "mean_KL_future": 0.00041179940780988744,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9438457265774538,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0006267699225666945,
          "max_future_KL": 0.002946156542748213,
          "max_propagation_gain": 1.0127457113214575,
          "mean_G_state": 0.9584656421642207,
          "mean_KL_future": 0.00039175649996944784,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9320322244047194,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.0008244788290298289,
          "max_future_KL": 0.003940672613680363,
          "max_propagation_gain": 0.9981247552069542,
          "mean_G_state": 0.9468178426637824,
          "mean_KL_future": 0.0005153451918131768,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9183273443441224,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.00022025580362559083,
          "max_future_KL": 0.0010159367229789495,
          "max_propagation_gain": 1.0623873134746278,
          "mean_G_state": 0.9798308232109157,
          "mean_KL_future": 0.0001377196090565358,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.974901773700676,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 64
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.00020377114721590318,
          "max_future_KL": 0.0009908628417178988,
          "max_propagation_gain": 1.0337339681029911,
          "mean_G_state": 0.9712384020597025,
          "mean_KL_future": 0.0001274416618721741,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9418582633401099,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0011147828232079605,
          "max_future_KL": 0.005008278880268335,
          "max_propagation_gain": 1.0614991963499374,
          "mean_G_state": 0.9954565479213847,
          "mean_KL_future": 0.000696823766651089,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9717651142966697,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0017056714069946688,
          "max_future_KL": 0.0084519749507308,
          "max_propagation_gain": 1.0688649928781233,
          "mean_G_state": 1.0025263659880093,
          "mean_KL_future": 0.0010661455863914676,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.995895415788963,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 64
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.00037535833546655797,
          "max_future_KL": 0.0018542988691478968,
          "max_propagation_gain": 1.028643574670563,
          "mean_G_state": 0.9713407139923391,
          "mean_KL_future": 0.00023461515575196046,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9481979315161606,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0011664437961573348,
          "max_future_KL": 0.005747369956225157,
          "max_propagation_gain": 1.082731310703551,
          "mean_G_state": 0.9822197050167439,
          "mean_KL_future": 0.0007290481084373801,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9431475690931268,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0008965915966253846,
          "max_future_KL": 0.004417074844241142,
          "max_propagation_gain": 1.018904989530941,
          "mean_G_state": 0.9565143950711013,
          "mean_KL_future": 0.0005603864110205192,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9264578155854105,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0011239693202290014,
          "max_future_KL": 0.005617231130599976,
          "max_propagation_gain": 1.0489548641545214,
          "mean_G_state": 0.9782521508447531,
          "mean_KL_future": 0.0007024967813853422,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.952402018733593,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0009151118380302137,
          "max_future_KL": 0.004490324296057224,
          "max_propagation_gain": 1.0174725055616767,
          "mean_G_state": 0.9592655095267697,
          "mean_KL_future": 0.0005720275120100204,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9332929653515812,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.004315043834083277,
          "max_future_KL": 0.02155190333724022,
          "max_propagation_gain": 1.0301409588145367,
          "mean_G_state": 0.9597325632857407,
          "mean_KL_future": 0.0026969013786136697,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.928369005795154,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0005428625983938673,
          "max_future_KL": 0.00270628253929317,
          "max_propagation_gain": 1.2552209387926143,
          "mean_G_state": 1.068536022319928,
          "mean_KL_future": 0.0003641689022892969,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0573470296609584,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 5.554179006269366e-05,
          "max_future_KL": 0.00027755857445299625,
          "max_propagation_gain": 0.9558124800810444,
          "mean_G_state": 0.7529100861842991,
          "mean_KL_future": 3.473635264109563e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6437888295753826,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0002672052341207376,
          "max_future_KL": 0.0009891221998259425,
          "max_propagation_gain": 0.9723766135054844,
          "mean_G_state": 0.8312093209727934,
          "mean_KL_future": 0.00016704351034158016,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7617961240182543,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.03521097350524728,
        "max": 1.0573470296609584,
        "mean": 0.9560567726295137,
        "median": 0.9442816309251291,
        "min": 0.9183273443441224,
        "p10": 0.9274134106902823,
        "p25": 0.9329777801148658,
        "p75": 0.9725492791476713,
        "p90": 0.9901230221998396,
        "std": 0.03366368969060524
      },
      "persistence_separation_ratio": 1.1513835847009148,
      "persistence_spread": 0.13901968531683595,
      "problem_id": "test/algebra/1214.json",
      "random_amplification_fraction": 0.9375,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.0008244788290298289,
          "max_future_KL": 0.003940672613680363,
          "max_propagation_gain": 0.9981247552069542,
          "persistence_score": 0.9183273443441224
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0008965915966253846,
          "max_future_KL": 0.004417074844241142,
          "max_propagation_gain": 1.018904989530941,
          "persistence_score": 0.9264578155854105
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.004315043834083277,
          "max_future_KL": 0.02155190333724022,
          "max_propagation_gain": 1.0301409588145367,
          "persistence_score": 0.928369005795154
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0006267699225666945,
          "max_future_KL": 0.002946156542748213,
          "max_propagation_gain": 1.0127457113214575,
          "persistence_score": 0.9320322244047194
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0009151118380302137,
          "max_future_KL": 0.004490324296057224,
          "max_propagation_gain": 1.0174725055616767,
          "persistence_score": 0.9332929653515812
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.00020377114721590318,
          "max_future_KL": 0.0009908628417178988,
          "max_propagation_gain": 1.0337339681029911,
          "persistence_score": 0.9418582633401099
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0011664437961573348,
          "max_future_KL": 0.005747369956225157,
          "max_propagation_gain": 1.082731310703551,
          "persistence_score": 0.9431475690931268
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0006587091329457451,
          "max_future_KL": 0.0027049502823501825,
          "max_propagation_gain": 1.0202863732043044,
          "persistence_score": 0.9438457265774538
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 5.632984321692902e-05,
          "max_future_KL": 0.00027904100716114044,
          "max_propagation_gain": 1.0259180192870958,
          "persistence_score": 0.9447175352728043
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.00037535833546655797,
          "max_future_KL": 0.0018542988691478968,
          "max_propagation_gain": 1.028643574670563,
          "persistence_score": 0.9481979315161606
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0011239693202290014,
          "max_future_KL": 0.005617231130599976,
          "max_propagation_gain": 1.0489548641545214,
          "persistence_score": 0.952402018733593
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0011147828232079605,
          "max_future_KL": 0.005008278880268335,
          "max_propagation_gain": 1.0614991963499374,
          "persistence_score": 0.9717651142966697
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.00022025580362559083,
          "max_future_KL": 0.0010159367229789495,
          "max_propagation_gain": 1.0623873134746278,
          "persistence_score": 0.974901773700676
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.002050569741740027,
          "max_future_KL": 0.010176138952374458,
          "max_propagation_gain": 1.1099612768620633,
          "persistence_score": 0.9843506286107162
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0017056714069946688,
          "max_future_KL": 0.0084519749507308,
          "max_propagation_gain": 1.0688649928781233,
          "persistence_score": 0.995895415788963
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0005428625983938673,
          "max_future_KL": 0.00270628253929317,
          "max_propagation_gain": 1.2552209387926143,
          "persistence_score": 1.0573470296609584
        }
      ],
      "random_persistent_amplification_fraction": 0.0625,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": 0.2676470588235286,
      "spearman_persistence_future_KL": 0.00882352941176468,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.8792851076753732,
        "max": 0.03846137485335248,
        "mean": 0.014219227200052673,
        "median": 0.008717930862070488,
        "min": 0.003355101295097995,
        "p10": 0.0038598186994008186,
        "p25": 0.005271165331300809,
        "p75": 0.016602712975175394,
        "p90": 0.03669134324044485,
        "std": 0.012502754720538195
      },
      "future_KL_spread": 0.035106273558254485,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 3.24372262007776,
        "max": 9.252939224243164,
        "mean": 0.6843327949754894,
        "median": 0.03579321503639221,
        "min": 0.008871670812368393,
        "p10": 0.010853548534214497,
        "p25": 0.016955638770014048,
        "p75": 0.14041176810860634,
        "p90": 0.5617338865995407,
        "std": 2.2197857667262744
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.3742436683961045,
        "max": 3.843034326052449,
        "mean": 2.120870636688028,
        "median": 1.9030762216728472,
        "min": 1.256810464328956,
        "p10": 1.2713922138912215,
        "p25": 1.536575826375601,
        "p75": 2.651624783932852,
        "p90": 3.2582725958478687,
        "std": 0.7937224072680837
      },
      "max_gain_spread": 2.586223861723493,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.002237653625036884,
          "max_future_KL": 0.005285477731376886,
          "max_propagation_gain": 0.9759088528511423,
          "mean_G_state": 0.9102247760574835,
          "mean_KL_future": 0.001454278536123281,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8806907638441526,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.03540254862263268,
          "max_future_KL": 9.252939224243164,
          "max_propagation_gain": 3.843034326052449,
          "mean_G_state": 3.0111353624297,
          "mean_KL_future": 1.1944490877983127,
          "mean_Top1_future": 0.875,
          "persistence_score": 2.67301853950679,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.0034789304731759783,
          "max_future_KL": 0.008959531784057617,
          "max_propagation_gain": 1.9126933104807335,
          "mean_G_state": 1.6212833218480975,
          "mean_KL_future": 0.0029573365474568902,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5834629982265336,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.006860841093339331,
          "max_future_KL": 0.026142846792936325,
          "max_propagation_gain": 1.5093185871053532,
          "mean_G_state": 1.4068833309575253,
          "mean_KL_future": 0.004970536026604577,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3878544917770121,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.008436081475076662,
          "max_future_KL": 0.03588452935218811,
          "max_propagation_gain": 1.545661572799017,
          "mean_G_state": 1.3821061386789886,
          "mean_KL_future": 0.0055190056676295285,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4354173151997205,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.012123923114292622,
          "max_future_KL": 0.03560996800661087,
          "max_propagation_gain": 2.2590826647473627,
          "mean_G_state": 1.7662728024307532,
          "mean_KL_future": 0.011617030145900431,
          "mean_Top1_future": 1.0,
          "persistence_score": 2.0136081845899536,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.03846137485335248,
          "max_future_KL": 0.13368386030197144,
          "max_propagation_gain": 2.67714419501778,
          "mean_G_state": 1.8617930493297565,
          "mean_KL_future": 0.026314356291329943,
          "mean_Top1_future": 0.875,
          "persistence_score": 2.151452066688299,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.011604046355616049,
          "max_future_KL": 0.035701900720596313,
          "max_propagation_gain": 2.0819832092338473,
          "mean_G_state": 1.6095408500477866,
          "mean_KL_future": 0.009719761942606375,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.702191107915716,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.012740702704581964,
          "max_future_KL": 0.04191381111741066,
          "max_propagation_gain": 1.6446006893003593,
          "mean_G_state": 1.3911174444169843,
          "mean_KL_future": 0.008617721175781412,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.421541029595113,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0047337035214312095,
          "max_future_KL": 0.0172351635992527,
          "max_propagation_gain": 1.274038805074407,
          "mean_G_state": 1.215216423166339,
          "mean_KL_future": 0.0037989020378050853,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2019783157070407,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.028188743786955683,
          "max_future_KL": 0.43559595942497253,
          "max_propagation_gain": 3.729426863474206,
          "mean_G_state": 2.52476589440741,
          "mean_KL_future": 0.09233606942905581,
          "mean_Top1_future": 1.0,
          "persistence_score": 2.7965936919317875,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.008999780249064315,
          "max_future_KL": 0.03945431858301163,
          "max_propagation_gain": 1.8934591328649606,
          "mean_G_state": 1.5986213730715553,
          "mean_KL_future": 0.0061115087924434874,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5822908447699382,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.037980137858257025,
          "max_future_KL": 0.16059549152851105,
          "max_propagation_gain": 2.7871183282215317,
          "mean_G_state": 1.9827545215675393,
          "mean_KL_future": 0.02610207863664815,
          "mean_Top1_future": 0.875,
          "persistence_score": 2.2822766494440687,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.005450319267924009,
          "max_future_KL": 0.012747565284371376,
          "max_propagation_gain": 1.256810464328956,
          "mean_G_state": 1.1821600996181534,
          "mean_KL_future": 0.0036441879319534243,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2051994861748394,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 16
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.005450693604419144,
          "max_future_KL": 0.016117064282298088,
          "max_propagation_gain": 1.6076941020282374,
          "mean_G_state": 1.399064548950566,
          "mean_KL_future": 0.0037144967386115724,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4726751530389794,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.004240706925625659,
          "max_future_KL": 0.6878718137741089,
          "max_propagation_gain": 2.6431183135712093,
          "mean_G_state": 2.142222725696973,
          "mean_KL_future": 0.09906048986995053,
          "mean_Top1_future": 1.0,
          "persistence_score": 2.101528057137126,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.003355101295097995,
          "max_future_KL": 0.008871670812368393,
          "max_propagation_gain": 1.268745622708036,
          "mean_G_state": 1.208511395846783,
          "mean_KL_future": 0.002261949415996334,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.20141989638219,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 64
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 0.00018716158709182197,
          "max_future_KL": 0.0009325314313173294,
          "max_propagation_gain": 0.9767926445896816,
          "mean_G_state": 0.8129020060006724,
          "mean_KL_future": 0.00013597605047688255,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7345515100759104,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0018408244417156717,
          "max_future_KL": 0.00392485223710537,
          "max_propagation_gain": 0.9758873332979737,
          "mean_G_state": 0.9067897969950854,
          "mean_KL_future": 0.0012048999258620907,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8751891009365348,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.2818974069794536,
        "max": 2.7965936919317875,
        "mean": 1.7632817392553195,
        "median": 1.582876921498236,
        "min": 1.20141989638219,
        "p10": 1.20358890094094,
        "p25": 1.4131193951405876,
        "p75": 2.1140090595249195,
        "p90": 2.477647594475429,
        "std": 0.49706455007057754
      },
      "persistence_separation_ratio": 2.3277404514031956,
      "persistence_spread": 1.5951737955495975,
      "problem_id": "test/intermediate_algebra/207.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.003355101295097995,
          "max_future_KL": 0.008871670812368393,
          "max_propagation_gain": 1.268745622708036,
          "persistence_score": 1.20141989638219
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0047337035214312095,
          "max_future_KL": 0.0172351635992527,
          "max_propagation_gain": 1.274038805074407,
          "persistence_score": 1.2019783157070407
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.005450319267924009,
          "max_future_KL": 0.012747565284371376,
          "max_propagation_gain": 1.256810464328956,
          "persistence_score": 1.2051994861748394
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.006860841093339331,
          "max_future_KL": 0.026142846792936325,
          "max_propagation_gain": 1.5093185871053532,
          "persistence_score": 1.3878544917770121
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.012740702704581964,
          "max_future_KL": 0.04191381111741066,
          "max_propagation_gain": 1.6446006893003593,
          "persistence_score": 1.421541029595113
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.008436081475076662,
          "max_future_KL": 0.03588452935218811,
          "max_propagation_gain": 1.545661572799017,
          "persistence_score": 1.4354173151997205
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.005450693604419144,
          "max_future_KL": 0.016117064282298088,
          "max_propagation_gain": 1.6076941020282374,
          "persistence_score": 1.4726751530389794
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.008999780249064315,
          "max_future_KL": 0.03945431858301163,
          "max_propagation_gain": 1.8934591328649606,
          "persistence_score": 1.5822908447699382
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.0034789304731759783,
          "max_future_KL": 0.008959531784057617,
          "max_propagation_gain": 1.9126933104807335,
          "persistence_score": 1.5834629982265336
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.011604046355616049,
          "max_future_KL": 0.035701900720596313,
          "max_propagation_gain": 2.0819832092338473,
          "persistence_score": 1.702191107915716
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.012123923114292622,
          "max_future_KL": 0.03560996800661087,
          "max_propagation_gain": 2.2590826647473627,
          "persistence_score": 2.0136081845899536
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.004240706925625659,
          "max_future_KL": 0.6878718137741089,
          "max_propagation_gain": 2.6431183135712093,
          "persistence_score": 2.101528057137126
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.03846137485335248,
          "max_future_KL": 0.13368386030197144,
          "max_propagation_gain": 2.67714419501778,
          "persistence_score": 2.151452066688299
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.037980137858257025,
          "max_future_KL": 0.16059549152851105,
          "max_propagation_gain": 2.7871183282215317,
          "persistence_score": 2.2822766494440687
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.03540254862263268,
          "max_future_KL": 9.252939224243164,
          "max_propagation_gain": 3.843034326052449,
          "persistence_score": 2.67301853950679
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.028188743786955683,
          "max_future_KL": 0.43559595942497253,
          "max_propagation_gain": 3.729426863474206,
          "persistence_score": 2.7965936919317875
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": 0.6911764705882332,
      "spearman_persistence_future_KL": 0.6735294117647038,
      "t0": 64,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "2/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.7470681268186892,
        "max": 0.004739757568859204,
        "mean": 0.0015838752598355652,
        "median": 0.00134753665520293,
        "min": 0.0001752156746078981,
        "p10": 0.0003757998565561138,
        "p25": 0.0005849966015944118,
        "p75": 0.0023154911331289443,
        "p90": 0.0028195617369078715,
        "std": 0.0011832627242268883
      },
      "future_KL_spread": 0.004564541894251306,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 3.0586905808031375,
        "max": 5.338512897491455,
        "mean": 0.42496477827080525,
        "median": 0.009372972417622805,
        "min": 0.0031398041173815727,
        "p10": 0.003919893875718117,
        "p25": 0.006422018748708069,
        "p75": 0.028546457644551992,
        "p90": 0.6539446525275707,
        "std": 1.2998357644730645
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.28448061252159174,
        "max": 3.789817218569606,
        "mean": 1.9904388606380758,
        "median": 1.902752247908884,
        "min": 1.2665614156406952,
        "p10": 1.3786858745725896,
        "p25": 1.7640967995573615,
        "p75": 2.123622785736106,
        "p90": 2.4241299938826693,
        "std": 0.5662412662613835
      },
      "max_gain_spread": 2.5232558029289107,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.00010168143085138581,
          "max_future_KL": 0.00040077793528325856,
          "max_propagation_gain": 0.9797313777720708,
          "mean_G_state": 0.8506724846431672,
          "mean_KL_future": 7.425379520764608e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7768738840620233,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.0023185001445199306,
          "max_future_KL": 0.010548964142799377,
          "max_propagation_gain": 1.332217447132413,
          "mean_G_state": 1.1551398958591677,
          "mean_KL_future": 0.0016117928076653243,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1416515861166983,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.00026306596482754684,
          "max_future_KL": 0.0068890973925590515,
          "max_propagation_gain": 1.8912950302666436,
          "mean_G_state": 1.3417301094057685,
          "mean_KL_future": 0.001559822080196227,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.270933408423756,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.002481712816302206,
          "max_future_KL": 1.1851152181625366,
          "max_propagation_gain": 2.4219260182355105,
          "mean_G_state": 1.8210683198873214,
          "mean_KL_future": 0.15073434371095118,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.6580185870072914,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0011309621899272316,
          "max_future_KL": 0.004499544855207205,
          "max_propagation_gain": 1.7941445809967191,
          "mean_G_state": 1.2736342949588786,
          "mean_KL_future": 0.0011948969197135284,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1904261575980395,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.0005904643143367138,
          "max_future_KL": 0.0033402428962290287,
          "max_propagation_gain": 2.1128813417597545,
          "mean_G_state": 1.469043894160599,
          "mean_KL_future": 0.0009042921982427288,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2922445588916252,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0023144881293319488,
          "max_future_KL": 0.12277408689260483,
          "max_propagation_gain": 2.4263339695298285,
          "mean_G_state": 1.6797423549152808,
          "mean_KL_future": 0.017213477768535768,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5952763134016004,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0015550238274954608,
          "max_future_KL": 0.005020782817155123,
          "max_propagation_gain": 1.9451959838787205,
          "mean_G_state": 1.391093165322733,
          "mean_KL_future": 0.0015145669276535045,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3435920392957335,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.001789789610711523,
          "max_future_KL": 0.007995451800525188,
          "max_propagation_gain": 1.9142094655511244,
          "mean_G_state": 1.4825133788197435,
          "mean_KL_future": 0.001602139843811301,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3960394040133974,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0005685934633675061,
          "max_future_KL": 0.016113484278321266,
          "max_propagation_gain": 2.1558471176651604,
          "mean_G_state": 1.475480049913873,
          "mean_KL_future": 0.002456739061834412,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3973664688809158,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0004885337482846807,
          "max_future_KL": 0.010433402843773365,
          "max_propagation_gain": 1.9952584937250493,
          "mean_G_state": 1.3556035171396081,
          "mean_KL_future": 0.0017583494617809947,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.255882904629382,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0018383909095064156,
          "max_future_KL": 0.008274692110717297,
          "max_propagation_gain": 1.8216387787413242,
          "mean_G_state": 1.2603545961750167,
          "mean_KL_future": 0.0013734698246068433,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2170299725880103,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0007900456548668444,
          "max_future_KL": 0.008312541991472244,
          "max_propagation_gain": 1.4251543020127662,
          "mean_G_state": 1.2188383697022043,
          "mean_KL_future": 0.0015547568339648876,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1209561312719014,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.004739757568859204,
          "max_future_KL": 0.02285979501903057,
          "max_propagation_gain": 1.8805871512646095,
          "mean_G_state": 1.4507335711295777,
          "mean_KL_future": 0.004331814772569942,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.358512599455822,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0031574106575135374,
          "max_future_KL": 5.338512897491455,
          "max_propagation_gain": 3.789817218569606,
          "mean_G_state": 2.6034559572773666,
          "mean_KL_future": 0.6836510186848272,
          "mean_Top1_future": 0.875,
          "persistence_score": 2.253635702099726,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.0011400494829103991,
          "max_future_KL": 0.04560644552111626,
          "max_propagation_gain": 1.2665614156406952,
          "mean_G_state": 1.1276874064938718,
          "mean_KL_future": 0.006723252946649438,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0716971476763348,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0001752156746078981,
          "max_future_KL": 0.0031398041173815727,
          "max_propagation_gain": 1.673953455239289,
          "mean_G_state": 1.2254407223023567,
          "mean_KL_future": 0.0005840217881569743,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2021380870318477,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 6.361439836055638e-06,
          "max_future_KL": 8.323719521285966e-05,
          "max_propagation_gain": 0.9777408644979921,
          "mean_G_state": 0.73422924719284,
          "mean_KL_future": 1.673434284898967e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6016822887346156,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0002302152409356495,
          "max_future_KL": 0.0005458443192765117,
          "max_propagation_gain": 0.9797795583557343,
          "mean_G_state": 0.8512956384127399,
          "mean_KL_future": 0.00020122173950554156,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7776759913705429,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.2039136788041726,
        "max": 2.253635702099726,
        "mean": 1.3603375667738802,
        "median": 1.2815889836576906,
        "min": 1.0716971476763348,
        "p10": 1.1313038586942998,
        "p25": 1.1992101046733956,
        "p75": 1.3963711702302772,
        "p90": 1.626647450204446,
        "std": 0.27739143765658264
      },
      "persistence_separation_ratio": 2.1028661940399673,
      "persistence_spread": 1.1819385544233911,
      "problem_id": "test/geometry/477.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.0011400494829103991,
          "max_future_KL": 0.04560644552111626,
          "max_propagation_gain": 1.2665614156406952,
          "persistence_score": 1.0716971476763348
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0007900456548668444,
          "max_future_KL": 0.008312541991472244,
          "max_propagation_gain": 1.4251543020127662,
          "persistence_score": 1.1209561312719014
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.0023185001445199306,
          "max_future_KL": 0.010548964142799377,
          "max_propagation_gain": 1.332217447132413,
          "persistence_score": 1.1416515861166983
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0011309621899272316,
          "max_future_KL": 0.004499544855207205,
          "max_propagation_gain": 1.7941445809967191,
          "persistence_score": 1.1904261575980395
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0001752156746078981,
          "max_future_KL": 0.0031398041173815727,
          "max_propagation_gain": 1.673953455239289,
          "persistence_score": 1.2021380870318477
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0018383909095064156,
          "max_future_KL": 0.008274692110717297,
          "max_propagation_gain": 1.8216387787413242,
          "persistence_score": 1.2170299725880103
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0004885337482846807,
          "max_future_KL": 0.010433402843773365,
          "max_propagation_gain": 1.9952584937250493,
          "persistence_score": 1.255882904629382
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.00026306596482754684,
          "max_future_KL": 0.0068890973925590515,
          "max_propagation_gain": 1.8912950302666436,
          "persistence_score": 1.270933408423756
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.0005904643143367138,
          "max_future_KL": 0.0033402428962290287,
          "max_propagation_gain": 2.1128813417597545,
          "persistence_score": 1.2922445588916252
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0015550238274954608,
          "max_future_KL": 0.005020782817155123,
          "max_propagation_gain": 1.9451959838787205,
          "persistence_score": 1.3435920392957335
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.004739757568859204,
          "max_future_KL": 0.02285979501903057,
          "max_propagation_gain": 1.8805871512646095,
          "persistence_score": 1.358512599455822
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.001789789610711523,
          "max_future_KL": 0.007995451800525188,
          "max_propagation_gain": 1.9142094655511244,
          "persistence_score": 1.3960394040133974
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0005685934633675061,
          "max_future_KL": 0.016113484278321266,
          "max_propagation_gain": 2.1558471176651604,
          "persistence_score": 1.3973664688809158
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0023144881293319488,
          "max_future_KL": 0.12277408689260483,
          "max_propagation_gain": 2.4263339695298285,
          "persistence_score": 1.5952763134016004
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.002481712816302206,
          "max_future_KL": 1.1851152181625366,
          "max_propagation_gain": 2.4219260182355105,
          "persistence_score": 1.6580185870072914
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0031574106575135374,
          "max_future_KL": 5.338512897491455,
          "max_propagation_gain": 3.789817218569606,
          "persistence_score": 2.253635702099726
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.20588235294117585,
      "spearman_persistence_future_KL": 0.4117647058823517,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.4375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "8/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.4375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "8/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.0723630409636584,
        "max": 0.0005751509618551864,
        "mean": 0.00013228683671602788,
        "median": 8.435080144266748e-05,
        "min": 8.101592115750122e-06,
        "p10": 2.7694214652917995e-05,
        "p25": 3.7220553413463973e-05,
        "p75": 0.00016382801247116774,
        "p90": 0.0002793827743843735,
        "std": 0.00014185951557262565
      },
      "future_KL_spread": 0.0005670493697394363,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 3.776835436305804,
        "max": 4.610424995422363,
        "mean": 0.29502280431916006,
        "median": 0.0066339680925011635,
        "min": 0.0014947500312700868,
        "p10": 0.0016368787037208676,
        "p25": 0.0019970469002146274,
        "p75": 0.009193470235913992,
        "p90": 0.023642677813768387,
        "std": 1.1142525818746936
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.19995281210108973,
        "max": 2.106209491535388,
        "mean": 1.3654254505010774,
        "median": 1.2882151569599136,
        "min": 1.042969528763741,
        "p10": 1.087038663391062,
        "p25": 1.1625512483990108,
        "p75": 1.4849607608910773,
        "p90": 1.6471258872873757,
        "std": 0.2730206585422877
      },
      "max_gain_spread": 1.0632399627716471,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 5.5557929178462474e-05,
          "max_future_KL": 0.00026185979368165135,
          "max_propagation_gain": 0.9831322300222887,
          "mean_G_state": 0.8767751237245709,
          "mean_KL_future": 5.752401431025067e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8270695530471072,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 2.604888128878713e-05,
          "max_future_KL": 0.0066306875087320805,
          "max_propagation_gain": 1.4754996464631598,
          "mean_G_state": 1.259861563202476,
          "mean_KL_future": 0.0011057055308854036,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2106277739997764,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 3.8581353989819434e-05,
          "max_future_KL": 0.0016296951798722148,
          "max_propagation_gain": 1.042969528763741,
          "mean_G_state": 0.9992594576553341,
          "mean_KL_future": 0.00035337902136545196,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9849003810728633,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0001117469262943871,
          "max_future_KL": 0.006884126458317041,
          "max_propagation_gain": 1.2538312804243357,
          "mean_G_state": 1.0548053357600828,
          "mean_KL_future": 0.0009792259516321344,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0450134679634269,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 3.31381516843976e-05,
          "max_future_KL": 0.0016440622275695205,
          "max_propagation_gain": 1.1325969435556393,
          "mean_G_state": 1.0380901769607895,
          "mean_KL_future": 0.0003269117121982301,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0094664979525922,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.00023001319465763004,
          "max_future_KL": 0.0014947500312700868,
          "max_propagation_gain": 1.0970730751806808,
          "mean_G_state": 1.0305624948417618,
          "mean_KL_future": 0.0004063617553407839,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0218443924504552,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0005751509618551864,
          "max_future_KL": 4.610424995422363,
          "max_propagation_gain": 2.106209491535388,
          "mean_G_state": 1.3729936465349728,
          "mean_KL_future": 0.5907993398061919,
          "mean_Top1_future": 0.75,
          "persistence_score": 1.2224154273651258,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 5.0419811119084557e-05,
          "max_future_KL": 0.024339601397514343,
          "max_propagation_gain": 1.776079989203326,
          "mean_G_state": 1.2862816520128506,
          "mean_KL_future": 0.003543144519764896,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0968321878413176,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0001710186023117899,
          "max_future_KL": 0.0066372486762702465,
          "max_propagation_gain": 1.2535267977225928,
          "mean_G_state": 1.075928134391229,
          "mean_KL_future": 0.0010126890185579995,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0287941320813583,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 4.756785084936155e-05,
          "max_future_KL": 0.0021072598174214363,
          "max_propagation_gain": 1.0770042516014433,
          "mean_G_state": 1.0218157137876107,
          "mean_KL_future": 0.0004533868252308837,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0130084816367617,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 8.101592115750122e-06,
          "max_future_KL": 0.004159484524279833,
          "max_propagation_gain": 1.4693306045334376,
          "mean_G_state": 1.1433953039421114,
          "mean_KL_future": 0.0006195635878241124,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0250276883507468,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 2.9339548017048856e-05,
          "max_future_KL": 0.0031104078516364098,
          "max_propagation_gain": 1.2057106725894304,
          "mean_G_state": 1.103802086461786,
          "mean_KL_future": 0.0004323194439423972,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0705347275206993,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.00016143114919096036,
          "max_future_KL": 0.00879820715636015,
          "max_propagation_gain": 1.5133441041748301,
          "mean_G_state": 1.1852680748460125,
          "mean_KL_future": 0.0012055146902529845,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.063215723668838,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.00010055725186077779,
          "max_future_KL": 0.007512921001762152,
          "max_propagation_gain": 1.1725360166801346,
          "mean_G_state": 1.050741170206895,
          "mean_KL_future": 0.0012575257689064756,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0185925884189642,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0001365774070857917,
          "max_future_KL": 0.01037925947457552,
          "max_propagation_gain": 1.3225990334954913,
          "mean_G_state": 1.0777605291119885,
          "mean_KL_future": 0.0016102494512224008,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9915772830447231,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.00032875235411111703,
          "max_future_KL": 0.0016664081485942006,
          "max_propagation_gain": 1.4303239867221789,
          "mean_G_state": 1.1124825166911028,
          "mean_KL_future": 0.00059066205263969,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1279036457371927,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 6.814435102455718e-05,
          "max_future_KL": 0.02294575423002243,
          "max_propagation_gain": 1.5181717853714254,
          "mean_G_state": 1.1832494744380844,
          "mean_KL_future": 0.003565326530814872,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.059095858932106,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 4.70149778681872e-06,
          "max_future_KL": 0.001695800106972456,
          "max_propagation_gain": 0.8833806177406326,
          "mean_G_state": 0.7009735243006957,
          "mean_KL_future": 0.00023897581626031172,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.615433271385696,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 6.125607833205038e-05,
          "max_future_KL": 0.0002817110507749021,
          "max_propagation_gain": 0.9831367441670946,
          "mean_G_state": 0.8776746303227663,
          "mean_KL_future": 5.924367019871646e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8284734748298943,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.06481487275086485,
        "max": 1.2224154273651258,
        "mean": 1.0618031411273092,
        "median": 1.0369038000223925,
        "min": 0.9849003810728633,
        "p10": 1.0005218904986577,
        "p25": 1.0171965617234136,
        "p75": 1.077109092600854,
        "p90": 1.1692657098684847,
        "std": 0.06882063547869996
      },
      "persistence_separation_ratio": 1.2411564162786632,
      "persistence_spread": 0.2375150462922625,
      "problem_id": "test/geometry/702.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 3.8581353989819434e-05,
          "max_future_KL": 0.0016296951798722148,
          "max_propagation_gain": 1.042969528763741,
          "persistence_score": 0.9849003810728633
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0001365774070857917,
          "max_future_KL": 0.01037925947457552,
          "max_propagation_gain": 1.3225990334954913,
          "persistence_score": 0.9915772830447231
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 3.31381516843976e-05,
          "max_future_KL": 0.0016440622275695205,
          "max_propagation_gain": 1.1325969435556393,
          "persistence_score": 1.0094664979525922
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 4.756785084936155e-05,
          "max_future_KL": 0.0021072598174214363,
          "max_propagation_gain": 1.0770042516014433,
          "persistence_score": 1.0130084816367617
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.00010055725186077779,
          "max_future_KL": 0.007512921001762152,
          "max_propagation_gain": 1.1725360166801346,
          "persistence_score": 1.0185925884189642
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.00023001319465763004,
          "max_future_KL": 0.0014947500312700868,
          "max_propagation_gain": 1.0970730751806808,
          "persistence_score": 1.0218443924504552
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 8.101592115750122e-06,
          "max_future_KL": 0.004159484524279833,
          "max_propagation_gain": 1.4693306045334376,
          "persistence_score": 1.0250276883507468
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0001710186023117899,
          "max_future_KL": 0.0066372486762702465,
          "max_propagation_gain": 1.2535267977225928,
          "persistence_score": 1.0287941320813583
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0001117469262943871,
          "max_future_KL": 0.006884126458317041,
          "max_propagation_gain": 1.2538312804243357,
          "persistence_score": 1.0450134679634269
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 6.814435102455718e-05,
          "max_future_KL": 0.02294575423002243,
          "max_propagation_gain": 1.5181717853714254,
          "persistence_score": 1.059095858932106
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.00016143114919096036,
          "max_future_KL": 0.00879820715636015,
          "max_propagation_gain": 1.5133441041748301,
          "persistence_score": 1.063215723668838
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 2.9339548017048856e-05,
          "max_future_KL": 0.0031104078516364098,
          "max_propagation_gain": 1.2057106725894304,
          "persistence_score": 1.0705347275206993
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 5.0419811119084557e-05,
          "max_future_KL": 0.024339601397514343,
          "max_propagation_gain": 1.776079989203326,
          "persistence_score": 1.0968321878413176
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.00032875235411111703,
          "max_future_KL": 0.0016664081485942006,
          "max_propagation_gain": 1.4303239867221789,
          "persistence_score": 1.1279036457371927
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 2.604888128878713e-05,
          "max_future_KL": 0.0066306875087320805,
          "max_propagation_gain": 1.4754996464631598,
          "persistence_score": 1.2106277739997764
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0005751509618551864,
          "max_future_KL": 4.610424995422363,
          "max_propagation_gain": 2.106209491535388,
          "persistence_score": 1.2224154273651258
        }
      ],
      "random_persistent_amplification_fraction": 0.875,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.19411764705882295,
      "spearman_persistence_future_KL": 0.2117647058823523,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.4375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "8/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.6875,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "12/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.056708781689809,
        "max": 0.013960908165191377,
        "mean": 0.003551292605133626,
        "median": 0.002586163131259056,
        "min": 2.897696305126374e-07,
        "p10": 0.00038784706796793356,
        "p25": 0.00043836950978111134,
        "p75": 0.004272065900321342,
        "p90": 0.008037449001757225,
        "std": 0.0037526820832514914
      },
      "future_KL_spread": 0.013960618395560865,
      "future_KL_spread_threshold": 0.0009039054597852925,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 0.9331681633612343,
        "max": 0.0697709321975708,
        "mean": 0.019233697261370253,
        "median": 0.017475939355790615,
        "min": 0.0019119706703349948,
        "p10": 0.002609505783766508,
        "p25": 0.00687787716742605,
        "p75": 0.02377435378730297,
        "p90": 0.03981718048453331,
        "std": 0.017948273948972047
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.12150842389940314,
        "max": 1.9511187640356786,
        "mean": 1.6268795260039421,
        "median": 1.6853309242181076,
        "min": 1.3092125006599136,
        "p10": 1.3727111823583846,
        "p25": 1.4022790224002575,
        "p75": 1.7726033356948623,
        "p90": 1.8445984548370489,
        "std": 0.19767956707906859
      },
      "max_gain_spread": 0.6419062633757651,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0016708632276745128,
          "max_future_KL": 0.0077520571649074554,
          "max_propagation_gain": 0.986521670128355,
          "mean_G_state": 0.9167834438391506,
          "mean_KL_future": 0.001062258407189931,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8788717450725626,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.006241237375077447,
          "max_future_KL": 0.03108849748969078,
          "max_propagation_gain": 1.7625274782435507,
          "mean_G_state": 1.3487937290393297,
          "mean_KL_future": 0.00470132015439928,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3415404665885615,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.00973681161709763,
          "max_future_KL": 0.04854586347937584,
          "max_propagation_gain": 1.881406370555385,
          "mean_G_state": 1.3837523446871978,
          "mean_KL_future": 0.00722491982316642,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3250155743951524,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0035020145844583796,
          "max_future_KL": 0.017476079985499382,
          "max_propagation_gain": 1.3843216144918291,
          "mean_G_state": 1.1560957898322106,
          "mean_KL_future": 0.0023648035173238213,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.1415984560295822,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0015421816266474764,
          "max_future_KL": 0.009113620035350323,
          "max_propagation_gain": 1.695098734817451,
          "mean_G_state": 1.2955241915393394,
          "mean_KL_future": 0.002103237453227713,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2739957172176828,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 2.897696305126374e-07,
          "max_future_KL": 0.0026080142706632614,
          "max_propagation_gain": 1.3092125006599136,
          "mean_G_state": 1.174597862757035,
          "mean_KL_future": 0.00032625041615497175,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1825940368500387,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.00044915668748330974,
          "max_future_KL": 0.021336393430829048,
          "max_propagation_gain": 1.9511187640356786,
          "mean_G_state": 1.639785157603118,
          "mean_KL_future": 0.0029479898309254793,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.7036389426975027,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0036156754087359744,
          "max_future_KL": 0.017475809901952744,
          "max_propagation_gain": 1.7047485292497262,
          "mean_G_state": 1.2624119727061704,
          "mean_KL_future": 0.002324075606408238,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2569073045302288,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.00040600797667451617,
          "max_future_KL": 0.002610997296869755,
          "max_propagation_gain": 1.3837455553312925,
          "mean_G_state": 1.1885634435903836,
          "mean_KL_future": 0.0005803678413309754,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1832466517342284,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0015571608851239205,
          "max_future_KL": 0.007752179633826017,
          "max_propagation_gain": 1.5528045825114762,
          "mean_G_state": 1.2697432288015027,
          "mean_KL_future": 0.0017736947184423535,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2794999361720027,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0016709215631529161,
          "max_future_KL": 0.00775207718834281,
          "max_propagation_gain": 1.593424844528952,
          "mean_G_state": 1.2408825350113213,
          "mean_KL_future": 0.0010624677950655803,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2553363613014576,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.013960908165191377,
          "max_future_KL": 0.0697709321975708,
          "max_propagation_gain": 1.675563113618764,
          "mean_G_state": 1.2786160112876332,
          "mean_KL_future": 0.00880083749577143,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2856404936122325,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0003865990271243125,
          "max_future_KL": 0.00425527710467577,
          "max_propagation_gain": 1.4082648250364005,
          "mean_G_state": 1.1840984974896767,
          "mean_KL_future": 0.0007737456936911039,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1809083361677666,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0003890951088115546,
          "max_future_KL": 0.0019119706703349948,
          "max_propagation_gain": 1.3616768093854767,
          "mean_G_state": 1.1463432925255612,
          "mean_KL_future": 0.0002614278595517716,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1472800512714791,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.006338086386416819,
          "max_future_KL": 0.03108823485672474,
          "max_propagation_gain": 1.8028309080487968,
          "mean_G_state": 1.3101606858199886,
          "mean_KL_future": 0.00413732588239224,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3022171585317517,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.003501404699365196,
          "max_future_KL": 0.017476068809628487,
          "max_propagation_gain": 1.7555372464296723,
          "mean_G_state": 1.3085793426454357,
          "mean_KL_future": 0.002364693971894871,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2996230533123516,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0035231308011466787,
          "max_future_KL": 0.017477139830589294,
          "max_propagation_gain": 1.807790539118713,
          "mean_G_state": 1.4598566929496941,
          "mean_KL_future": 0.0023410931761626147,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.4814092927466527,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 6.595626175176505e-06,
          "max_future_KL": 0.00014367862604558468,
          "max_propagation_gain": 0.9183322032351329,
          "mean_G_state": 0.6757285750976363,
          "mean_KL_future": 2.2080102868304863e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.5398987006507048,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.00361562183914117,
          "max_future_KL": 0.017475860193371773,
          "max_propagation_gain": 0.9863875535010058,
          "mean_G_state": 0.9161659414217261,
          "mean_KL_future": 0.002259771299916835,
          "mean_Top1_future": 0.875,
          "persistence_score": 0.8771052761065927,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.10467873550321129,
        "max": 1.7036389426975027,
        "mean": 1.290028239572417,
        "median": 1.2767478266948427,
        "min": 1.1415984560295822,
        "p10": 1.164094193719623,
        "p25": 1.183083498013181,
        "p75": 1.3079167624976018,
        "p90": 1.411474879667607,
        "std": 0.135038524881979
      },
      "persistence_separation_ratio": 1.4923276513715467,
      "persistence_spread": 0.5620404866679205,
      "problem_id": "test/algebra/1332.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0035020145844583796,
          "max_future_KL": 0.017476079985499382,
          "max_propagation_gain": 1.3843216144918291,
          "persistence_score": 1.1415984560295822
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0003890951088115546,
          "max_future_KL": 0.0019119706703349948,
          "max_propagation_gain": 1.3616768093854767,
          "persistence_score": 1.1472800512714791
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0003865990271243125,
          "max_future_KL": 0.00425527710467577,
          "max_propagation_gain": 1.4082648250364005,
          "persistence_score": 1.1809083361677666
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 2.897696305126374e-07,
          "max_future_KL": 0.0026080142706632614,
          "max_propagation_gain": 1.3092125006599136,
          "persistence_score": 1.1825940368500387
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.00040600797667451617,
          "max_future_KL": 0.002610997296869755,
          "max_propagation_gain": 1.3837455553312925,
          "persistence_score": 1.1832466517342284
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0016709215631529161,
          "max_future_KL": 0.00775207718834281,
          "max_propagation_gain": 1.593424844528952,
          "persistence_score": 1.2553363613014576
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0036156754087359744,
          "max_future_KL": 0.017475809901952744,
          "max_propagation_gain": 1.7047485292497262,
          "persistence_score": 1.2569073045302288
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0015421816266474764,
          "max_future_KL": 0.009113620035350323,
          "max_propagation_gain": 1.695098734817451,
          "persistence_score": 1.2739957172176828
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0015571608851239205,
          "max_future_KL": 0.007752179633826017,
          "max_propagation_gain": 1.5528045825114762,
          "persistence_score": 1.2794999361720027
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.013960908165191377,
          "max_future_KL": 0.0697709321975708,
          "max_propagation_gain": 1.675563113618764,
          "persistence_score": 1.2856404936122325
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.003501404699365196,
          "max_future_KL": 0.017476068809628487,
          "max_propagation_gain": 1.7555372464296723,
          "persistence_score": 1.2996230533123516
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.006338086386416819,
          "max_future_KL": 0.03108823485672474,
          "max_propagation_gain": 1.8028309080487968,
          "persistence_score": 1.3022171585317517
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.00973681161709763,
          "max_future_KL": 0.04854586347937584,
          "max_propagation_gain": 1.881406370555385,
          "persistence_score": 1.3250155743951524
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.006241237375077447,
          "max_future_KL": 0.03108849748969078,
          "max_propagation_gain": 1.7625274782435507,
          "persistence_score": 1.3415404665885615
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0035231308011466787,
          "max_future_KL": 0.017477139830589294,
          "max_propagation_gain": 1.807790539118713,
          "persistence_score": 1.4814092927466527
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.00044915668748330974,
          "max_future_KL": 0.021336393430829048,
          "max_propagation_gain": 1.9511187640356786,
          "persistence_score": 1.7036389426975027
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.6147058823529393,
      "spearman_persistence_future_KL": 0.5352941176470573,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "top_quartile",
        "empirical_percentile_vs_random": 0.8125,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "14/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "7/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "top_quartile",
        "empirical_percentile_vs_random": 0.9375,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "16/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": false,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.39632038580684725,
        "max": 0.0063001135010504195,
        "mean": 0.0036739837059851944,
        "median": 0.003736518868277544,
        "min": 0.0011527330252695477,
        "p10": 0.001542773416103893,
        "p25": 0.002848176957683601,
        "p75": 0.004812785220314775,
        "p90": 0.005382295638548085,
        "std": 0.0014560746402004431
      },
      "future_KL_spread": 0.005147380475780871,
      "future_KL_spread_threshold": 0.0015535455106687967,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 0.8049924604614739,
        "max": 0.0781714990735054,
        "mean": 0.021326750313164666,
        "median": 0.015270852018147707,
        "min": 0.0051136985421180725,
        "p10": 0.007922125747427344,
        "p25": 0.011529708048328757,
        "p75": 0.024418992456048727,
        "p90": 0.03539974056184292,
        "std": 0.017167873209046926
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.04115925269218407,
        "max": 1.152347464901102,
        "mean": 1.055758095869605,
        "median": 1.0469809617824564,
        "min": 1.0096563548473665,
        "p10": 1.0137975644907034,
        "p25": 1.023419681789382,
        "p75": 1.0653557136416132,
        "p90": 1.1273476385377537,
        "std": 0.043454214249757324
      },
      "max_gain_spread": 0.14269111005373558,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.005077442599549009,
          "max_future_KL": 0.01573064923286438,
          "max_propagation_gain": 1.0262565300502537,
          "mean_G_state": 0.8272683873711959,
          "mean_KL_future": 0.005140339005376027,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7346379468076003,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.0063001135010504195,
          "max_future_KL": 0.03220057860016823,
          "max_propagation_gain": 1.113508243815344,
          "mean_G_state": 1.0133399457627599,
          "mean_KL_future": 0.00811615370648923,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9751546453286934,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.003751977016893715,
          "max_future_KL": 0.0781714990735054,
          "max_propagation_gain": 1.1411870332601635,
          "mean_G_state": 1.0062428729377684,
          "mean_KL_future": 0.012116669514304845,
          "mean_Top1_future": 0.875,
          "persistence_score": 0.962037368008135,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.00412018965838874,
          "max_future_KL": 0.011699717491865158,
          "max_propagation_gain": 1.0321679322670363,
          "mean_G_state": 0.9329541326011151,
          "mean_KL_future": 0.0036577011883684035,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8884477071081929,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.005001636783526253,
          "max_future_KL": 0.023712221533060074,
          "max_propagation_gain": 1.0142230616083014,
          "mean_G_state": 0.93864247344869,
          "mean_KL_future": 0.006091386951765837,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9040750778292409,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.005606121215815829,
          "max_future_KL": 0.02606462687253952,
          "max_propagation_gain": 1.0800803413850484,
          "mean_G_state": 0.9673090034739982,
          "mean_KL_future": 0.006763333195272725,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9204425170782656,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.004749834699244282,
          "max_future_KL": 0.014827783219516277,
          "max_propagation_gain": 1.0503656004979423,
          "mean_G_state": 0.98203653672735,
          "mean_KL_future": 0.004694874099304802,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9548992723613072,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.002912360553973636,
          "max_future_KL": 0.01446606032550335,
          "max_propagation_gain": 1.025629139152823,
          "mean_G_state": 0.9232900648137714,
          "mean_KL_future": 0.00429138530191589,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8741889962238648,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.003721060719661373,
          "max_future_KL": 0.015790428966283798,
          "max_propagation_gain": 1.0184310787143445,
          "mean_G_state": 0.9429153888125085,
          "mean_KL_future": 0.003864190395171767,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9053493439562386,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.00171396315740715,
          "max_future_KL": 0.0051136985421180725,
          "max_propagation_gain": 1.0133720673731053,
          "mean_G_state": 0.9246604159568453,
          "mean_KL_future": 0.001648166679586227,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8808781393840693,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.004228192516569695,
          "max_future_KL": 0.015713920816779137,
          "max_propagation_gain": 1.0096563548473665,
          "mean_G_state": 0.9381552460810785,
          "mean_KL_future": 0.0028557306399328852,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9037874663808079,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0032917979588590997,
          "max_future_KL": 0.014134188182651997,
          "max_propagation_gain": 1.152347464901102,
          "mean_G_state": 1.0158337076662716,
          "mean_KL_future": 0.0038293486493145013,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9737872589594673,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0011527330252695477,
          "max_future_KL": 0.005808715242892504,
          "max_propagation_gain": 1.0435963230669703,
          "mean_G_state": 0.9618327251577139,
          "mean_KL_future": 0.0014480920938554354,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9273976533603051,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0026556261688134965,
          "max_future_KL": 0.010035536251962185,
          "max_propagation_gain": 1.0585938530786436,
          "mean_G_state": 0.9593315630513072,
          "mean_KL_future": 0.0019447744647105025,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9205981606417246,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.001371583674800636,
          "max_future_KL": 0.03859890252351761,
          "max_propagation_gain": 1.0250825494810611,
          "mean_G_state": 0.9416487050751743,
          "mean_KL_future": 0.005683392030760237,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9036439504542886,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.00515847006128034,
          "max_future_KL": 0.023870447650551796,
          "max_propagation_gain": 1.0604475043938015,
          "mean_G_state": 0.973741509790761,
          "mean_KL_future": 0.006369050329823045,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9475300882531243,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.003048078584208902,
          "max_future_KL": 0.011019679717719555,
          "max_propagation_gain": 1.0534409860706253,
          "mean_G_state": 0.9491959472548248,
          "mean_KL_future": 0.0028356616883123964,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9030750873307092,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 0.0007085404153439133,
          "max_future_KL": 0.002585635520517826,
          "max_propagation_gain": 0.9687962324152752,
          "mean_G_state": 0.7184806181203097,
          "mean_KL_future": 0.0006869206067416623,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.5840121612742158,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.006214182042675187,
          "max_future_KL": 0.015151944942772388,
          "max_propagation_gain": 1.0254620610634595,
          "mean_G_state": 0.8278339486050785,
          "mean_KL_future": 0.005778549732206206,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7358273278659009,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.033885447212124116,
        "max": 0.9751546453286934,
        "mean": 0.921580795791152,
        "median": 0.9128959305172522,
        "min": 0.8741889962238648,
        "p10": 0.8846629232461312,
        "p25": 0.9035017346733938,
        "p75": 0.94937238428017,
        "p90": 0.9679123134838011,
        "std": 0.0312281774075223
      },
      "persistence_separation_ratio": 1.115496362388274,
      "persistence_spread": 0.10096564910482864,
      "problem_id": "test/counting_and_probability/119.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.002912360553973636,
          "max_future_KL": 0.01446606032550335,
          "max_propagation_gain": 1.025629139152823,
          "persistence_score": 0.8741889962238648
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.00171396315740715,
          "max_future_KL": 0.0051136985421180725,
          "max_propagation_gain": 1.0133720673731053,
          "persistence_score": 0.8808781393840693
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.00412018965838874,
          "max_future_KL": 0.011699717491865158,
          "max_propagation_gain": 1.0321679322670363,
          "persistence_score": 0.8884477071081929
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.003048078584208902,
          "max_future_KL": 0.011019679717719555,
          "max_propagation_gain": 1.0534409860706253,
          "persistence_score": 0.9030750873307092
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.001371583674800636,
          "max_future_KL": 0.03859890252351761,
          "max_propagation_gain": 1.0250825494810611,
          "persistence_score": 0.9036439504542886
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.004228192516569695,
          "max_future_KL": 0.015713920816779137,
          "max_propagation_gain": 1.0096563548473665,
          "persistence_score": 0.9037874663808079
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.005001636783526253,
          "max_future_KL": 0.023712221533060074,
          "max_propagation_gain": 1.0142230616083014,
          "persistence_score": 0.9040750778292409
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.003721060719661373,
          "max_future_KL": 0.015790428966283798,
          "max_propagation_gain": 1.0184310787143445,
          "persistence_score": 0.9053493439562386
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.005606121215815829,
          "max_future_KL": 0.02606462687253952,
          "max_propagation_gain": 1.0800803413850484,
          "persistence_score": 0.9204425170782656
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0026556261688134965,
          "max_future_KL": 0.010035536251962185,
          "max_propagation_gain": 1.0585938530786436,
          "persistence_score": 0.9205981606417246
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0011527330252695477,
          "max_future_KL": 0.005808715242892504,
          "max_propagation_gain": 1.0435963230669703,
          "persistence_score": 0.9273976533603051
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.00515847006128034,
          "max_future_KL": 0.023870447650551796,
          "max_propagation_gain": 1.0604475043938015,
          "persistence_score": 0.9475300882531243
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.004749834699244282,
          "max_future_KL": 0.014827783219516277,
          "max_propagation_gain": 1.0503656004979423,
          "persistence_score": 0.9548992723613072
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.003751977016893715,
          "max_future_KL": 0.0781714990735054,
          "max_propagation_gain": 1.1411870332601635,
          "persistence_score": 0.962037368008135
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0032917979588590997,
          "max_future_KL": 0.014134188182651997,
          "max_propagation_gain": 1.152347464901102,
          "persistence_score": 0.9737872589594673
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.0063001135010504195,
          "max_future_KL": 0.03220057860016823,
          "max_propagation_gain": 1.113508243815344,
          "persistence_score": 0.9751546453286934
        }
      ],
      "random_persistent_amplification_fraction": 0.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.28235294117646975,
      "spearman_persistence_future_KL": 0.402941176470587,
      "t0": 64,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.25,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "5/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.7209660656961232,
        "max": 0.014228682101730072,
        "mean": 0.004895040570257536,
        "median": 0.004495189571444768,
        "min": 0.0011474000436564324,
        "p10": 0.0016719327743325962,
        "p25": 0.0022394459336412707,
        "p75": 0.005098085478025638,
        "p90": 0.010138684790559438,
        "std": 0.0035291581420824497
      },
      "future_KL_spread": 0.01308128205807364,
      "future_KL_spread_threshold": 0.0005135975721487273,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 0.8644122639806462,
        "max": 0.05921326205134392,
        "mean": 0.016942755857598968,
        "median": 0.013051738031208515,
        "min": 0.0024368467275053263,
        "p10": 0.005031372653320432,
        "p25": 0.007913439185358584,
        "p75": 0.01629515364766121,
        "p90": 0.03741579130291939,
        "std": 0.01464552594980289
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.13977633978283077,
        "max": 1.723735178749933,
        "mean": 1.3966434869115485,
        "median": 1.3754684402255064,
        "min": 1.1121445075358307,
        "p10": 1.1517071103342458,
        "p25": 1.206063795249845,
        "p75": 1.539883931640817,
        "p90": 1.6764832396850067,
        "std": 0.19521771458216594
      },
      "max_gain_spread": 0.6115906712141024,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0005817780875527312,
          "max_future_KL": 0.002163912169635296,
          "max_propagation_gain": 0.9724718834593378,
          "mean_G_state": 0.8749137441789232,
          "mean_KL_future": 0.0003641128337361077,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8257953150293567,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.005048924123309461,
          "max_future_KL": 0.016587531194090843,
          "max_propagation_gain": 1.2111654982295632,
          "mean_G_state": 1.0968531864989934,
          "mean_KL_future": 0.003412677641868811,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1052800564673493,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.00439733862789069,
          "max_future_KL": 0.012094114907085896,
          "max_propagation_gain": 1.5707962993670503,
          "mean_G_state": 1.3324414286092872,
          "mean_KL_future": 0.003915919545212532,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2432035680392435,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.014228682101730072,
          "max_future_KL": 0.05921326205134392,
          "max_propagation_gain": 1.5295798090654062,
          "mean_G_state": 1.2355737015351507,
          "mean_KL_future": 0.011733944251875794,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1778625238979692,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0011474000436564324,
          "max_future_KL": 0.004565841052681208,
          "max_propagation_gain": 1.4608708138975568,
          "mean_G_state": 1.321137568484621,
          "mean_KL_future": 0.0013084025169325386,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3325888812890336,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.0027105965996000237,
          "max_future_KL": 0.007980825379490852,
          "max_propagation_gain": 1.4738957572881528,
          "mean_G_state": 1.2579790646733675,
          "mean_KL_future": 0.0026732408637544225,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.238203076431087,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0019254693235996711,
          "max_future_KL": 0.00798010639846325,
          "max_propagation_gain": 1.3184230441918485,
          "mean_G_state": 1.213226397955,
          "mean_KL_future": 0.0012597800828324512,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2198557279222613,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0014183962250655213,
          "max_future_KL": 0.0024368467275053263,
          "max_propagation_gain": 1.1121445075358307,
          "mean_G_state": 1.0473800617562907,
          "mean_KL_future": 0.0009382262413542009,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0449999281273272,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0023284934532540546,
          "max_future_KL": 0.007713437546044588,
          "max_propagation_gain": 1.19075868631069,
          "mean_G_state": 1.0456291369875346,
          "mean_KL_future": 0.0017408000553889025,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9906426512798653,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.004658662002593417,
          "max_future_KL": 0.012856886722147465,
          "max_propagation_gain": 1.3679807252457301,
          "mean_G_state": 1.1773544082600482,
          "mean_KL_future": 0.004198486203011775,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1653854521553602,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.005245569542174167,
          "max_future_KL": 0.016197694465517998,
          "max_propagation_gain": 1.696477160075966,
          "mean_G_state": 1.45734723589145,
          "mean_KL_future": 0.004571290008884787,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.454535817433253,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.009683472952858097,
          "max_future_KL": 0.038731709122657776,
          "max_propagation_gain": 1.6564893192940473,
          "mean_G_state": 1.4523776964473452,
          "mean_KL_future": 0.006953082801443244,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3938690549430448,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.004593040514998847,
          "max_future_KL": 0.015336363576352596,
          "max_propagation_gain": 1.3476086154592248,
          "mean_G_state": 1.19749003953061,
          "mean_KL_future": 0.003469205749034643,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2055782396302872,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0037641337156856026,
          "max_future_KL": 0.013246589340269566,
          "max_propagation_gain": 1.153466180937131,
          "mean_G_state": 1.0671927476876544,
          "mean_KL_future": 0.0027024357813445476,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0764911725291644,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 64
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0019723033748029195,
          "max_future_KL": 0.005496904253959656,
          "max_propagation_gain": 1.1499480397313608,
          "mean_G_state": 1.0815072950935538,
          "mean_KL_future": 0.0012490470992916336,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0853188163529282,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 64
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.010593896628260779,
          "max_future_KL": 0.036099873483181,
          "max_propagation_gain": 1.723735178749933,
          "mean_G_state": 1.416114747407283,
          "mean_KL_future": 0.007340213234956927,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4200709323543397,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.004604269894640823,
          "max_future_KL": 0.01454610750079155,
          "max_propagation_gain": 1.3829561552052827,
          "mean_G_state": 1.146743460455202,
          "mean_KL_future": 0.0028927648907179915,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1883968488097256,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 8
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 0.0004058465659724675,
          "max_future_KL": 0.0019528904231265187,
          "max_propagation_gain": 0.9634503506676478,
          "mean_G_state": 0.7580507248986523,
          "mean_KL_future": 0.0002537839835943245,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6458587429226331,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0020543902885949094,
          "max_future_KL": 0.0073154279962182045,
          "max_propagation_gain": 0.9724788075629001,
          "mean_G_state": 0.8763432041203232,
          "mean_KL_future": 0.0012966501127779084,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8281188211532673,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.10902293412317915,
        "max": 1.454535817433253,
        "mean": 1.20889267172889,
        "median": 1.1969875442200064,
        "min": 0.9906426512798653,
        "p10": 1.060745550328246,
        "p25": 1.100289746438744,
        "p75": 1.2655498963516911,
        "p90": 1.4069699936486924,
        "std": 0.13179702611200184
      },
      "persistence_separation_ratio": 1.4682749784219271,
      "persistence_spread": 0.4638931661533877,
      "problem_id": "test/intermediate_algebra/207.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0023284934532540546,
          "max_future_KL": 0.007713437546044588,
          "max_propagation_gain": 1.19075868631069,
          "persistence_score": 0.9906426512798653
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0014183962250655213,
          "max_future_KL": 0.0024368467275053263,
          "max_propagation_gain": 1.1121445075358307,
          "persistence_score": 1.0449999281273272
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0037641337156856026,
          "max_future_KL": 0.013246589340269566,
          "max_propagation_gain": 1.153466180937131,
          "persistence_score": 1.0764911725291644
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0019723033748029195,
          "max_future_KL": 0.005496904253959656,
          "max_propagation_gain": 1.1499480397313608,
          "persistence_score": 1.0853188163529282
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.005048924123309461,
          "max_future_KL": 0.016587531194090843,
          "max_propagation_gain": 1.2111654982295632,
          "persistence_score": 1.1052800564673493
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.004658662002593417,
          "max_future_KL": 0.012856886722147465,
          "max_propagation_gain": 1.3679807252457301,
          "persistence_score": 1.1653854521553602
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.014228682101730072,
          "max_future_KL": 0.05921326205134392,
          "max_propagation_gain": 1.5295798090654062,
          "persistence_score": 1.1778625238979692
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.004604269894640823,
          "max_future_KL": 0.01454610750079155,
          "max_propagation_gain": 1.3829561552052827,
          "persistence_score": 1.1883968488097256
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.004593040514998847,
          "max_future_KL": 0.015336363576352596,
          "max_propagation_gain": 1.3476086154592248,
          "persistence_score": 1.2055782396302872
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0019254693235996711,
          "max_future_KL": 0.00798010639846325,
          "max_propagation_gain": 1.3184230441918485,
          "persistence_score": 1.2198557279222613
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.0027105965996000237,
          "max_future_KL": 0.007980825379490852,
          "max_propagation_gain": 1.4738957572881528,
          "persistence_score": 1.238203076431087
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.00439733862789069,
          "max_future_KL": 0.012094114907085896,
          "max_propagation_gain": 1.5707962993670503,
          "persistence_score": 1.2432035680392435
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0011474000436564324,
          "max_future_KL": 0.004565841052681208,
          "max_propagation_gain": 1.4608708138975568,
          "persistence_score": 1.3325888812890336
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.009683472952858097,
          "max_future_KL": 0.038731709122657776,
          "max_propagation_gain": 1.6564893192940473,
          "persistence_score": 1.3938690549430448
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.010593896628260779,
          "max_future_KL": 0.036099873483181,
          "max_propagation_gain": 1.723735178749933,
          "persistence_score": 1.4200709323543397
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.005245569542174167,
          "max_future_KL": 0.016197694465517998,
          "max_propagation_gain": 1.696477160075966,
          "persistence_score": 1.454535817433253
        }
      ],
      "random_persistent_amplification_fraction": 0.9375,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": 0.6382352941176451,
      "spearman_persistence_future_KL": 0.3617647058823519,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 3.389191489549053,
        "max": 0.3129742144374177,
        "mean": 0.0221586542165376,
        "median": 0.0024162832563888515,
        "min": 0.0011216518609330705,
        "p10": 0.001147433639562223,
        "p25": 0.0017929399936747361,
        "p75": 0.004385184622879024,
        "p90": 0.004828876606188715,
        "std": 0.07509992229393866
      },
      "future_KL_spread": 0.31185256257648464,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 3.4833318314400494,
        "max": 1.5365580320358276,
        "mean": 0.10604818270076066,
        "median": 0.01106757577508688,
        "min": 0.0026116962544620037,
        "p10": 0.0030739381909370422,
        "p25": 0.005946473102085292,
        "p75": 0.015746523160487413,
        "p90": 0.020565739832818508,
        "std": 0.36940101047141294
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.19887315487164875,
        "max": 2.326518552595575,
        "mean": 1.6031691797535343,
        "median": 1.4967048028766194,
        "min": 1.2549780584097219,
        "p10": 1.2932020180717276,
        "p25": 1.3570128670673056,
        "p75": 1.7713728918127658,
        "p90": 2.0646389971817136,
        "std": 0.3188273125707776
      },
      "max_gain_spread": 1.0715404941858533,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0007853197897929931,
          "max_future_KL": 0.0017042330000549555,
          "max_propagation_gain": 0.9698329518943325,
          "mean_G_state": 0.8543788857114537,
          "mean_KL_future": 0.000527185897453819,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7926874237568003,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.0020312374057539274,
          "max_future_KL": 0.0066831992007792,
          "max_propagation_gain": 1.3618976127445848,
          "mean_G_state": 1.196646664426057,
          "mean_KL_future": 0.0021565768442428634,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1272316137818812,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.004848560923710466,
          "max_future_KL": 0.01301117055118084,
          "max_propagation_gain": 1.7519093155015522,
          "mean_G_state": 1.4263144703568378,
          "mean_KL_future": 0.0033436060111853294,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4175865513844683,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.004505218286067247,
          "max_future_KL": 0.014743829146027565,
          "max_propagation_gain": 1.9483877212567637,
          "mean_G_state": 1.5966630260482855,
          "mean_KL_future": 0.004991824891476426,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5443755066373064,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0019752918393351137,
          "max_future_KL": 0.01901603862643242,
          "max_propagation_gain": 1.4281564774346016,
          "mean_G_state": 1.1590865198698597,
          "mean_KL_future": 0.0051803211237029245,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0723922130679147,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.0011216518609330705,
          "max_future_KL": 0.0031222589313983917,
          "max_propagation_gain": 1.8297636207464065,
          "mean_G_state": 1.5471776695469364,
          "mean_KL_future": 0.0010685005787607338,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5249570339624658,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0011326687395921907,
          "max_future_KL": 0.0030256174504756927,
          "max_propagation_gain": 1.2785689383858445,
          "mean_G_state": 1.1005370443393596,
          "mean_KL_future": 0.000946351262882672,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0599743222686413,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0014790657047342392,
          "max_future_KL": 0.0037362948060035706,
          "max_propagation_gain": 1.2549780584097219,
          "mean_G_state": 1.1190962485584088,
          "mean_KL_future": 0.001358656027946381,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0559215460954465,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0036359510326292367,
          "max_future_KL": 0.00834354106336832,
          "max_propagation_gain": 1.4197542805550494,
          "mean_G_state": 1.2443646387475402,
          "mean_KL_future": 0.003416003660390743,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1841826125189638,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0028013291070237757,
          "max_future_KL": 0.022115441039204597,
          "max_propagation_gain": 1.7206386366618225,
          "mean_G_state": 1.4614468517814885,
          "mean_KL_future": 0.004722911056887824,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.4002549491797152,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0019907183479517697,
          "max_future_KL": 0.010353031568229198,
          "max_propagation_gain": 1.5670676132043577,
          "mean_G_state": 1.3681904998313876,
          "mean_KL_future": 0.002675456318684155,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3110064541863484,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.003828430792782456,
          "max_future_KL": 0.011782119981944561,
          "max_propagation_gain": 2.1808902731066637,
          "mean_G_state": 1.815360309130702,
          "mean_KL_future": 0.0026778433893923648,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.8309884572106825,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.004809192288666964,
          "max_future_KL": 0.01551874727010727,
          "max_propagation_gain": 1.3078350977576105,
          "mean_G_state": 1.1875173676359936,
          "mean_KL_future": 0.003527331951772794,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2089328496449547,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.0011621985395322554,
          "max_future_KL": 0.0026116962544620037,
          "max_propagation_gain": 1.3423586300354677,
          "mean_G_state": 1.2048602702690316,
          "mean_KL_future": 0.0010175410097872373,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.189071769181175,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.3129742144374177,
          "max_future_KL": 1.5365580320358276,
          "max_propagation_gain": 2.326518552595575,
          "mean_G_state": 1.5065145045926651,
          "mean_KL_future": 0.197407981341712,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.5488202823148018,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.0018975647566549015,
          "max_future_KL": 0.016429850831627846,
          "max_propagation_gain": 1.5652531283186373,
          "mean_G_state": 1.3513953517197177,
          "mean_KL_future": 0.003540636086199811,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2994243906880747,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.004345173401816282,
          "max_future_KL": 0.00972005445510149,
          "max_propagation_gain": 1.3667289193418886,
          "mean_G_state": 1.2079976308715432,
          "mean_KL_future": 0.003330943480705173,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1723521568072068,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 0.0005305560893668293,
          "max_future_KL": 0.002608784008771181,
          "max_propagation_gain": 0.9584948816209762,
          "mean_G_state": 0.6965887273649691,
          "mean_KL_future": 0.00036391957541798137,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.5711107798309893,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0007982311743454318,
          "max_future_KL": 0.0015232376754283905,
          "max_propagation_gain": 0.969701418606842,
          "mean_G_state": 0.8539455325065588,
          "mean_KL_future": 0.0005264652649237256,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7920942101148031,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.16270027378368512,
        "max": 1.8309884572106825,
        "mean": 1.3092170443081277,
        "median": 1.2541786201665146,
        "min": 1.0559215460954465,
        "p10": 1.0661832676682779,
        "p25": 1.1610720210508754,
        "p75": 1.4444291720289675,
        "p90": 1.5465978944760541,
        "std": 0.21300997155136212
      },
      "persistence_separation_ratio": 1.7340194107976297,
      "persistence_spread": 0.775066911115236,
      "problem_id": "test/geometry/477.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0014790657047342392,
          "max_future_KL": 0.0037362948060035706,
          "max_propagation_gain": 1.2549780584097219,
          "persistence_score": 1.0559215460954465
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0011326687395921907,
          "max_future_KL": 0.0030256174504756927,
          "max_propagation_gain": 1.2785689383858445,
          "persistence_score": 1.0599743222686413
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0019752918393351137,
          "max_future_KL": 0.01901603862643242,
          "max_propagation_gain": 1.4281564774346016,
          "persistence_score": 1.0723922130679147
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.0020312374057539274,
          "max_future_KL": 0.0066831992007792,
          "max_propagation_gain": 1.3618976127445848,
          "persistence_score": 1.1272316137818812
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.004345173401816282,
          "max_future_KL": 0.00972005445510149,
          "max_propagation_gain": 1.3667289193418886,
          "persistence_score": 1.1723521568072068
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0036359510326292367,
          "max_future_KL": 0.00834354106336832,
          "max_propagation_gain": 1.4197542805550494,
          "persistence_score": 1.1841826125189638
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.0011621985395322554,
          "max_future_KL": 0.0026116962544620037,
          "max_propagation_gain": 1.3423586300354677,
          "persistence_score": 1.189071769181175
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.004809192288666964,
          "max_future_KL": 0.01551874727010727,
          "max_propagation_gain": 1.3078350977576105,
          "persistence_score": 1.2089328496449547
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.0018975647566549015,
          "max_future_KL": 0.016429850831627846,
          "max_propagation_gain": 1.5652531283186373,
          "persistence_score": 1.2994243906880747
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0019907183479517697,
          "max_future_KL": 0.010353031568229198,
          "max_propagation_gain": 1.5670676132043577,
          "persistence_score": 1.3110064541863484
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0028013291070237757,
          "max_future_KL": 0.022115441039204597,
          "max_propagation_gain": 1.7206386366618225,
          "persistence_score": 1.4002549491797152
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.004848560923710466,
          "max_future_KL": 0.01301117055118084,
          "max_propagation_gain": 1.7519093155015522,
          "persistence_score": 1.4175865513844683
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.0011216518609330705,
          "max_future_KL": 0.0031222589313983917,
          "max_propagation_gain": 1.8297636207464065,
          "persistence_score": 1.5249570339624658
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.004505218286067247,
          "max_future_KL": 0.014743829146027565,
          "max_propagation_gain": 1.9483877212567637,
          "persistence_score": 1.5443755066373064
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.3129742144374177,
          "max_future_KL": 1.5365580320358276,
          "max_propagation_gain": 2.326518552595575,
          "persistence_score": 1.5488202823148018
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.003828430792782456,
          "max_future_KL": 0.011782119981944561,
          "max_propagation_gain": 2.1808902731066637,
          "persistence_score": 1.8309884572106825
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.42941176470588105,
      "spearman_persistence_future_KL": 0.47058823529411625,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "2/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "middle_50",
        "empirical_percentile_vs_random": 0.3125,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "6/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.9727933706991437,
        "max": 0.01476389678398391,
        "mean": 0.004102164317300084,
        "median": 0.0023691694719303767,
        "min": 0.00034537270153123245,
        "p10": 0.0006305359862835759,
        "p25": 0.00135031411040778,
        "p75": 0.00546676238047894,
        "p90": 0.009201301404307572,
        "std": 0.003990558254360894
      },
      "future_KL_spread": 0.014418524082452677,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 2.0922044815040963,
        "max": 5.320837497711182,
        "mean": 0.7806727510178462,
        "median": 0.1056555025279522,
        "min": 0.02180125005543232,
        "p10": 0.03857444226741791,
        "p25": 0.08085788786411285,
        "p75": 0.2983504682779312,
        "p90": 2.734097421169281,
        "std": 1.6333270282697614
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.1649623181841847,
        "max": 2.673233778026601,
        "mean": 1.9321593708536353,
        "median": 1.9470058503572778,
        "min": 1.4531314660227512,
        "p10": 1.5839235180054438,
        "p25": 1.674521331748893,
        "p75": 2.07750981542961,
        "p90": 2.3248075406518485,
        "std": 0.3187334889174765
      },
      "max_gain_spread": 1.2201023120038499,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.00039015607628130057,
          "max_future_KL": 0.0019316282123327255,
          "max_propagation_gain": 0.9919237201148677,
          "mean_G_state": 0.8612021487420595,
          "mean_KL_future": 0.0005903566689759998,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7939422235477412,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.00034537270153123245,
          "max_future_KL": 0.6442564725875854,
          "max_propagation_gain": 1.8572548402293847,
          "mean_G_state": 1.4223961668322496,
          "mean_KL_future": 0.08418802580243323,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2487223081266294,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.004810331821340696,
          "max_future_KL": 0.023917190730571747,
          "max_propagation_gain": 1.7088867555223555,
          "mean_G_state": 1.4341099332885225,
          "mean_KL_future": 0.00503531841322058,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3588889771844372,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.01476389678398391,
          "max_future_KL": 0.07365935295820236,
          "max_propagation_gain": 2.118545108875836,
          "mean_G_state": 1.7554546892430034,
          "mean_KL_future": 0.010183071848402392,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.7024622101826363,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0013151590852416462,
          "max_future_KL": 0.02180125005543232,
          "max_propagation_gain": 2.673233778026601,
          "mean_G_state": 2.298458451342028,
          "mean_KL_future": 0.004950813259369191,
          "mean_Top1_future": 1.0,
          "persistence_score": 2.307507366020239,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.0020784338007558746,
          "max_future_KL": 5.320837497711182,
          "max_propagation_gain": 1.9877769395926819,
          "mean_G_state": 1.4606003870113182,
          "mean_KL_future": 0.6762818664430448,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.259625436213838,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.004649770965954758,
          "max_future_KL": 0.08454343676567078,
          "max_propagation_gain": 1.5904773929595921,
          "mean_G_state": 1.293243114058475,
          "mean_KL_future": 0.016980477817220674,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.2061888659620468,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.0008217878637537979,
          "max_future_KL": 0.5371218323707581,
          "max_propagation_gain": 2.0180501646420113,
          "mean_G_state": 1.4987598513624683,
          "mean_KL_future": 0.07148431808295896,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.3026916141810871,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0015859687257488986,
          "max_future_KL": 0.18515127897262573,
          "max_propagation_gain": 2.398136987045032,
          "mean_G_state": 1.931791162970912,
          "mean_KL_future": 0.025644683193178253,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.8293214587407611,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.007740406486561824,
          "max_future_KL": 4.823938369750977,
          "max_propagation_gain": 2.00601286609813,
          "mean_G_state": 1.6232827942059609,
          "mean_KL_future": 0.6127290194152053,
          "mean_Top1_future": 0.75,
          "persistence_score": 1.5065222553386106,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0015848106969428954,
          "max_future_KL": 0.11498808115720749,
          "max_propagation_gain": 1.5773696430512956,
          "mean_G_state": 1.2389583469437173,
          "mean_KL_future": 0.01624585791632427,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.1224019756100387,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0033792180609907518,
          "max_future_KL": 0.09701504558324814,
          "max_propagation_gain": 2.063831384280868,
          "mean_G_state": 1.6420036186392233,
          "mean_KL_future": 0.01593989993241962,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.532687668186243,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0013620324521298245,
          "max_future_KL": 0.08325739949941635,
          "max_propagation_gain": 1.6969777875322403,
          "mean_G_state": 1.3426531789825098,
          "mean_KL_future": 0.012110133464446449,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.220503041607305,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.007436054057893671,
          "max_future_KL": 0.21876001358032227,
          "max_propagation_gain": 2.2514780942586645,
          "mean_G_state": 1.824856980404228,
          "mean_KL_future": 0.04023226760763521,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.800376627029345,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0026599051431048792,
          "max_future_KL": 0.09398914128541946,
          "max_propagation_gain": 1.9062347611218735,
          "mean_G_state": 1.4353873497251082,
          "mean_KL_future": 0.013862609463117792,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2422944196168832,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.0004392841088133537,
          "max_future_KL": 0.11429595947265625,
          "max_propagation_gain": 1.6071519643988512,
          "mean_G_state": 1.3145303189032607,
          "mean_KL_future": 0.01606200802503821,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2079317560225438,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.010662196322053319,
          "max_future_KL": 0.05323169380426407,
          "max_propagation_gain": 1.4531314660227512,
          "mean_G_state": 1.2347077675510179,
          "mean_KL_future": 0.008511299021010998,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.1578500631068034,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 2.6453906880918864e-05,
          "max_future_KL": 0.0017090394394472241,
          "max_propagation_gain": 0.9658261411114162,
          "mean_G_state": 0.7746443461962027,
          "mean_KL_future": 0.00023333789073201605,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6800320505125919,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0015536572719270225,
          "max_future_KL": 0.007748868316411972,
          "max_propagation_gain": 0.9921909054741183,
          "mean_G_state": 0.8629191846890741,
          "mean_KL_future": 0.0013056818093346578,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.7966333608767545,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.2190088805089653,
        "max": 2.307507366020239,
        "mean": 1.4378735026955904,
        "median": 1.2811585251974624,
        "min": 1.1224019756100387,
        "p10": 1.182019464534425,
        "p25": 1.2173602202111147,
        "p75": 1.5751313036853412,
        "p90": 1.814849042885053,
        "std": 0.314907066139085
      },
      "persistence_separation_ratio": 2.055865381708746,
      "persistence_spread": 1.1851053904102005,
      "problem_id": "test/algebra/1214.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0015848106969428954,
          "max_future_KL": 0.11498808115720749,
          "max_propagation_gain": 1.5773696430512956,
          "persistence_score": 1.1224019756100387
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.010662196322053319,
          "max_future_KL": 0.05323169380426407,
          "max_propagation_gain": 1.4531314660227512,
          "persistence_score": 1.1578500631068034
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.004649770965954758,
          "max_future_KL": 0.08454343676567078,
          "max_propagation_gain": 1.5904773929595921,
          "persistence_score": 1.2061888659620468
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.0004392841088133537,
          "max_future_KL": 0.11429595947265625,
          "max_propagation_gain": 1.6071519643988512,
          "persistence_score": 1.2079317560225438
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0013620324521298245,
          "max_future_KL": 0.08325739949941635,
          "max_propagation_gain": 1.6969777875322403,
          "persistence_score": 1.220503041607305
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0026599051431048792,
          "max_future_KL": 0.09398914128541946,
          "max_propagation_gain": 1.9062347611218735,
          "persistence_score": 1.2422944196168832
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.00034537270153123245,
          "max_future_KL": 0.6442564725875854,
          "max_propagation_gain": 1.8572548402293847,
          "persistence_score": 1.2487223081266294
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.0020784338007558746,
          "max_future_KL": 5.320837497711182,
          "max_propagation_gain": 1.9877769395926819,
          "persistence_score": 1.259625436213838
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.0008217878637537979,
          "max_future_KL": 0.5371218323707581,
          "max_propagation_gain": 2.0180501646420113,
          "persistence_score": 1.3026916141810871
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.004810331821340696,
          "max_future_KL": 0.023917190730571747,
          "max_propagation_gain": 1.7088867555223555,
          "persistence_score": 1.3588889771844372
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.007740406486561824,
          "max_future_KL": 4.823938369750977,
          "max_propagation_gain": 2.00601286609813,
          "persistence_score": 1.5065222553386106
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0033792180609907518,
          "max_future_KL": 0.09701504558324814,
          "max_propagation_gain": 2.063831384280868,
          "persistence_score": 1.532687668186243
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.01476389678398391,
          "max_future_KL": 0.07365935295820236,
          "max_propagation_gain": 2.118545108875836,
          "persistence_score": 1.7024622101826363
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.007436054057893671,
          "max_future_KL": 0.21876001358032227,
          "max_propagation_gain": 2.2514780942586645,
          "persistence_score": 1.800376627029345
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0015859687257488986,
          "max_future_KL": 0.18515127897262573,
          "max_propagation_gain": 2.398136987045032,
          "persistence_score": 1.8293214587407611
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0013151590852416462,
          "max_future_KL": 0.02180125005543232,
          "max_propagation_gain": 2.673233778026601,
          "persistence_score": 2.307507366020239
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": 0.03529411764705872,
      "spearman_persistence_future_KL": 0.14999999999999955,
      "t0": 64,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": false,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 1.0957603695619953,
        "max": 0.0001785776567853503,
        "mean": 3.8861081520261136e-05,
        "median": 2.4801089197068472e-05,
        "min": 2.5646222613140423e-06,
        "p10": 3.988679344546497e-06,
        "p25": 1.1213064924964213e-05,
        "p75": 5.1466898814367394e-05,
        "p90": 6.934996538034976e-05,
        "std": 4.258243414398054e-05
      },
      "future_KL_spread": 0.00017601303452403626,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.5353493348643255,
        "max": 0.01025521568953991,
        "mean": 0.0018765451263789146,
        "median": 0.0006752421904820949,
        "min": 4.0215607441496104e-05,
        "p10": 9.848746412899345e-05,
        "p25": 0.00015891007569734938,
        "p75": 0.001583888049935922,
        "p90": 0.006000118795782328,
        "std": 0.002881152313164107
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.08991298946910443,
        "max": 1.6004816076410588,
        "mean": 1.281255345509279,
        "median": 1.2673034571640511,
        "min": 1.1153952755292802,
        "p10": 1.1743449413607259,
        "p25": 1.230635318233987,
        "p75": 1.2773481661215307,
        "p90": 1.4206280544484682,
        "std": 0.11520149838809948
      },
      "max_gain_spread": 0.48508633211177865,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 2.536480154891052e-06,
          "max_future_KL": 8.727575732336845e-06,
          "max_propagation_gain": 0.9930615139955264,
          "mean_G_state": 0.8977021172482756,
          "mean_KL_future": 1.5888154822896183e-06,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8459241225043271,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 1.3659665992626913e-05,
          "max_future_KL": 4.0215607441496104e-05,
          "max_propagation_gain": 1.2597934313072592,
          "mean_G_state": 1.1339585231503537,
          "mean_KL_future": 8.550039546473265e-06,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1154849532276774,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 4.894801899801848e-06,
          "max_future_KL": 0.0015838267281651497,
          "max_propagation_gain": 1.247446373017837,
          "mean_G_state": 1.114793769921251,
          "mean_KL_future": 0.00020104140638377288,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1202404593775201,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 6.88934539692987e-05,
          "max_future_KL": 0.0006752978661097586,
          "max_propagation_gain": 1.2065442074415114,
          "mean_G_state": 1.0619521210771674,
          "mean_KL_future": 0.00012749782356796802,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0564665923798446,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 6.980647679140084e-05,
          "max_future_KL": 0.001584072015248239,
          "max_propagation_gain": 1.2386656884981457,
          "mean_G_state": 1.0976381435548233,
          "mean_KL_future": 0.0002416362306252584,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0723152225085737,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 1.824607035700865e-05,
          "max_future_KL": 0.00014961363922338933,
          "max_propagation_gain": 1.2713647186539327,
          "mean_G_state": 1.1007183735293251,
          "mean_KL_future": 3.0164202200033685e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0968220633253332,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 2.5646222613140423e-06,
          "max_future_KL": 0.0001495431934017688,
          "max_propagation_gain": 1.269713883627434,
          "mean_G_state": 1.1119205426576508,
          "mean_KL_future": 2.0366010364536447e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1113322454719812,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 8.84712996764847e-06,
          "max_future_KL": 0.0072084092535078526,
          "max_propagation_gain": 1.268695193421185,
          "mean_G_state": 1.136165898968938,
          "mean_KL_future": 0.0009065859772348084,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1291605732156675,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 4.340665445141667e-05,
          "max_future_KL": 0.0002146460465155542,
          "max_propagation_gain": 1.2659117209069173,
          "mean_G_state": 1.1587264780783213,
          "mean_KL_future": 4.595847830879407e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1592047931906042,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 1.2001709910736125e-05,
          "max_future_KL": 4.74317348562181e-05,
          "max_propagation_gain": 1.2944948224852086,
          "mean_G_state": 1.1185033899522019,
          "mean_KL_future": 7.5365126033510865e-06,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.102143875326393,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 1.8999668819041916e-05,
          "max_future_KL": 0.0006751915207132697,
          "max_propagation_gain": 1.1865076584156642,
          "mean_G_state": 1.0732628675528342,
          "mean_KL_future": 9.627998892924872e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.050140151201913,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 6.417195270671527e-05,
          "max_future_KL": 0.01025521568953991,
          "max_propagation_gain": 1.6004816076410588,
          "mean_G_state": 1.2553426288716718,
          "mean_KL_future": 0.0013220184799784818,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2253231830297588,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 4.723188085025143e-05,
          "max_future_KL": 0.0002283583744429052,
          "max_propagation_gain": 1.1153952755292802,
          "mean_G_state": 1.0427602563176406,
          "mean_KL_future": 4.981685421406867e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0367056274993787,
          "tau_of_max_future_KL": 32,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 3.060250957509503e-05,
          "max_future_KL": 0.00016200888785533607,
          "max_propagation_gain": 1.3463624169029893,
          "mean_G_state": 1.1784036683787322,
          "mean_KL_future": 3.94184253734442e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1678579162937321,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 3.0825567892911466e-06,
          "max_future_KL": 0.0006752928602509201,
          "max_propagation_gain": 1.1621822243057878,
          "mean_G_state": 1.0651827219624974,
          "mean_KL_future": 8.634952304270582e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0472991528705824,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.0001785776567853503,
          "max_future_KL": 0.004791828338056803,
          "max_propagation_gain": 1.4948936919939473,
          "mean_G_state": 1.2343253319848495,
          "mean_KL_future": 0.000710630709760407,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2330980932165516,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 3.679049319718075e-05,
          "max_future_KL": 0.0015837702667340636,
          "max_propagation_gain": 1.271632614000305,
          "mean_G_state": 1.122518396862488,
          "mean_KL_future": 0.00022098426679040273,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1264121858435834,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 8
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 2.3404252793568503e-07,
          "max_future_KL": 0.00014957570238038898,
          "max_propagation_gain": 0.9853838009445273,
          "mean_G_state": 0.714616730711498,
          "mean_KL_future": 1.883951895557262e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.5605169221205799,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 2.143026144985072e-06,
          "max_future_KL": 8.90646333573386e-06,
          "max_propagation_gain": 0.9930422251037011,
          "mean_G_state": 0.903752579460065,
          "mean_KL_future": 1.3521000017391316e-06,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8549984764602678,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.0510492116352856,
        "max": 1.2330980932165516,
        "mean": 1.1156254429986934,
        "median": 1.1134085993498293,
        "min": 1.0367056274993787,
        "p10": 1.0487196520362478,
        "p25": 1.0683530649763915,
        "p75": 1.1366716282094016,
        "p90": 1.1965905496617455,
        "std": 0.0569517993454006
      },
      "persistence_separation_ratio": 1.189438988760675,
      "persistence_spread": 0.19639246571717295,
      "problem_id": "test/counting_and_probability/119.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 4.723188085025143e-05,
          "max_future_KL": 0.0002283583744429052,
          "max_propagation_gain": 1.1153952755292802,
          "persistence_score": 1.0367056274993787
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 3.0825567892911466e-06,
          "max_future_KL": 0.0006752928602509201,
          "max_propagation_gain": 1.1621822243057878,
          "persistence_score": 1.0472991528705824
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 1.8999668819041916e-05,
          "max_future_KL": 0.0006751915207132697,
          "max_propagation_gain": 1.1865076584156642,
          "persistence_score": 1.050140151201913
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 6.88934539692987e-05,
          "max_future_KL": 0.0006752978661097586,
          "max_propagation_gain": 1.2065442074415114,
          "persistence_score": 1.0564665923798446
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 6.980647679140084e-05,
          "max_future_KL": 0.001584072015248239,
          "max_propagation_gain": 1.2386656884981457,
          "persistence_score": 1.0723152225085737
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 1.824607035700865e-05,
          "max_future_KL": 0.00014961363922338933,
          "max_propagation_gain": 1.2713647186539327,
          "persistence_score": 1.0968220633253332
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 1.2001709910736125e-05,
          "max_future_KL": 4.74317348562181e-05,
          "max_propagation_gain": 1.2944948224852086,
          "persistence_score": 1.102143875326393
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 2.5646222613140423e-06,
          "max_future_KL": 0.0001495431934017688,
          "max_propagation_gain": 1.269713883627434,
          "persistence_score": 1.1113322454719812
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 1.3659665992626913e-05,
          "max_future_KL": 4.0215607441496104e-05,
          "max_propagation_gain": 1.2597934313072592,
          "persistence_score": 1.1154849532276774
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 4.894801899801848e-06,
          "max_future_KL": 0.0015838267281651497,
          "max_propagation_gain": 1.247446373017837,
          "persistence_score": 1.1202404593775201
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 3.679049319718075e-05,
          "max_future_KL": 0.0015837702667340636,
          "max_propagation_gain": 1.271632614000305,
          "persistence_score": 1.1264121858435834
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 8.84712996764847e-06,
          "max_future_KL": 0.0072084092535078526,
          "max_propagation_gain": 1.268695193421185,
          "persistence_score": 1.1291605732156675
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 4.340665445141667e-05,
          "max_future_KL": 0.0002146460465155542,
          "max_propagation_gain": 1.2659117209069173,
          "persistence_score": 1.1592047931906042
        },
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 3.060250957509503e-05,
          "max_future_KL": 0.00016200888785533607,
          "max_propagation_gain": 1.3463624169029893,
          "persistence_score": 1.1678579162937321
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 6.417195270671527e-05,
          "max_future_KL": 0.01025521568953991,
          "max_propagation_gain": 1.6004816076410588,
          "persistence_score": 1.2253231830297588
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.0001785776567853503,
          "max_future_KL": 0.004791828338056803,
          "max_propagation_gain": 1.4948936919939473,
          "persistence_score": 1.2330980932165516
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.1323529411764702,
      "spearman_persistence_future_KL": 0.18235294117647005,
      "t0": 128,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "2/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0625,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "2/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.8428719919900015,
        "max": 0.006424327645413541,
        "mean": 0.0016735908800302344,
        "median": 0.001461113463540009,
        "min": 0.00011811420051088817,
        "p10": 0.0005748807240070874,
        "p25": 0.0007575181338370208,
        "p75": 0.002080069928628858,
        "p90": 0.0024098696297953025,
        "std": 0.0014106228796702553
      },
      "future_KL_spread": 0.006306213444902653,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 1.8697943063695575,
        "max": 0.8233904242515564,
        "mean": 0.13946329675673041,
        "median": 0.01092069549486041,
        "min": 0.0022905003279447556,
        "p10": 0.003428146825172007,
        "p25": 0.007390680257230997,
        "p75": 0.04849391197785735,
        "p90": 0.5861110985279083,
        "std": 0.2607676782251323
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.17082909503968538,
        "max": 2.527252862335212,
        "mean": 1.8196049416166025,
        "median": 1.7033264865975268,
        "min": 1.375482136650335,
        "p10": 1.503372177147775,
        "p25": 1.6135060017046663,
        "p75": 2.0090009184272333,
        "p90": 2.2295768401692087,
        "std": 0.3108414655062746
      },
      "max_gain_spread": 1.151770725684877,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.00031869251429288424,
          "max_future_KL": 0.0011268503731116652,
          "max_propagation_gain": 1.1744941441145698,
          "mean_G_state": 0.961015625581965,
          "mean_KL_future": 0.00020861765622293138,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9128635760230626,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.0011911925116905309,
          "max_future_KL": 0.49282097816467285,
          "max_propagation_gain": 1.9891984723854357,
          "mean_G_state": 1.4996942462885439,
          "mean_KL_future": 0.06993265136857385,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2558794050543796,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.0017828539011659928,
          "max_future_KL": 0.00840422511100769,
          "max_propagation_gain": 1.9869730910181764,
          "mean_G_state": 1.5625630927811835,
          "mean_KL_future": 0.0015043888064285582,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.5271170971080807,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.0006109588744273253,
          "max_future_KL": 0.003051220439374447,
          "max_propagation_gain": 1.7603882095916916,
          "mean_G_state": 1.271753093914729,
          "mean_KL_future": 0.0005930081246368069,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2128362590319814,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.002216181303600706,
          "max_future_KL": 0.013304311782121658,
          "max_propagation_gain": 2.1917701718238463,
          "mean_G_state": 1.5974542569117676,
          "mean_KL_future": 0.0031668213936022482,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.5070915486546954,
          "tau_of_max_future_KL": 2,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.002603557955989899,
          "max_future_KL": 0.012544317170977592,
          "max_propagation_gain": 1.5935415969805269,
          "mean_G_state": 1.2126731455752315,
          "mean_KL_future": 0.0017942214781994092,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1556753674672162,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0005388025735868496,
          "max_future_KL": 0.0022905003279447556,
          "max_propagation_gain": 1.6988837158746428,
          "mean_G_state": 1.2990437774193517,
          "mean_KL_future": 0.0004697574529615167,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3186849232361908,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.001091923623198454,
          "max_future_KL": 0.12390223890542984,
          "max_propagation_gain": 2.068408256552627,
          "mean_G_state": 1.5702565569390528,
          "mean_KL_future": 0.01979555859745119,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3740828186016945,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0020722484958977994,
          "max_future_KL": 0.010244989767670631,
          "max_propagation_gain": 1.6883483214857515,
          "mean_G_state": 1.3636733231754685,
          "mean_KL_future": 0.0016130717456091515,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3377520079684742,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.0009904310509526227,
          "max_future_KL": 0.0047997236251831055,
          "max_propagation_gain": 1.4132027573150232,
          "mean_G_state": 1.157951872945591,
          "mean_KL_future": 0.0007151159136311591,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.138290470160714,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0006390651364057476,
          "max_future_KL": 0.6794012188911438,
          "max_propagation_gain": 2.527252862335212,
          "mean_G_state": 1.5069023501237149,
          "mean_KL_future": 0.0853833525663692,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.4568608949565989,
          "tau_of_max_future_KL": 4,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.00011811420051088817,
          "max_future_KL": 0.011327050626277924,
          "max_propagation_gain": 1.7077692573204106,
          "mean_G_state": 1.3193360464496984,
          "mean_KL_future": 0.0031715958653348864,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1325019646125303,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0018662256991177628,
          "max_future_KL": 0.023357803001999855,
          "max_propagation_gain": 1.62995379575655,
          "mean_G_state": 1.2716207196914762,
          "mean_KL_future": 0.006144098958563049,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1303038272308963,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.002103534226822035,
          "max_future_KL": 0.010514340363442898,
          "max_propagation_gain": 1.375482136650335,
          "mean_G_state": 1.106785088541521,
          "mean_KL_future": 0.0014983370935475193,
          "mean_Top1_future": 0.875,
          "persistence_score": 1.0913196519094046,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0007970024663141118,
          "max_future_KL": 0.0038050732109695673,
          "max_propagation_gain": 1.595672364981933,
          "mean_G_state": 1.249397175446727,
          "mean_KL_future": 0.0007061771118641591,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1784129198630484,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.006424327645413541,
          "max_future_KL": 0.8233904242515564,
          "max_propagation_gain": 2.2673835085145715,
          "mean_G_state": 1.5922890492010773,
          "mean_KL_future": 0.11210827330590689,
          "mean_Top1_future": 0.75,
          "persistence_score": 1.3382793606609629,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 4
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.0017310344153894873,
          "max_future_KL": 0.008254332467913628,
          "max_propagation_gain": 1.6194505472789107,
          "mean_G_state": 1.3132532679606204,
          "mean_KL_future": 0.0014326126139347917,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2916751872094179,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 4
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 7.016285496250774e-05,
          "max_future_KL": 0.0003508578520268202,
          "max_propagation_gain": 0.9785615852592452,
          "mean_G_state": 0.7708016511040068,
          "mean_KL_future": 4.41078100702208e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.6611376712722521,
          "tau_of_max_future_KL": 128,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.00014098830125774953,
          "max_future_KL": 0.0004933224408887327,
          "max_propagation_gain": 1.1728986623687145,
          "mean_G_state": 0.9573567150379172,
          "mean_KL_future": 9.655171314926214e-05,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.9073070052581149,
          "tau_of_max_future_KL": 16,
          "tau_of_max_gain": 4
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.10567281780903977,
        "max": 1.5271170971080807,
        "mean": 1.2779227314828927,
        "median": 1.2737772961318987,
        "min": 1.0913196519094046,
        "p10": 1.1314028959217133,
        "p25": 1.1513291431405905,
        "p75": 1.3472302251461459,
        "p90": 1.4819762218056471,
        "std": 0.13504169597812785
      },
      "persistence_separation_ratio": 1.3993307042852137,
      "persistence_spread": 0.43579744519867614,
      "problem_id": "test/intermediate_algebra/207.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.002103534226822035,
          "max_future_KL": 0.010514340363442898,
          "max_propagation_gain": 1.375482136650335,
          "persistence_score": 1.0913196519094046
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0018662256991177628,
          "max_future_KL": 0.023357803001999855,
          "max_propagation_gain": 1.62995379575655,
          "persistence_score": 1.1303038272308963
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.00011811420051088817,
          "max_future_KL": 0.011327050626277924,
          "max_propagation_gain": 1.7077692573204106,
          "persistence_score": 1.1325019646125303
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.0009904310509526227,
          "max_future_KL": 0.0047997236251831055,
          "max_propagation_gain": 1.4132027573150232,
          "persistence_score": 1.138290470160714
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.002603557955989899,
          "max_future_KL": 0.012544317170977592,
          "max_propagation_gain": 1.5935415969805269,
          "persistence_score": 1.1556753674672162
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0007970024663141118,
          "max_future_KL": 0.0038050732109695673,
          "max_propagation_gain": 1.595672364981933,
          "persistence_score": 1.1784129198630484
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.0006109588744273253,
          "max_future_KL": 0.003051220439374447,
          "max_propagation_gain": 1.7603882095916916,
          "persistence_score": 1.2128362590319814
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.0011911925116905309,
          "max_future_KL": 0.49282097816467285,
          "max_propagation_gain": 1.9891984723854357,
          "persistence_score": 1.2558794050543796
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.0017310344153894873,
          "max_future_KL": 0.008254332467913628,
          "max_propagation_gain": 1.6194505472789107,
          "persistence_score": 1.2916751872094179
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0005388025735868496,
          "max_future_KL": 0.0022905003279447556,
          "max_propagation_gain": 1.6988837158746428,
          "persistence_score": 1.3186849232361908
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0020722484958977994,
          "max_future_KL": 0.010244989767670631,
          "max_propagation_gain": 1.6883483214857515,
          "persistence_score": 1.3377520079684742
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.006424327645413541,
          "max_future_KL": 0.8233904242515564,
          "max_propagation_gain": 2.2673835085145715,
          "persistence_score": 1.3382793606609629
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.001091923623198454,
          "max_future_KL": 0.12390223890542984,
          "max_propagation_gain": 2.068408256552627,
          "persistence_score": 1.3740828186016945
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0006390651364057476,
          "max_future_KL": 0.6794012188911438,
          "max_propagation_gain": 2.527252862335212,
          "persistence_score": 1.4568608949565989
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.002216181303600706,
          "max_future_KL": 0.013304311782121658,
          "max_propagation_gain": 2.1917701718238463,
          "persistence_score": 1.5070915486546954
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.0017828539011659928,
          "max_future_KL": 0.00840422511100769,
          "max_propagation_gain": 1.9869730910181764,
          "persistence_score": 1.5271170971080807
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row non-truncated termination control",
      "spearman_max_gain_future_KL": -0.0558823529411763,
      "spearman_persistence_future_KL": 0.1117647058823526,
      "t0": 256,
      "valid_random_direction_count": 16
    },
    {
      "ANGLE_GATE": "PASS",
      "DIRECTION_DIVERSITY_GATE": "PASS",
      "NORM_MATCH_GATE": "PASS",
      "ORTHOGONAL_REALDIR_future_KL_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_max_gain_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "ORTHOGONAL_REALDIR_persistence_rank": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "PANEL_IMPLEMENTATION_GATE": "PASS",
      "PROTOCOL_GATE": "PASS",
      "REAL_R128_descriptive_future_KL_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_descriptive_persistence_position": {
        "bucket": "bottom_quartile",
        "empirical_percentile_vs_random": 0.0,
        "note": "N=16 random panel; percentile is coarse descriptive positioning only.",
        "rank_among_random_plus_target": "1/17"
      },
      "REAL_R128_note": "REAL_R128_PULSE is not angle-matched to the orthogonal panel; rank is descriptive only.",
      "clear_future_KL_separation": true,
      "clear_persistence_separation": true,
      "future_KL_distribution": {
        "N": 16,
        "cv": 0.6734278116139437,
        "max": 0.009545619669887628,
        "mean": 0.00330172026917237,
        "median": 0.0026257194028794116,
        "min": 0.0006947972612638686,
        "p10": 0.0013790893383527792,
        "p25": 0.0017760604751309475,
        "p75": 0.003919548280451934,
        "p90": 0.00585508382900457,
        "std": 0.002223470256103578
      },
      "future_KL_spread": 0.00885082240862376,
      "future_KL_spread_threshold": 0.0005,
      "max_future_KL_distribution": {
        "N": 16,
        "cv": 2.440979286912343,
        "max": 0.7579002976417542,
        "mean": 0.07738310508284485,
        "median": 0.012162231840193272,
        "min": 0.001811909838579595,
        "p10": 0.006038869498297572,
        "p25": 0.009206259273923934,
        "p75": 0.020268727093935013,
        "p90": 0.16476415190845728,
        "std": 0.1888905566666265
      },
      "max_gain_distribution": {
        "N": 16,
        "cv": 0.21449260960689484,
        "max": 2.37479332272094,
        "mean": 1.4690905975255424,
        "median": 1.4030349322743016,
        "min": 1.1378219605790547,
        "p10": 1.180748752977672,
        "p25": 1.2767324854963946,
        "p75": 1.5266884388955526,
        "p90": 1.8211398177547622,
        "std": 0.31510907601242055
      },
      "max_gain_spread": 1.2369713621418854,
      "metrics": {
        "FP_STATE": {
          "future_KL_score": 0.0,
          "max_future_KL": 0.0,
          "max_propagation_gain": 0.0,
          "mean_G_state": 0.0,
          "mean_KL_future": 0.0,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.0,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHOGONAL_REALDIR_PULSE": {
          "future_KL_score": 0.0004273462853984,
          "max_future_KL": 0.002124711172655225,
          "max_propagation_gain": 0.9825000043669542,
          "mean_G_state": 0.909283249682843,
          "mean_KL_future": 0.00030457649458415226,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8877830043649617,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_00": {
          "future_KL_score": 0.001506075494145609,
          "max_future_KL": 0.005830023903399706,
          "max_propagation_gain": 1.5181623465798828,
          "mean_G_state": 1.3010793503656115,
          "mean_KL_future": 0.001032063854678189,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2321205299241837,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_01": {
          "future_KL_score": 0.006559820402185323,
          "max_future_KL": 0.03100776858627796,
          "max_propagation_gain": 1.5522667158425627,
          "mean_G_state": 1.3478150733069953,
          "mean_KL_future": 0.004148001328260875,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.285941939275063,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_02": {
          "future_KL_score": 0.009545619669887628,
          "max_future_KL": 0.2985205352306366,
          "max_propagation_gain": 2.37479332272094,
          "mean_G_state": 1.8521427079520685,
          "mean_KL_future": 0.04488477036680294,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.6669525374393612,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_03": {
          "future_KL_score": 0.0012521031825599494,
          "max_future_KL": 0.006247715093195438,
          "max_propagation_gain": 1.2928782529794416,
          "mean_G_state": 1.0747081824395628,
          "mean_KL_future": 0.0007876363586092339,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0884435195489475,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_04": {
          "future_KL_score": 0.002137771292725077,
          "max_future_KL": 0.010626210831105709,
          "max_propagation_gain": 1.3735181784397221,
          "mean_G_state": 1.2211019471152804,
          "mean_KL_future": 0.0017279718109801667,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1745763651987966,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_05": {
          "future_KL_score": 0.0035178172309002775,
          "max_future_KL": 0.010319344699382782,
          "max_propagation_gain": 1.6180638557304425,
          "mean_G_state": 1.3515700379076907,
          "mean_KL_future": 0.00236182246917771,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2994480805019828,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_06": {
          "future_KL_score": 0.00224424232287177,
          "max_future_KL": 0.011158128269016743,
          "max_propagation_gain": 1.172951307686235,
          "mean_G_state": 1.0649728454817287,
          "mean_KL_future": 0.0014068789910308688,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0828024902281217,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_07": {
          "future_KL_score": 0.0022821064651001423,
          "max_future_KL": 0.00969117134809494,
          "max_propagation_gain": 1.4861385824396125,
          "mean_G_state": 1.2529496054025875,
          "mean_KL_future": 0.0016010873020777527,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.1703858421116755,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_08": {
          "future_KL_score": 0.005124741429106905,
          "max_future_KL": 0.025621438398957253,
          "max_propagation_gain": 1.301276126676442,
          "mean_G_state": 1.1083270619048982,
          "mean_KL_future": 0.0032509071546931168,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.132394041637173,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_09": {
          "future_KL_score": 0.0016995964844909394,
          "max_future_KL": 0.7579002976417542,
          "max_propagation_gain": 2.0242157797790816,
          "mean_G_state": 1.5252339862371254,
          "mean_KL_future": 0.09862275880256277,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.3452483338197343,
          "tau_of_max_future_KL": 1,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_10": {
          "future_KL_score": 0.0032603958184409974,
          "max_future_KL": 0.014639955013990402,
          "max_propagation_gain": 1.228295183047254,
          "mean_G_state": 1.1115225789245406,
          "mean_KL_future": 0.0021164477508133217,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0735788224336251,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_11": {
          "future_KL_score": 0.0029693323406586813,
          "max_future_KL": 0.0131663354113698,
          "max_propagation_gain": 1.4903460889664948,
          "mean_G_state": 1.226233211492618,
          "mean_KL_future": 0.002095027216276746,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2048634186889315,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_12": {
          "future_KL_score": 0.005150347255823817,
          "max_future_KL": 0.018484489992260933,
          "max_propagation_gain": 1.3136239745635279,
          "mean_G_state": 1.1306506863367602,
          "mean_KL_future": 0.003397904594272916,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0610893111956123,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "ORTHO_RAND_13": {
          "future_KL_score": 0.0006947972612638686,
          "max_future_KL": 0.001811909838579595,
          "max_propagation_gain": 1.1378219605790547,
          "mean_G_state": 1.1032369087112128,
          "mean_KL_future": 0.0005916539472716198,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0962620233189435,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 2
        },
        "ORTHO_RAND_14": {
          "future_KL_score": 0.003081209184585987,
          "max_future_KL": 0.015352834016084671,
          "max_propagation_gain": 1.432551686108881,
          "mean_G_state": 1.180669207840634,
          "mean_KL_future": 0.001967837956613039,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.2011664155556372,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 32
        },
        "ORTHO_RAND_15": {
          "future_KL_score": 0.00180154847201095,
          "max_future_KL": 0.0077515230514109135,
          "max_propagation_gain": 1.1885461982691088,
          "mean_G_state": 1.100700608116561,
          "mean_KL_future": 0.001174678178367472,
          "mean_Top1_future": 1.0,
          "persistence_score": 1.0721394939613669,
          "tau_of_max_future_KL": 8,
          "tau_of_max_gain": 2
        },
        "PARALLEL_PLUS_PULSE": {
          "future_KL_score": 0.00045006183858156137,
          "max_future_KL": 0.0021928264759480953,
          "max_propagation_gain": 0.9691370902071257,
          "mean_G_state": 0.7447569509432898,
          "mean_KL_future": 0.00031770254933327635,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.653675295777543,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        },
        "REAL_R128_PULSE": {
          "future_KL_score": 0.0005648308194281526,
          "max_future_KL": 0.0028234212659299374,
          "max_propagation_gain": 0.9825249236700392,
          "mean_G_state": 0.9051047675888183,
          "mean_KL_future": 0.00039021639942316,
          "mean_Top1_future": 1.0,
          "persistence_score": 0.8811297527601667,
          "tau_of_max_future_KL": 64,
          "tau_of_max_gain": 1
        }
      },
      "number_of_injections_after_t0": {
        "FP_STATE": 0,
        "ORTHOGONAL_REALDIR_PULSE": 0,
        "ORTHO_RAND_00": 0,
        "ORTHO_RAND_01": 0,
        "ORTHO_RAND_02": 0,
        "ORTHO_RAND_03": 0,
        "ORTHO_RAND_04": 0,
        "ORTHO_RAND_05": 0,
        "ORTHO_RAND_06": 0,
        "ORTHO_RAND_07": 0,
        "ORTHO_RAND_08": 0,
        "ORTHO_RAND_09": 0,
        "ORTHO_RAND_10": 0,
        "ORTHO_RAND_11": 0,
        "ORTHO_RAND_12": 0,
        "ORTHO_RAND_13": 0,
        "ORTHO_RAND_14": 0,
        "ORTHO_RAND_15": 0,
        "PARALLEL_PLUS_PULSE": 0,
        "REAL_R128_PULSE": 0
      },
      "persistence_distribution": {
        "N": 16,
        "cv": 0.12370668901754986,
        "max": 1.6669525374393612,
        "mean": 1.1992133228024473,
        "median": 1.172481103655236,
        "min": 1.0610893111956123,
        "p10": 1.072859158197496,
        "p25": 1.0870332622187409,
        "p75": 1.2455758822619036,
        "p90": 1.3223482071608585,
        "std": 0.1483507095897487
      },
      "persistence_separation_ratio": 1.5709823102067668,
      "persistence_spread": 0.6058632262437489,
      "problem_id": "test/geometry/702.json",
      "random_amplification_fraction": 1.0,
      "random_direction_table_sorted_by_persistence": [
        {
          "direction": "ORTHO_RAND_12",
          "future_KL_score": 0.005150347255823817,
          "max_future_KL": 0.018484489992260933,
          "max_propagation_gain": 1.3136239745635279,
          "persistence_score": 1.0610893111956123
        },
        {
          "direction": "ORTHO_RAND_15",
          "future_KL_score": 0.00180154847201095,
          "max_future_KL": 0.0077515230514109135,
          "max_propagation_gain": 1.1885461982691088,
          "persistence_score": 1.0721394939613669
        },
        {
          "direction": "ORTHO_RAND_10",
          "future_KL_score": 0.0032603958184409974,
          "max_future_KL": 0.014639955013990402,
          "max_propagation_gain": 1.228295183047254,
          "persistence_score": 1.0735788224336251
        },
        {
          "direction": "ORTHO_RAND_06",
          "future_KL_score": 0.00224424232287177,
          "max_future_KL": 0.011158128269016743,
          "max_propagation_gain": 1.172951307686235,
          "persistence_score": 1.0828024902281217
        },
        {
          "direction": "ORTHO_RAND_03",
          "future_KL_score": 0.0012521031825599494,
          "max_future_KL": 0.006247715093195438,
          "max_propagation_gain": 1.2928782529794416,
          "persistence_score": 1.0884435195489475
        },
        {
          "direction": "ORTHO_RAND_13",
          "future_KL_score": 0.0006947972612638686,
          "max_future_KL": 0.001811909838579595,
          "max_propagation_gain": 1.1378219605790547,
          "persistence_score": 1.0962620233189435
        },
        {
          "direction": "ORTHO_RAND_08",
          "future_KL_score": 0.005124741429106905,
          "max_future_KL": 0.025621438398957253,
          "max_propagation_gain": 1.301276126676442,
          "persistence_score": 1.132394041637173
        },
        {
          "direction": "ORTHO_RAND_07",
          "future_KL_score": 0.0022821064651001423,
          "max_future_KL": 0.00969117134809494,
          "max_propagation_gain": 1.4861385824396125,
          "persistence_score": 1.1703858421116755
        },
        {
          "direction": "ORTHO_RAND_04",
          "future_KL_score": 0.002137771292725077,
          "max_future_KL": 0.010626210831105709,
          "max_propagation_gain": 1.3735181784397221,
          "persistence_score": 1.1745763651987966
        },
        {
          "direction": "ORTHO_RAND_14",
          "future_KL_score": 0.003081209184585987,
          "max_future_KL": 0.015352834016084671,
          "max_propagation_gain": 1.432551686108881,
          "persistence_score": 1.2011664155556372
        },
        {
          "direction": "ORTHO_RAND_11",
          "future_KL_score": 0.0029693323406586813,
          "max_future_KL": 0.0131663354113698,
          "max_propagation_gain": 1.4903460889664948,
          "persistence_score": 1.2048634186889315
        },
        {
          "direction": "ORTHO_RAND_00",
          "future_KL_score": 0.001506075494145609,
          "max_future_KL": 0.005830023903399706,
          "max_propagation_gain": 1.5181623465798828,
          "persistence_score": 1.2321205299241837
        },
        {
          "direction": "ORTHO_RAND_01",
          "future_KL_score": 0.006559820402185323,
          "max_future_KL": 0.03100776858627796,
          "max_propagation_gain": 1.5522667158425627,
          "persistence_score": 1.285941939275063
        },
        {
          "direction": "ORTHO_RAND_05",
          "future_KL_score": 0.0035178172309002775,
          "max_future_KL": 0.010319344699382782,
          "max_propagation_gain": 1.6180638557304425,
          "persistence_score": 1.2994480805019828
        },
        {
          "direction": "ORTHO_RAND_09",
          "future_KL_score": 0.0016995964844909394,
          "max_future_KL": 0.7579002976417542,
          "max_propagation_gain": 2.0242157797790816,
          "persistence_score": 1.3452483338197343
        },
        {
          "direction": "ORTHO_RAND_02",
          "future_KL_score": 0.009545619669887628,
          "max_future_KL": 0.2985205352306366,
          "max_propagation_gain": 2.37479332272094,
          "persistence_score": 1.6669525374393612
        }
      ],
      "random_persistent_amplification_fraction": 1.0,
      "role": "INT8-row truncated pathological candidate",
      "spearman_max_gain_future_KL": 0.3999999999999988,
      "spearman_persistence_future_KL": 0.1999999999999994,
      "t0": 64,
      "valid_random_direction_count": 16
    }
  ],
  "positive_unit_count": 18,
  "rho_persistence_future_KL_ge_0p5_count": 2,
  "rho_persistence_future_KL_positive_count": 16
}
```

## Limits
- Empirical directional sensitivity panel only; no eigenmode, unstable eigenspace, or Lyapunov-vector claim.
- Single oracle pulse only; no repeated quantization after t0.
- Teacher-forced continuation uses retokenized P0 FP_STATE decoded responses; not exact replay.
- No method design is run.
