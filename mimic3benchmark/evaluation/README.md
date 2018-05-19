# Evaluation Scripts

This directory contains four scripts (each for one benchmark task) for evaluating the models.
These scripts take a prediction file and calculate different task-related metrics.
Additionally the scripts use bootstrapping to estimate the standard deviations of the scores and to find an 95% confidence interval estimate.
The calculated statistics are stored in a `json` file similar to the following.
```json
{
  "n_iters": 10000,
  "AUC of PRC": {
    "std": 0.02659696284278772,
    "median": 0.5334109872918915,
    "value": 0.532815668505795,
    "97.5% percentile": 0.5834589517806671,
    "2.5% percentile": 0.4797467991222657,
    "mean": 0.532867859291327
  },
  "min(+P, Se)": {
    "std": 0.02248172735602079,
    "median": 0.5012820597111092,
    "value": 0.5,
    "97.5% percentile": 0.5435366790927356,
    "2.5% percentile": 0.45604395604395603,
    "mean": 0.500725209322173
  },
  "AUC of ROC": {
    "std": 0.008897102122236327,
    "median": 0.8703254626477475,
    "value": 0.8701699757471121,
    "97.5% percentile": 0.8871606639685519,
    "2.5% percentile": 0.8523142405908618,
    "mean": 0.87016429799627
  }
}
```

## Usage

The usage of the scrips is the following:
```
python -m mimic3benchmark.evaluation.evaluate_{task} [-h] [--test_listfile TEST_LISTFILE] [--n_iters N_ITERS]\
                                                     [--save_file SAVE_FILE] prediction
```

* `test_listile` should be a `csv` file similar to `data/{task}/train/listfile.csv` files.
This file should contain all the samples for which the models is predicting.
The default value of this parameter is the `data/{task}/test/listfile.csv`.
* `save_file` is the name of `json` file that should be produced.
* `n_iters` specifies the number of bootstrap iterations.
* `prediction` is a `csv` file similar to `test_listfile` with one addition that it also contains column(s) related to predictions.

The reason we have two similar files (`test_litfile` and `prediction`) is to have a way to ensure that there is a prediction for all samples of `test_listfile` and that `prediction` doesn't contain any wrong information about the targets.  
The format of `prediction` is task-specific and is described below.

### In-hospital morality
The prediction file should have 3 columns: `stay`, `prediction`, `y_true`.
Similar to task listfiles (`data/in-hospital-mortality/{set}/listfile.csv`), each `stay` defines an sample with `y_true` label.
Here is a part of a valid prediction file.
```angular2html
stay,prediction,y_true
28401_episode1_timeseries.csv,0.198586,0
52467_episode2_timeseries.csv,0.469585,1
8380_episode1_timeseries.csv,0.236593,1
30633_episode1_timeseries.csv,0.026102,0
```

### Decompensation and length of stay
The prediction file should have 4 columns: `stay`, `period_length`, `prediction` and `y_true`.
Similar to task listfiles (`data/{task}/{set}/listfile.csv`), each (`stay`, `period_length`) pair defines an sample with `y_true` label.
Here is a part of a valid prediction file for decompensation prediction task.
```angular2html
stay,period_length,prediction,y_true
12573_episode1_timeseries.csv,394.000000,0.000571,0
5598_episode4_timeseries.csv,394.000000,0.001081,0
94_episode2_timeseries.csv,394.000000,0.077848,0
93381_episode1_timeseries.csv,394.000000,0.002446,0
```
Here is a part of a valid prediction file for LOS prediction task.
```angular2html
stay,period_length,prediction,y_true
25256_episode5_timeseries.csv,5.000000,63.554131,480.366400
25256_episode5_timeseries.csv,6.000000,51.653477,479.366400
6468_episode1_timeseries.csv,83.000000,21.742018,6.788800
6468_episode1_timeseries.csv,84.000000,20.712656,5.788800
```

### Phenotyping
The prediction list-file should have 52 columns: `stay`, `period_length`, `pred_{i}` and `label_{i}`, where `i` ranges from 1 to 25 (the number of phenotypes).
Similar to task list-files (`data/phenotyping/{set}/listfile.csv`), each (`stay`, `period_length`) pair defines an sample with `label_{i}` labels.
Here is a part of a valid prediction file.
```angular2html
stay,period_length,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12,pred_13,pred_14,pred_15,pred_16,pred_17,pred_18,pred_19,pred_20,pred_21,pred_22,pred_23,pred_24,pred_25,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11,label_12,label_13,label_14,label_15,label_16,label_17,label_18,label_19,label_20,label_21,label_22,label_23,label_24,label_25
99091_episode1_timeseries.csv,177.612000,0.262400,0.015973,0.052016,0.333623,0.071021,0.321734,0.074759,0.026440,0.279651,0.113590,0.002932,0.023886,0.132705,0.338218,0.458486,0.021708,0.059589,0.113542,0.329968,0.236365,0.191117,0.496825,0.764580,0.111754,0.018451,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0
87754_episode1_timeseries.csv,177.201600,0.206742,0.027687,0.187667,0.783724,0.234875,0.275119,0.707655,0.136530,0.468962,0.638654,0.034629,0.312135,0.487467,0.573972,0.347912,0.024706,0.183768,0.136024,0.099600,0.051225,0.273147,0.072753,0.146137,0.057414,0.069350,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1
83962_episode1_timeseries.csv,178.195200,0.658058,0.004599,0.182063,0.716813,0.736834,0.100796,0.248655,0.244482,0.750731,0.655207,0.317332,0.404463,0.491454,0.236359,0.516147,0.133137,0.685693,0.290062,0.045600,0.005601,0.106505,0.065165,0.028664,0.330075,0.268535,1,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,1,1
65407_episode1_timeseries.csv,178.891200,0.617822,0.053118,0.115121,0.571016,0.088866,0.204868,0.366116,0.029166,0.394338,0.153550,0.083783,0.259495,0.133264,0.412249,0.679179,0.111867,0.096425,0.145118,0.075616,0.175652,0.194995,0.606794,0.936352,0.657388,0.298098,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,1,0,0
```
