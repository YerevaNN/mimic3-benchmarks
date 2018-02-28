MIMIC-III Benchmarks
=========================

[![Join the chat at https://gitter.im/YerevaNN/mimic3-benchmarks](https://badges.gitter.im/YerevaNN/mimic3-benchmarks.svg)](https://gitter.im/YerevaNN/mimic3-benchmarks?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
Python suite to construct benchmark machine learning datasets from the MIMIC-III clinical database. Currently, we are focused on building a multitask learning benchmark dataset that includes four key inpatient clinical prediction tasks that map onto core machine learning problems: prediction of mortality from early admission data (classification), real-time detection of decompensation (time series classification), forecasting length of stay (regression), and phenotype classification (multilabel sequence classification).

## News

* 2017 March 23: We are pleased to announce the first official release of these benchmarks. We expect to release a revision within the coming months that will add at least ~50 additional input variables. We are likewise pleased to announce that the manuscript associated with these benchmarks is now [available on arXiv](https://arxiv.org/abs/1703.07771).

## Citation

If you use this code or these benchmarks in your research, please cite the following publication: *Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. arXiv:1703.07771* which is now available [on arXiv](https://arxiv.org/abs/1703.07771). This paper is currently under review for SIGKDD and if accepted, the citation will change. **Please be sure also to cite the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**

## Motivation

Despite rapid growth in research that applies machine learning to clinical data, progress in the field appears far less dramatic than in other applications of machine learning. In image recognition, for example, the winning error rates in the [ImageNet Large Scale Visual Recognition Challenge](http://image-net.org/challenges/LSVRC/) (ILSVRC) plummeted almost 90% from 2010 (0.2819) to 2016 (0.02991).
There are many reasonable explanations for this discrepancy: clinical data sets are [inherently noisy and uncertain](http://www-scf.usc.edu/~dkale/papers/marlin-ihi2012-ehr_clustering.pdf) and often small relative to their complexity, and for many problems of interest, [ground truth labels for training and evaluation are unavailable](https://academic.oup.com/jamia/article-abstract/23/6/1166/2399304/Learning-statistical-models-of-phenotypes-using?redirectedFrom=PDF).

However, there is another, simpler explanation: practical progress has been difficult to measure due to the absence of community benchmarks like ImageNet. Such benchmarks play an important role in accelerating progress in machine learning research. For one, they focus the community on specific problems and stoke ongoing debate about what those problems should be. They also reduce the startup overhead for researchers moving into a new area. Finally and perhaps most important, benchmarks facilitate reproducibility and direct comparison of competing ideas.

Here we present four public benchmarks for machine learning researchers interested in health care, built using data from the publicly available Medical Information Mart for Intensive Care (MIMIC-III) database ([paper](http://www.nature.com/articles/sdata201635), [website](http://mimic.physionet.org)). Our four clinical prediction tasks are critical care variants of four opportunities to transform health care using in "big clinical data" as described in [Bates, et al, 2014](http://content.healthaffairs.org/content/33/7/1123.abstract):

* early triage and risk assessment, i.e., mortality prediction
* prediction of physiologic decompensation
* identification of high cost patients, i.e. length of stay forecasting
* characterization of complex, multi-system diseases, i.e., acute care phenotyping

In [Harutyunyan, Khachatrian, Kale, and Galstyan 2017](https://arxiv.org/abs/1703.07771), we propose a multitask RNN architecture to solve these four tasks simultaneously and show that this model generally outperforms strong single task baselines.

## Requirements

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

For logistic regression  baselines [sklearn](http://scikit-learn.org/) is required. LSTM models use [Keras](https://keras.io/).

## Building a benchmark

Here are the required steps to build the benchmark. It assumes that you already have MIMIC-III dataset (lots of CSV files) on the disk.
1. Clone the repo.

       git clone https://github.com/YerevaNN/mimic3-benchmarks/
       cd mimic3-benchmarks/
    
2. Add the path to the `PYTHONPATH` (sorry for this).
 
       export PYTHONPATH=$PYTHONPATH:[PATH TO THIS REPOSITORY]

3. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/[SUBJECT_ID/stays.csv`, diagnoses to `data/[SUBJECT_ID]/diagnoses.csv`, and events to `data/[SUBJECT_ID]/events.csv`. This step might take around an hour.

       python scripts/extract_subjects.py [PATH TO MIMIC-III CSVs] data/root/

4. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows.

       python scripts/validate_events.py data/root/

5. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```[SUBJECT_ID]/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```[SUBJECT_ID]/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.).

       python scripts/extract_episodes_from_subjects.py data/root/

6. The next command splits the whole dataset into training and testing sets. Note that all benchmarks use the same split:

       python scripts/split_train_and_test.py data/root/
	
7. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       python scripts/create_in_hospital_mortality.py data/root/ data/in-hospital-mortality/
       python scripts/create_decompensation.py data/root/ data/decompensation/
       python scripts/create_length_of_stay.py data/root/ data/length-of-stay/
       python scripts/create_phenotyping.py data/root/ data/phenotyping/
       python scripts/create_multitask.py data/root/ data/multitask/
        
## Working with baseline models

For each of the 4 main tasks we provide logistic regression and LSTM baselines. 
Please note that running linear models can take hours because of extensive grid search. You can change the `chunk_size` parameter in codes and they will became faster (of course the performance will not be the same).

### Train / validation split

Use the following command to extract validation set from the traning set. This step is required for running the baseline models.

       python mimic3models/split_train_val.py [TASK]
       
`[TASK]` is either `in-hospital-mortality`, `decompensation`, `length-of-stay`, `phenotyping` or `multitask`.


### In-hospital mortality prediction

Run the following command to train the neural network which gives the best result. We got the best performance on validation set after 28 epochs.
       
       cd mimic3models/in_hospital_mortality/
       python -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8

Use the following command to train logistic regression. The best model we got used L2 regularization with `C=0.001`:
       
       cd mimic3models/in_hospital_mortality/logistic/
       python -u main.py --l2 --C 0.001

### Decompensation prediction

The best model we got for this task was trained for 36 chunks (that's less than one epoch; it overfits before reaching one epoch because there are many training samples for the same patient with different lengths).
       
       cd mimic3models/decompensation/
       python -u main.py --network ../common_keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode train --batch_size 8

Use the following command to train a logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
       
       cd mimic3models/decompensation/logistic/
       python -u main.py

### Length of stay prediction

The best model we got for this task was trained for 19 chunks.
       
       cd mimic3models/length_of_stay/
       python -u main.py --network ../common_keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --partition custom

Use the following command to train a logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
       
       cd mimic3models/length_of_stay/logistic/
       python -u main_cf.py

### Phenotype classification

The best model we got for this task was trained for 20 epochs.
       
       cd mimic3models/phenotyping/
       python -u main.py --network ../common_keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8

Use the following command for logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
       
       cd mimic3models/phenotyping/logistic/
       python -u main.py

### Multitask learning

`ihm_C`, `decomp_C`, `los_C` and `ph_C` coefficients control the relative weight of the tasks in the multitask model. Default is `1.0`. Multitask network architectures are stored in `mimic3models/multitask/keras_models`. Here is a sample command for running a multitask model.
       
       cd mimic3models/multitask/
       python -u main.py --network keras_models/lstm.py --dim 512 --timestep 1 --mode train --batch_size 16 --dropout 0.3 --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0
       

## General todos:

- Test and debug
- Add comments and documentation
- Add comments about channel-wise LSTMs and deep superivison
- Add the best state files for each baseline
- Refactor, where appropriate, to make code more generally useful
- Expand coverage of variable map and variable range files.
- Decide whether we are missing any other high-priority data (CPT codes, inputs, etc.)

## More on validating results

With `validate_events.py` try to assert some assumptions about the data; find events with some problems and fix these problems if possible.  
Assumptions we assert:
* There is one-to-one mapping between HADM_ID and ICUSTAY_ID in `stays.csv` files.
* HADM_ID and ICUSTAY_ID are not empty in `stays.csv` files.
* `stays.csv` and `events.csv` files are always present.
* There is no case, where after initial filtering we cannot recover empty ICUSTAY_IDs.
  
Problems we fix (the order of this steps is fixed):
* Remove all events for which HADM_ID is missing.
* Remove all events for which HADM_ID is not present in `stays.csv`.
* If ICUSTAY_ID is missing in an event and HADM_ID is not missing, then we look at `stays.csv` and try to recover ICUSTAY_ID.
* Remove all events for which we cannot recover ICUSTAY_ID.
* Remove all events for which ICUSTAY_ID is not present in `stays.csv`.

Here is the output of `validate_events.py`:

| Type | Description | Number of rows |
| --- | --- | --- |
| `n_events` | total number of events | 253116833 |
| `empty_hadm` | HADM_ID is empty in `events.csv`. We exclude such events. | 5162703 |
| `no_hadm_in_stay` | HADM_ID does not appear in `stays.csv`. We exclude such events. | 32266173 |
| `no_icustay` | ICUSTAY_ID is empty in `events.csv`. We try to fix such events. | 15735688 |
| `recovered` | empty ICUSTAY_IDs are recovered according to stays.csv files (given HADM_ID) | 15735688 |
| `could_not_recover` | empty ICUSTAY_IDs that are not recovered. This should be zero. | 0 |
| `icustay_missing_in_stays` | ICUSTAY_ID does not appear in stays.csv. We exclude such events. | 7115720 |
