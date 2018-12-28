MIMIC-III Benchmarks
=========================

[![Join the chat at https://gitter.im/YerevaNN/mimic3-benchmarks](https://badges.gitter.im/YerevaNN/mimic3-benchmarks.svg)](https://gitter.im/YerevaNN/mimic3-benchmarks?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Python suite to construct benchmark machine learning datasets from the MIMIC-III clinical database. Currently, the benchmark datasets cover four key inpatient clinical prediction tasks that map onto core machine learning problems: prediction of mortality from early admission data (classification), real-time detection of decompensation (time series classification), forecasting length of stay (regression), and phenotype classification (multilabel sequence classification).


## News

* 2018 December 28: The second draft of the paper is released on [arXiv](https://arxiv.org/abs/1703.07771).
* 2017 December 8: This work was presented as a spotlight presentation at NIPS 2017 [Machine Learning for Health Workshop](https://ml4health.github.io/2017/).
* 2017 March 23: We are pleased to announce the first official release of these benchmarks. We expect to release a revision within the coming months that will add at least ~50 additional input variables. We are likewise pleased to announce that the manuscript associated with these benchmarks is now [available on arXiv](https://arxiv.org/abs/1703.07771).

## Citation

If you use this code or these benchmarks in your research, please cite the following publication: *Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. arXiv:1703.07771* which is now available [on arXiv](https://arxiv.org/abs/1703.07771). **Please be sure also to cite the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**

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

## Structure
The content of this repository can be divided into four big parts:
* Tools for creating the benchmark datasets.  
* Tools for reading the benchmark datasets.
* Evaluation scripts.
* Baseline models and helper tools.

The `mimic3benchmark/scripts` directory contains scripts for creating the benchmark datasets.
The reading tools are in `mimic3benchmark/readers.py`.
All evaluation scripts are stored in the `mimic3benchmark/evaluation` directory.
The `mimic3models` directory contains the baselines models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.


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
    
2. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

       python -m mimic3benchmark.scripts.validate_events data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

5. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

       python -m mimic3benchmark.scripts.split_train_and_test data/root/
	
6. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
       python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
       python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
       python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
       python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.


## Readers
To simplify the reading of benchmark data we wrote special classes.
The `mimic3benchmark/readers.py` contains class `Reader` and five other task-specific classes derived from it.
These are designed to simplify reading of benchmark data. The classes require a directory containing ICU stays and a listfile specifying the samples.
Again, we encourage to use these readers to avoid mistakes in the reading step (for example using events that happened after the first `period_length` hours).  
For more information about using readers view the [`mimic3benchmark/more_on_readers.md`](mimic3benchmark/more_on_readers.md) file.


## Evaluation
For each of the four tasks we provide scripts for evaluating models.
These scripts receive a `csv` file containing the predictions and produce a `json` file containing the scores and confidence intervals for different metrics.
We highly encourage to use these scripts to prevent any mistake in the evaluation step.
For details about the usage of the evaluation scripts view the [`mimic3benchmark/evaluation/README.md`](mimic3benchmark/evaluation/README.md) file.


## Baselines
For each of the four main tasks we provide 7 baselines:  
* Linear/logistic regression
* Standard LSTM
* Standard LSTM + deep supervision
* Channel-wise LSTM
* Channel-wise LSTM + deep supervision
* Multitask standard LSTM
* Multitask channel-wise LSTM

The detailed descriptions of the baselines will appear in the next version of the paper.

Linear models can be found in `mimic3models/{task}/logistic` directories.
LSTM-based models are in `mimic3models/keras_models` directory.

Please note that running linear models can take hours because of extensive grid search and feature extraction.
You can change the size of the training data of linear models in the scripts and they will became faster (of course the performance will not be the same).

### Train / validation split

Use the following command to extract validation set from the training set. This step is required for running the baseline models. Likewise the train/test split, the train/validation split is the same for all tasks.

       python -m mimic3models.split_train_val {dataset-directory}
       
`{dataset-directory}` can be either `data/in-hospital-mortality`, `data/decompensation`, `data/length-of-stay`, `data/phenotyping` or `data/multitask`.


### In-hospital mortality prediction

Run the following command to train the neural network which gives the best result. We got the best performance on validation set after 28 epochs.
       
       python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality

Use the following command to train logistic regression. The best model we got used L2 regularization with `C=0.001`:
       
       python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/logistic

### Decompensation prediction

The best model we got for this task was trained for 36 chunks (that's less than one epoch; it overfits before reaching one epoch because there are many training samples for the same patient with different lengths).
       
       python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode train --batch_size 8 --output_dir mimic3models/decompensation

Use the following command to train a logistic regression. It will have L2 regularization with `C=0.001`, which gave us the best result. To run a grid search over a space of hyper-parameters add `--grid-search` to the command.
       
       python -um mimic3models.decompensation.logistic.main --output_dir mimic3models/decompensation/logistic

### Length of stay prediction

The best model we got for this task was trained for 19 chunks.
       
       python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --partition custom --output_dir mimic3models/length_of_stay

Use the following command to train a logistic regression. It will have L1 regularization with `C=0.00001`. To run a grid search over a space of hyper-parameters add `--grid-search` to the command.
       
       python -um mimic3models.length_of_stay.logistic.main_cf --output_dir mimic3models/length_of_stay/logistic

To run a linear regression use this command:

        python -um mimic3models.length_of_stay.logistic.main --output_dir mimic3models/length_of_stay/logistic


### Phenotype classification

The best model we got for this task was trained for 20 epochs.
       
       python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/phenotyping

Use the following command for logistic regression. It will have L1 regularization with `C=0.1`. To run a grid search over a space of hyper-parameters add `--grid-search` to the command.
       
       python -um mimic3models.phenotyping.logistic.main --output_dir mimic3models/phenotyping/logistic

### Multitask learning

`ihm_C`, `decomp_C`, `los_C` and `ph_C` coefficients control the relative weight of the tasks in the multitask model. Default is `1.0`. Multitask network architectures are stored in `mimic3models/multitask/keras_models`. Here is a sample command for running a multitask model.
       
       python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_lstm.py --dim 512 --timestep 1 --mode train --batch_size 16 --dropout 0.3 --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0 --output_dir mimic3models/multitask
       

## General todos:

- Improve comments and documentation
- Add comments about channel-wise LSTMs and deep superivison
- Add the best state files for each baseline
- Add https://zenodo.org/ 
- Release 1.0
- Update citation section with Zenodo DOI
- Add to MIMIC's derived data repo
- Refactor, where appropriate, to make code more generally useful
- Expand coverage of variable map and variable range files.
- Decide whether we are missing any other high-priority data (CPT codes, inputs, etc.)

