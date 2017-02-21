MIMIC-III Benchmarks
=========================
Python suite to construct benchmark machine learning datasets from the MIMIC-III clinical database. Currently, we are focused on building a multitask learning benchmark dataset that includes four key inpatient clinical prediction tasks that map onto core machine learning problems: prediction of mortality from early admission data (classification), real-time detection of decompensation (time series classification), forecasting length of stay (regression), and phenotype classification (multilabel sequence classification).

*N.B. 1: these benchmarks are currently works-in-progress and undergoing rapid development. We expect to do our first official "release" no later March 1, 2017. In the meantime, we invite the community to experiment with it, to provide feedback, and most especially to send pull requests, but we reserve the right to make changes that are not backward compatible until the first release.*

*N.B. 2: these benchmarks are attached to a manuscript currently under review and to be posted on arXiv no later than March 1, 2017. Anyone who uses these benchmarks in their research should cite this manuscript once it is available, as well as the original [MIMIC-III paper]((http://www.nature.com/articles/sdata201635)). For futher detail, see the license.*

## Motivation

Despite rapid growth in research that applies machine learning to clinical data, progress in the field appears far less dramatic than in other applications of machine learning. In image recognition, for example, the winning error rates in the [ImageNet Large Scale Visual Recognition Challenge](http://image-net.org/challenges/LSVRC/) (ILSVRC) plummeted almost 90\% from 2010 (0.2819) to 2016 (0.02991).
There are many reasonable explanations for this discrepancy: clinical data sets are [inherently noisy and uncertain](http://www-scf.usc.edu/~dkale/papers/marlin-ihi2012-ehr_clustering.pdf) and often small relative to their complexity, and for many problems of interest, [ground truth labels for training and evaluation are unavailable](https://academic.oup.com/jamia/article-abstract/23/6/1166/2399304/Learning-statistical-models-of-phenotypes-using?redirectedFrom=PDF).

However, there is another, simpler explanation: practical progress has been difficult to measure due to the absence of community benchmarks like ImageNet. Such benchmarks play an important role in accelerating progress in machine learning research. For one, they focus the community on specific problems and stoke ongoing debate about what those problems should be. They also reduce the startup overhead for researchers moving into a new area. Finally and perhaps most important, benchmarks facilitate reproducibility and direct comparison of competing ideas.

Here we present four public benchmarks for machine learning researchers interested in health care, built using data from the publicly available Medical Information Mart for Intensive Care (MIMIC-III) database ([paper](http://www.nature.com/articles/sdata201635), [website](http://mimic.physionet.org)). Our four clinical prediction tasks are critical care variants of four opportunities to transform health care using in "big clinical data" as described in [Bates, et al, 2014](http://content.healthaffairs.org/content/33/7/1123.abstract):

* early triage and risk assessment, i.e., mortality prediction
* prediction of physiologic decompensation
* identification of high cost patients, i.e. length of stay forecasting
* characterization of complex, multi-system diseases, i.e., acute care phenotyping

In Hrarutyunyan, Khachatrian, Kale, and Galstyan, 2017 (under review for SIGKDD 2017, arXiv manuscript forthcoming), we propose a multitask RNN architecture to solve these four tasks simultaneously and show that this model generally outperforms strong single task baselines.

## Requirements

We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

## Building a benchmark

This is very, VERY early so use at your own peril! We don't have any benchmarks ready just yet. Rather, we have a pair of scripts that process MIMIC into a slightly more usable format. First, running

```python scripts/extract_subjects.py [PATH TO MIMIC-III CSVs] [OUTPUT PATH]```

will break up and store the ICU stay, diagnosis, and events tables by subject. It generates one directory per SUBJECT_ID and writes ICU stay information to ```[OUTPUT PATH]/[SUBJECT_ID]/stays.csv```, diagnoses to ```[SUBJECT_ID]/diagnoses.csv```, and events to  ```[SUBJECT_ID]/events.csv```.

**Be warned: the above takes ~~FOREVER~~ less than 2 hours (after [this commit](https://github.com/Hrant-Khachatrian/mimic3-benchmarks/commit/ba9f53c2b593fe13ba62deff02dcea1a2027e9f1)) if you include the CHARTEVENTS and LABEVENTS data tables.**

We are making some choices (in that script) that are specific to the phenotyping benchmark, such as excluding patients with transfers or multiple ICU visits within the same hospital admission, that may not be appropriate for other projects. However, we have attempted to write the code to be modular enough that it could be used for other benchmarks in the future.

#### Validating results
The script at `scripts/validate_events.py` looks for various problems in the generated CSVs, attempts to fix some of them and removes the others. Here are the results on randomly chosen 1000 subjects:

| Type | Description | Number of rows |
| --- | --- | --- |
| `n_events` | total number of events | 5937206 |
| `nohadminstay` | HADM_ID does not appear in `stays.csv` | 836341 |
| `emptyhadm` | HADM_ID is empty | 126480 |
| `icustaymissinginstays` | ICUSTAY_ID does not appear in `stays.csv` | 232624 |
| `noicustay` | ICUSTAY_ID is empty | 347768 |
| `recovered` | empty ICUSTAY_IDs are recovered according to `stays.csv` files (given `HADM_ID`) | 347768 |
| `couldnotrecover` | empty ICUSTAY_IDs that are not recovered. This should be zero, because the unrecoverable ones are counted in `icustaymissinginstays` | 0 |

4741761 events (80%) remain after removing all suspicious rows.

#### Extracting events of known types
Next, running

```python scripts/extract_episodes_from_subjects.py [PATH TO SUBJECTS] [PATH TO VARIABLE MAP FILE] [PATH TO VARIABLE RANGES FILE]```

will break up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```[SUBJECT_ID]/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```[SUBJECT_ID]/episode{#}.csv```.

The second script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Examples can be found in resources, but these are active development, and the checked in versions may lag, so ask for access to the Google docs where we're revising them.

## TODO

There is a LOT of work to be done before we're ready to publish an actual benchmark and share this code with the world. Here's a brief braindump, but I'll file actual issues soon.

### Phenotyping-specific todos:

- Choose subset of ICD-9 codes to use as labels
- Decide on whether and how to aggregate higher precision ICD-9 codes
- Check assumptions made along the way, including
  - exclude patients with transfers
  - exclude patients with multiple stays within same hospital admission
  - should we apply any additional exclusion criteria
- Decide whether we are missing any other high-priority data (CPT codes, inputs, etc.)
- Write code to process data into final format (numpy arrays? CSVs? JSON?)
- Get some sanity-checking results with simpler models

### General todos:

- Test and debug
- Add comments and documentation
- Refactor, where appropriate, to make code more generally useful
- Expand coverage of variable map and variable range files.
