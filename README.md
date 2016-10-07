MIMIC-III Benchmarks
=========================

Python suite to construct benchmark machine learning datasets from the MIMIC-III clinical database. Currently, we are focused on building a benchmark dataset for phenotyping, i.e., classifying MIMIC-III ICU episodes where the labels are ICD-9 diagnostic codes.

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
