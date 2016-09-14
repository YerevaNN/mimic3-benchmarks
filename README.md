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

**Be warned: the above takes FOREVER if you include the CHARTEVENTS or LABEVENTS data tables.**

We are making some choices (in that script) that are specific to the phenotyping benchmark, such as excluding patients with transfers or multiple ICU visits within the same hospital admission, that may not be appropriate for other projects. However, we have attempted to write the code to be modular enough that it could be used for other benchmarks in the future.

Next, running

```python scripts/extract_episodes_from_subjects.py [PATH TO SUBJECTS] [PATH TO VARIABLE MAP FILE] [PATH TO VARIABLE RANGES FILE]```

will break up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```[SUBJECT_ID]/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```[SUBJECT_ID]/episode{#}.csv```.

The second script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Examples can be found in resources, but these are active development, and the checked in versions may lag, so ask for access to the Google docs where we're revising them.

## TODO
