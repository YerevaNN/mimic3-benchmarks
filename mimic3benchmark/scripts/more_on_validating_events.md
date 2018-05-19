# More on validating results

The `validate_events.py` tries to assert some assumptions about the data; find events with specific problems and fix these problems if possible.  
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
