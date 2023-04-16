# Changelog


## April 16, 2023

1. Updated tests.


## April 13, 2023


#### Summary
1. Updated pandas and yaml along with code where needed.

2. Using a newer version of scikit-learn.

3. Renamed normalizer names to replace colons ':' with dashes '-'.

4. Fix the requirements.txt file, making sure that both the benchmark and models parts run.

5. Add notes on installation.

#### Renamed normalizer files
The following files are broken in a Windows context because of the `:` in the names. These files were renamed replacing `:` with `-`.
```
./mimic3models/length_of_stay/los_ts0.8.input_str:previous.start_time:zero.n5e4.normalizer
./mimic3models/length_of_stay/los_ts1.0.input_str:previous.start_time:zero.n5e4.normalizer
./mimic3models/in_hospital_mortality/ihm_ts0.8.input_str:previous.start_time:zero.normalizer
./mimic3models/in_hospital_mortality/ihm_ts1.0.input_str:previous.start_time:zero.normalizer
./mimic3models/in_hospital_mortality/ihm_ts2.0.input_str:previous.start_time:zero.normalizer
./mimic3models/multitask/mult_ts1.0.input_str:previous.start_time:zero.normalizer
./mimic3models/phenotyping/ph_ts0.8.input_str:previous.start_time:zero.normalizer
./mimic3models/phenotyping/ph_ts1.0.input_str:previous.start_time:zero.normalizer
./mimic3models/decompensation/decomp_ts0.8.input_str:previous.n1e5.start_time:zero.normalizer
./mimic3models/decompensation/decomp_ts1.0.input_str:previous.n1e5.start_time:zero.normalizer
```

The following commands were issued:
```
 find . -type f -name "*:*" -exec rename 's/:/-/g' {} +
```
Searching for `:.*normalizer` all semicolons in the string were replaced.