# Changelog

Updated pandas and yaml along with code where needed.

# Renamed files
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