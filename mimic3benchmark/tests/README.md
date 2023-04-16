# Tests for the benchmark

The main script is the `mimic3benchmark/tests/hash_tables.py` script which takes a directory and hashes all `.csv` files inside it recursively.
The output is a `.csv` file which lists all `.csv` files and their hashes.
If this files matches with the corresponding file from `mimic3benchmark/tests/resources` then the test is successful.

The commands below check whether the benchmark was generated correctly. Note that it is possible to have different `{task}/train/listfile.csv` and `{task}/test/listfile.csv` files sometimes. If that is the only difference, then there is nothing to worry. This means that the content of listfiles is expected, but the order of examples is different.

**Warning:** the tests are somehow sensitive to Python and library versions, so please use Python 3.7.13 and the exact library versions specified in `requirements.txt`.


### In-hospital mortality
```bash
python -um mimic3benchmark.tests.hash_tables -d data/in-hospital-mortality -o ihm.csv;
diff ihm.csv mimic3benchmark/tests/resources/ihm.csv;
```

### Decompensation
```bash
python -um mimic3benchmark.tests.hash_tables -d data/decompensation -o decomp.csv;
diff decomp.csv mimic3benchmark/tests/resources/decomp.csv;
```

### Length-of-stay
```bash
python -um mimic3benchmark.tests.hash_tables -d data/length-of-stay -o los.csv;
diff los.csv mimic3benchmark/tests/resources/los.csv;
```

### Phenotyping
```bash
python -um mimic3benchmark.tests.hash_tables -d data/phenotyping -o pheno.csv;
diff pheno.csv mimic3benchmark/tests/resources/pheno.csv;
```

### Multitasking
```bash
python -um mimic3benchmark.tests.hash_tables -d data/multitask -o multitask.csv;
diff multitask.csv mimic3benchmark/tests/resources/multitask.csv;
```

### Root directory
The content of `data/root` directory can also be verified, but this needs to be done after the step that runs `mimic3benchmark/scripts/split_train_and_test.py`.
This step is not essential and can take a long time.
```bash
python -um mimic3benchmark.tests.hash_tables -d data/root -o root-final.csv;
diff root-final.csv mimic3benchmark/tests/resources/root-final.csv;
```
