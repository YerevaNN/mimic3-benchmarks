# Helper Tools
To simplify the reading and pre-processing of benchmark data we provide some helper tools:

* Discretizer
* Normalizer
* Tools for computing metrics


## Discretizer

The file [`mimic3models/preprocessing.py`](mimic3models/preprocessing.py) contains `Discretizer` class, which can be used for re-sampling time-series into regularly spaced intervals and for imputing missing values.
The `Discretizer` class has the following signature:
```python
class Discretizer():
    
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero'):
        """ This class can be used for re-sampling time-series data into regularly spaced intervals
            and for imputing the missing values.
  
        :param timestep: Defines the length of intervals.
        :param store_masks: When this parameter is True, the discretizer will append a binary vector to
                            the data of each time-step. This binary vector specifies which entries are imputed.
        :param impute_strategy: Specifies the imputation strategy. Possible values are 'zero', 'normal_value',
                               'previous' and 'next'.
        :param start_time: Specifies when to start to re-sample the data. Possible values are 'zero' and 'relative'.
                           In case of 'zero' the discretizer will start to re-sample the data from time 0 and in case of 
                           'relative it will start to re-sample from the moment when the first ICU event happens'.
        """
        ...
  
    def transform(self, X, header=None, end=None):
        ...
  
    def print_statistics(self):
        ...

```


## Normalizer
The file [`mimic3models/preprocessing.py`](mimic3models/preprocessing.py) also contains `Normalizer` class which can be used to standardize discetized data.
The normalizer class has the following signature:
```python
class Normalizer():

    def __init__(self, fields=None):
        ...
  
    def _feed_data(self, x):
        ...
  
    def _save_params(self, save_file_path):
        ...
  
    def load_params(self, load_file_path):
        ...
  
    def transform(self, X):
        ...

```

To create a normalizer state file use the [`mimic3models/create_normalizer_state.py`](mimic3models/create_normalizer_state.py) script.
The details of usage can be found in the script.


## Metrics
The file [`mimic3models/metrics.py`](mimic3models/metrics.py) contains functions that are used to compute metrics used in the benchmarks.

| Task | Function |  
| :------ | :------ |  
| In-hospital mortality  | `print_metrics_binary` |
| Decompensation | `print_metrics_binary` |
| Length of stay (regression) | `print_metrics_regression` |  
| Length of stay (classification) | `print_metrics_custom_bins` |  
| Phenotyping | `print_metrics_multilabel` |
