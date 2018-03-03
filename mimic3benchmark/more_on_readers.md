## Readers
The `mimic3benchmark/readers.py` contains class `Reader` and five other task-specific classes derived from it:  
* InHospitalMortalityReader
* DecompensationReader
* LengthOfStayReader
* PhenotypingReader
* MultitaskReader 

All of them have the same structure as the `Reader` class:
```python
class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        ...
  
    def get_number_of_examples(self):
        ...
  
    def random_shuffle(self, seed=None):
        ...
  
    def read_example(self, index):
        ...
  
    def read_next(self):
        ...
```

The initializer requires two paths: `dataset_dir` and `listfile`.
The former specifies a directory with ICU stays and the latter specifies a listfile that describes the samples.
If `listfile` is not given, the code will try to use `listfile.csv` file of the `dataset_dir`. 

Two functions are there for reading a sample: `read_example` and `read_next`.
The `read_example` function reads the sample with the given index, while `read_next` reads the next sample by using a cyclic counter inside.
Both of them return a dictionary. The details about returned dictionary are written in the documentation of `read_example` functions.
For example, [here](https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/readers.py#L120) is the description of the return dictionary of `InHospitalMortalityReader`.

Example of usage:
```python
from mimic3benchmark.readers import DecompensationReader

reader = DecompensationReader(dataset_dir='data/decompensation/train',
                              listfile='data/decompensation/train/listfile.csv')

print(reader.read_example(10))
```

The output will be:
```angular2html
{
  'X': array([['0.0455555555556', '', '69.0', ..., '38.111110263400604', '', ''],
         ['0.378888888889', '', '92.0', ..., '', '', ''],
         ['1.04555555556', '', '90.0', ..., '', '', ''],
         ...,
         ['78.0455555556', '', '60.0', ..., '', '', ''],
         ['78.6288888889', '', '', ..., '', '', ''],
         ['79.0455555556', '', '63.0', ..., '', '', '']], dtype='|S18'),
  'y': 0,
  't': 80.0,
  'name': '17378_episode1_timeseries.csv',
  'header': [
    'Hours',
    'Capillary refill rate',
    'Diastolic blood pressure',
    'Fraction inspired oxygen',
    'Glascow coma scale eye opening',
    'Glascow coma scale motor response',
    'Glascow coma scale total',
    'Glascow coma scale verbal response',
    'Glucose',
    'Heart Rate',
    'Height',
    'Mean blood pressure',
    'Oxygen saturation',
    'Respiratory rate',
    'Systolic blood pressure',
    'Temperature',
    'Weight',
    'pH'
  ]
}
```