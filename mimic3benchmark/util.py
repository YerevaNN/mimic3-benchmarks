from __future__ import absolute_import
from __future__ import print_function

import pandas as pd


def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)
