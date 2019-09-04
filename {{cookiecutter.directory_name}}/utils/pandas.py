'''
Utilities and wrappers for pandas
'''

import pandas as pd


# TODO: 'transpose: bool = True' parameter
def describe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns transposed describe for the dataframe enriched with column types.
    '''
    desc = df.describe(include = 'all').transpose()
    desc.insert(0, 'type', [df.dtypes[x] for x in desc.index])
    return desc
