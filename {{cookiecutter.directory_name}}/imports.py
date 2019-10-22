# Generic libraries

import numpy as np

try:
    #raise ImportError('Skipping modin import.') # Uncomment when encountering issues, for example this issue: https://github.com/modin-project/modin/issues/468
    import modin.pandas as pd
    print('Note: Imported modin.pandas in place of pandas as pd.')
except ImportError:
    try:
        import pandas as pd
        print('Imported regular pandas')
    except ImportError:
        print("Failed to import pandas from any known place")

import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

# import scipy as sp
# import sklearn
# import re

# Custom utils
from utils.geo import *
from utils.filesystem import *
from utils.xml import *
from utils.pandas import *
from utils.plots import *

# Custom default settings
# from utils.settings import *
plt.style.use('seaborn')