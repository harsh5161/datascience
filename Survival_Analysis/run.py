import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import joblib
from dask.distributed import progress, Client
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

client = Client()

pd.set_option('display.max_columns', None)

seed = 42
# tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
# print(tf.__version__)

with joblib.parallel_backend('dask'):
    rate = input('Do you want to perform quick/slow results? (Enter "y" for quick results): ').upper()
    start = time.time()
    if rate == 'Y':
        from backbone_uni import *
        !mkdir unknown
        !mkdir scoring
        !mkdir default
        complete = INIT()
        if complete is None:
            print("Process Failed")
        else:
            print(f"Process Success : Time Taken {time.time()-start}")
    else:
        from backbone_multi import *
        !mkdir unknown
        !mkdir scoring
        !mkdir default
        # from backbone_uni import *
        # complete = INIT()
        complete_multi = INIT_multi()
        if complete_multi is None:
            print("Process Failed")
        else:
            print(f"Process Success : Time Taken {time.time() - start}")