from backbone import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

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

analysis = input('Do you want to perform Multivariate analysis or Univariate Analysis? (M/U): ')
if analysis == 'U' or analysis == 'u':
    start = time.time()
    complete = INIT()
    if complete is None:
        print("Process Failed")
    else:
        print(f"Process Success : Time Taken {time.time()-start}")
elif analysis == 'M' or analysis == 'm':
    start = time.time()
    complete = INIT_multi()
    if complete is None:
        print("Process Failed")
    else:
        print(f"Process Success : Time Taken {time.time()-start}")