# Adding Imports
# File Imports
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from bayes_opt import BayesianOptimization
from tqdm import tqdm, tqdm_notebook
from pylab import rcParams
from itertools import islice
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, svm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from random import random
from statsmodels.tsa.ar_model import AR
from userInputs import *
from backbone import *

# Library Imports
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd  # Basic library for all of our dataset operations
import numpy as np
import requests
import io
import statsmodels.tsa.api as smt
import statsmodels as sm
import pmdarima as pm
import warnings
import xgboost as xgb
from math import sqrt
from metrics import evaluate
from plots import bar_metrics

warnings.filterwarnings("ignore")


# from fbprophet import Prophet


# Extra settings
seed = 42
# tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
# print(tf.__version__)

start = time.time()
complete = INIT()
if complete is None:
    print("Process Failed")

print(f"Process Success : Time Taken {time.time()-start}")
