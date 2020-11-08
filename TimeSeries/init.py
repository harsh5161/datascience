import pandas as pd
from userInputs import importFile
from engineerings import numeric_engineering,date_engineering
from traces import TimeSeries
import joblib
from datetime import datetime

def INIT(path,info):
    print('\nINIT STARTED!')
    print('IMPORTING NECESSARY FILES')
    df,_ = importFile('./test/' + path)
    print(df.shape)

    df = numeric_engineering(df)
    df[info['PrimaryDate']] = pd.to_datetime(df[info['PrimaryDate']])

    return None,None

# For testing purpose
# INIT('messy3.csv',joblib.load('info'))
