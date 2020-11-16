import pandas as pd
from userInputs import importFile
from engineerings import numeric_engineering,time_engineering
import joblib

def INIT(path,info):
    # exceptionsHandled = 0 # For Future testing
    print('\nINIT STARTED!')
    print('IMPORTING NECESSARY FILES')
    df,_ = importFile('./test/' + path)
    print(df.shape)

    df[info['PrimaryDate']] = pd.to_datetime(df[info['PrimaryDate']])

    print('\n#### Preparing Data For Univariate Analysis! ####\n')
    uni_df = df[[info['PrimaryDate'],info['Target']]]
    print(uni_df.shape)

    uni_df = numeric_engineering(uni_df)

    print('#### Running Time Engineering on Primary Date Column! ####')
    TIME_DF = time_engineering(uni_df)
    print(TIME_DF)

    print('\n')
    print(uni_df.head())
    return None,None

# For testing purpose
# INIT('messy3.csv',joblib.load('info'))
