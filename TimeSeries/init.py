import pandas as pd
from userInputs import importFile
from engineerings import numeric_engineering,time_engineering
import joblib

def INIT(path,info):
    exceptionsHandled = 0 # For Future testing
    print('\nINIT STARTED!')
    print('IMPORTING NECESSARY FILES')
    df,_ = importFile('./test/' + path)
    print(df.shape)

    df[info['PrimaryDate']] = pd.to_datetime(df[info['PrimaryDate']],infer_datetime_format=True)

    print('\n#### Preparing Data For Univariate Analysis! ####\n')
    uni_df = df[[info['PrimaryDate'],info['Target']]]
    print(uni_df.shape)

    # uni_df is univariate dataframe, has 2 columns
    # 1. Primary Date Column, 2. Target Column
    uni_df = numeric_engineering(uni_df) # Numeric Engineering on 2 columns only

    # Creating single parameter for chained function calls
    props = {'df':uni_df,'info':info,'exceptionsHandled':exceptionsHandled} 
    print('\nPrinting DataFrame head!\n')                      
    print(uni_df.head())
    return props

# For testing purpose
# INIT('messy3.csv',joblib.load('info'))
