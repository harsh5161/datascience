import pandas as pd
from userInputs import importFile
from engineerings import numeric_engineering,date_engineering
import joblib

def INIT(path,info):
    exceptionsHandled = 0
    print('\nINIT STARTED!')
    print('IMPORTING NECESSARY FILES')
    df,_ = importFile('./test/' + path)
    print(df.shape)

#     df = numeric_engineering(df)
    df[info['PrimaryDate']] = pd.to_datetime(df[info['PrimaryDate']])

#     print('Printing Missing information')
#     MISSING = pd.DataFrame(((df.isnull().sum().sort_values(ascending=False)*100/len(df)).round(2)),columns=['Missing in %'])[:10]
#     print(MISSING)

#     beforeCols = df.columns
#     print('Dropping columns with more than 50% missing data')
#     df.dropna(axis=1,thresh=len(df)/2,inplace=True)
#     afterCols = df.columns
#     droppedCols = list(set(beforeCols)-set(afterCols))
#     print('{} columns had more than 50% missing data!'.format(len(droppedCols)))
#     print('The dropped columns are : ')
#     for col in droppedCols:
#         print(col)

#     if len(df) == 0:
#         print('The data uploaded has huge number of missing! Quitting!')
#         return None,None   
    print('\n#### Preparing Data For Univariate Analysis! ####\n')
    uni_df = df[[info['PrimaryDate'],info['Target']]]
    print(uni_df.shape)

    print('#### Running Date Engineering on Primary Date Column! ####')
    # try:
    DATE_DF,_ = date_engineering(df[info['PrimaryDate']],None)
    if DATE_DF == None:
        return None,None
    uni_df = uni_df.concat([DATE_DF],axis=1)
    # except:
    #     print('Date Engineering did not run! Date Features are not taken into account')
    #     exceptionsHandled += 1
    print('\n')
    print(uni_df.head())
    return None,None

# For testing purpose
# INIT('messy3.csv',joblib.load('info'))
