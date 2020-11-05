import pandas as pd
from userInputs import importFile
from engineerings import numeric_engineering,date_engineering
from fbprophet.forecaster import Prophet # According to the installed directory and not as in documentation


def INIT(path,info):
    print('\nINIT STARTED!')
    print('IMPORTING NECESSARY FILES')
    df,_ = importFile('./test/' + path)
    print(df.shape)

    print('\nRunning univariate time series using FBProphet without data cleaning!')
    print('\n Creating FBProphet Instance!')
    m = Prophet()
    print('\nPreparing DataSet!')
    prophetDF = df[[info['PrimaryDate'],info['Target']]]
    prophetDF.columns = ['ds','y']
    prophetDF.ds = pd.to_datetime(prophetDF.ds,errors='coerce')
    prophetDF.dropna(inplace=True)
    print(prophetDF.isnull().sum())
    m.fit(prophetDF)

    return None,None
