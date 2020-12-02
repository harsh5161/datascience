# Modular Imports
# import numpy as np
# import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from userInputs import importFile,getInfo
from engineerings import getDateColumns,time_engineering,train_test_split,interpolateTarget
from plots import basicPlot,decompositionPlot
from init import INIT
import joblib
from pmdarima import auto_arima
from pmdarima.metrics import smape
# =============================================================================
# import matplotlib.pyplot as plt
# =============================================================================

def main(test=False,props=None):
    print('This is Time Series Folder and All functions and files will be contained here')
    print('#### For now use only CSV files ####')
    ## Importing File ## We need to store CSV files in test folder in TimeSeries ##
    if not test:
        # For single file manual testing
        path = input('Enter filename for TimeSeries : ').strip()
        if path == '':
            print('\nNo File Selected, QUITTING!')
            return 1,0
        df,_ = importFile('./test/' + path,nrows=100)
    else:
        # For Automated Testing
        df,_ = importFile('./test/' + props[path],nrows=100)
    # Stripping trailing and leading spaces and replacing spaces in between with _
    df.columns = [x.strip().replace(' ','_') for x in df.columns]
    datecols,_ = getDateColumns(df.select_dtypes('object'))
    if len(datecols) == 0:
        print('No datecolumns found, QUITTING!')
        return None,None

    info = getInfo(df.columns,datecols) # Get Target and Primary Date Column

    if not info:
        print('QUITTING!')
        return None,None

    info['DateColumns'] = datecols
    joblib.dump(info,'info');print('INFO SAVED!')

    # Reimporting complete data, slicing date and target columns,
    props = INIT(path,info)
    frontEndProgressBar = 0.05
    print('\n{}% done on frontEndProgessBar\n'.format(frontEndProgressBar*100))

    props = time_engineering(props)
    if props == dict():
        print('QUITTING!');return None,None
    frontEndProgressBar = 0.10
    print('\n{}% done on frontEndProgessBar'.format(frontEndProgressBar*100))

    basicPlot(props)
    frontEndProgressBar = 0.20
    print('\n{}% done on frontEndProgessBar'.format(frontEndProgressBar*100))
    
    print('\nApplying Linear Interpolation to the Training Target Column')    
    props = interpolateTarget(props)
    frontEndProgressBar = 0.30
    print('\n{}% done on frontEndProgessBar'.format(frontEndProgressBar*100))
    
    decompositionPlot(props)
    frontEndProgressBar = 0.40
    print('\n{}% done on frontEndProgessBar'.format(frontEndProgressBar*100))

    props['Margin'] = int(len(df)*0.8)
    X_train,y_train,X_test,y_test = train_test_split(props)
    print('\nTrain_Test_Split (FIT SAMPLE / HOLD_OUT SAMPLE SPLIT DONE!)')
    
# =============================================================================
#     AUTO ARIMA
# =============================================================================

    print('\n#### FITTING AUTO-ARIMA MODEL #### RUNNING WAIT ####')

    AutoArimaModel = auto_arima(y_train,exogenous=X_train,seasonal=True,suppress_warnings=True)
    AutoArimaForecasts = AutoArimaModel.predict(len(X_test),exogenous=X_test)
    
    print('\nThe mean absolute percentage error for Auto Arima is')
    print(smape(y_test,AutoArimaForecasts))
    props['AutoArima'] = AutoArimaModel
    
# =============================================================================
#     AUTO ARIMA
# =============================================================================

    return 1,props['exceptionsHandled']

if __name__ == '__main__':
    returnValue,numberOfErrors = main()
    print('\n#### Code run successfully ####\n')