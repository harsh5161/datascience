import numpy as np
import pandas as pd
from userInputs import importFile,getInfo
from engineerings import getDateColumns
from init import Table

def main(test=False,props=None):
    print('This is Time Series Folder and All functions and files will be contained here')
    print('#### For now use only CSV files ####')
    ## Importing File ## We need to store CSV files in test folder in TimeSeries ##
    if not test:
        # For single file manual testing
        path = input('Enter filename for TimeSeries : ').strip()
        df,_ = importFile('./test/' + path,nrows=100)
    else:
        # For Automated Testing
        df,_ = importFile('./test/' + props[path],nrows=100)
    # Stripping trailing and leading spaces and replacing spaces in between with _
    df.columns = [x.strip().replace(' ','_') for x in df.columns]
    datecols = getDateColumns(df.select_dtypes('object'))
    if len(datecols) == 0:
        print('No datecolumns found, QUITTING!')
        return None,None
    else:
        info = getInfo(df.columns,datecols) # Get Target and Primary Date Column
    if not info:
        print('QUITTING!')
        return None,None
    else:
        validation,init_info = INIT(df,info)

if __name__ == '__main__':
    returnValue,numberOfErrors = main()