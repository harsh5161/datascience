from seg_userInputs import *
from seg_INIT import *
from seg_score import *
import time
import pandas as pd
import numpy as np
import swifter
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from seg_engineerings import numeric_engineering
from IPython.display import display
def output():
    pre = pd.read_csv('preview.csv')
    pre.index = np.arange(1,len(pre)+1)

    # Set CSS properties for th elements in dataframe
    th_prop = [
        ('padding', '5px'),
        ('font-family', 'arial'),
        ('font-size', '100%'),
        ('color', 'Black'),
        ('border', '0.5px') ,
        ('border', 'solid black'),
        ('text-align', 'center')
      ]

    # Set CSS properties for td elements in dataframe
    td_prop = [
    #     ('background', 'rgb(232, 247, 252)'),
        ('border', '0.5px'),
        ('border','solid black'),
        ('color', 'black'),
        ('font-family', 'arial')
      ]

    # Set table styles
    styls = [
      dict(selector="th", props=th_prop),
      dict(selector="td", props=td_prop),
      dict(selector="caption", props=[("text-align", "left"),("font-size", "120%"),("color", 'black')])
      ]

    # pre.style.set_table_styles(styls).set_caption("Preview of Test Dataset(100 rows) with Predictions and Actual Values")

    display(pre) # to display only upto 3 decimal places

    # Set CSS properties for th elements in dataframe
    th_props = [
        ('background', 'rgb(12, 64, 90)'),
        ('background', 'linear-gradient(0deg, rgba(21, 112, 157) 0%, rgba(12, 64, 90) 120%)'),
        ('padding', '5px'),
        ('font-family', 'arial'),
        ('font-size', '100%'),
        ('color', 'white'),
        ('border', '0.5px') ,
        ('border', 'solid #0c405a'),
        ('text-align', 'center')
      ]

    # Set CSS properties for td elements in dataframe
    td_props = [
    #     ('background', 'rgb(232, 247, 252)'),
        ('border', '0.5px'),
        ('border','solid #0c405a')    
      ]

    # Set table styles
    styles = [
      dict(selector="th", props=th_props),
      dict(selector="td", props=td_props),
      dict(selector="caption", props=[("text-align", "left"),("font-size", "120%"),("color", 'black')])
      ]

    def color_func(value):           # setting different color for F1 or RMSE column
        if value.name in ['Weighted F1','RMSE']:
            color= '#f7f7ba'
        else:
    #         print(value)
            color= '#e8f7fc'
        return ['background-color: %s' %color]*len(value)


    MC = pd.read_csv('MC.csv')
    if 'Weighted F1' in MC.columns:          # for setting caption
        cap='This table is sorted by F1 Score(Weighted F1), higher the better'
    else:
        cap ='This table is sorted by Root Mean Squared Error(RMSE), lower the better'
    MC.index = np.arange(1,len(MC)+1)       # adjusting index
    if 'Weighted F1' in MC.columns:        #for setting decimal places
        mc= MC.style.set_table_styles(styles).set_caption(cap).apply(color_func, axis=0).set_precision(3)
    else:
        mc= MC.style.set_table_styles(styles).set_caption(cap).apply(color_func, axis=0).set_precision(2)
        
    display(mc)
def seg_main():
    '''
    PROTON MAIN FUNCTION
    '''
    spinnerBool = False
    path = input('Enter the path here : ')
    error = False
    if path:
        df,csvPath = importFile(path,nrows=30)
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        df = dataHandler(df) # If first few rows contains unnecessary info
        info = getUserInput(df)
        if not info:
            error = True
    else:
        df = None
        print('\nQuitting Process\n')
        info = None
        error = True

    te = time.time()
    try:
        if info:
            spinnerBool = True
            ################## TRAINING INIT ##################
            if csvPath:
                path = 'SheetSheetSheet.csv'
            df,_ = importFile(path,nrows=None)
            df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            df = dataHandler(df) # If first few rows contains unnecessary info
            tts = time.time()
            if isinstance(df,pd.DataFrame):
                init_info= INIT(df,info)
            else:
                init_info = None,None
            tte = time.time()
            print('\n TOTAL TRAINING DATA CLEANING AND PLOTS : {}'.format(tte-tts))
            ################## TRAINING INIT ##################
            spinnerBool = False
    except KeyboardInterrupt:
        print('QUITTING!')   
        return None
#     except Exception as e:
#         print('Code did not run completely')
#         print('Code ran into an error')
#         print('The error message received is')
#         print(e)
#         return None
    ee = time.time()
    print('\n#### TOTAL TIME TAKEN : {} ####'.format(ee-te))

    return 1
