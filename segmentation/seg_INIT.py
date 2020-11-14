import pandas as pd
import numpy as np
from segmentation.seg_all_other_functions import *
from segmentation.seg_modelling import *
from segmentation.seg_engineerings import *
from sklearn.preprocessing import LabelEncoder,PowerTransformer,MinMaxScaler
from category_encoders import TargetEncoder
import joblib
from segmentation.seg_Viz import *
from imblearn.over_sampling import RandomOverSampler

def INIT(df,info):
    key = info['key']
    cols = info['cols']
    if key:
        cols.append(key)
    df = df[cols]
    # Print columns with missing data in the descending order
    MISSING = pd.DataFrame(((df.isnull().sum().sort_values(ascending=False)*100/len(df)).round(2)),columns=['Missing in %'])[:10]
    print(MISSING)

    ############ TARGET NUMERIC ENGINEERING ###########
    print('\n ### Entering Numeric Engineering of Target### \n')
    if key:
        df = pd.concat([df[key],numeric_engineering(df.drop(key,axis=1))],axis=1)
    else:
        df = numeric_engineering(df)
    ############ TARGET NUMERIC ENGINEERING ###########

    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    # ues = time.time()
    # userInteractVisualization(df,key)
    # uee = time.time()
    # print('Dashboard time taken : {}'.format(uee-ues))
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################

    init_cols = df.columns
    print("Init columns are as follows",init_cols)
    X = df 
    if key:
        X.drop(key,axis=1,inplace=True)
    print(X.head(10))
    del df

    # Remove columns and rows with more than 50% missing values
    print('\nRemoving Rows and Columns with more than 50% missing\n')
    X = DatasetSelection(X)

    print('After removal of highly missing rows and columns')
    MISSING = pd.DataFrame(((X.isnull().sum().sort_values(ascending=False)*100/len(X)).round(2)),columns=['Missing in %'])[:10]
    print(MISSING)

    print('Shape of Xis {}'.format(X.shape))
   
    ######## DATE ENGINEERING #######
    print('\n#### DATE ENGINEERING RUNNING WAIT ####')
#     date_cols = getDateColumns(X.sample(1500) if len(X) > 1500 else X)  # old logic
    date_cols,possible_datecols = getDateColumns(X.sample(1500) if len(X) > 1500 else X)   

    if possible_datecols:
        date_cols= date_cols + possible_datecols

    print('Date Columns found are {}'.format(date_cols))
    if date_cols:
        print('Respective columns will undergo date engineering and will be imputed in the function itself')
        print('\n#### DATE ENGINEERING RUNNING WAIT ####')
        try:
            DATE_DF = date_engineering(X[date_cols],possible_datecols)
            print(DATE_DF.shape)
            DATE_DF.index = X.index
            X.drop(date_cols,axis=1,inplace=True)
        except Exception as exceptionMessage:
            print('#### DATE ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            X.drop(date_cols,axis=1,inplace=True)
            DATE_DF = pd.DataFrame(None)
            date_cols = []
    else:
        DATE_DF = pd.DataFrame(None)
        date_cols = []
    print(' #### DONE ####')
    ######## DATE ENGINEERING #######

    ######## COLUMN SEGREGATION ########
    print('\n ### Entering Segregation Zone ### \n')

    num_df, disc_df, useless_cols = Segregation(X)
    if not disc_df.empty:
        disc_df = disc_df.astype('category')
        disc_cat = {}
        for column in disc_df:
            disc_cat[column] = disc_df[column].cat.categories
    else:
        disc_df = pd.DataFrame()
        disc_cat = {}
    print('Segregation Done!')
    ############# OUTLIER WINSORIZING ###########
    print('\n#### OUTLIER WINSORIZING ####')
    num_df.clip(lower=num_df.quantile(0.1),upper=num_df.quantile(0.9),inplace=True,axis=1)
    print(' #### DONE ####')
    ############# OUTLIER WINSORIZING ###########

    # ############# OUTLIER removal ###########
    # print('\n#### OUTLIER REMOVAL ####')
    # num_df.reset_index(drop=True, inplace=True)
    # disc_df.reset_index(drop=True, inplace=True)
    # y.reset_index(drop=True, inplace=True)
    # ourlierRows = removeOutliers(num_df)
    # num_df.drop(ourlierRows,inplace=True)
    # disc_df.drop(ourlierRows,inplace=True)
    # y.drop(ourlierRows,inplace=True)
    # print(' #### DONE ####')
    # ############# OUTLIER removal ###########

    ######## TEXT ENGINEERING #######
    start1 = time.time()
    start = time.time()
    some_list, remove_list = findReviewColumns(X[useless_cols])  #list1 contains list of usable comment columns, list2 contains list of unusable comment columns
    end = time.time()
    print("Extracting Review Columns time",end-start)
    if (some_list is None):
      TEXT_DF = pd.DataFrame(None)
      lda_models = pd.DataFrame(None)
      print("No review/comment columns found")
    else:
        try:
            print('Respective columns will undergo text engineering and will be imputed in the function itself')
            print('\n#### TEXT ENGINEERING RUNNING WAIT ####')
            print("The review/comment columns found are", some_list)
            start = time.time()
            sentiment_frame = sentiment_analysis(X[some_list])
            sentiment_frame.fillna(value=0.0,inplace=True)
            print(sentiment_frame)
            #TEXT_DF = pd.concat([df, sentiment_frame], axis=1, sort=False)
            TEXT_DF = sentiment_frame.copy()
            TEXT_DF.reset_index(drop=True,inplace=True)
            end = time.time()
            print("Sentiment time",end-start)
            start = time.time()
            new_frame = X[some_list].copy()
            new_frame.fillna(value="None",inplace=True)
            lda_models = pd.DataFrame(index= range(5),columns=['Model'])
            ind = 0

            for col in new_frame.columns:
                topic_frame, lda_model = topicExtraction(new_frame[[col]])
                topic_frame.rename(columns={0:str(col)+"_Topic"},inplace=True)
                print(topic_frame)
                topic_frame.reset_index(drop=True, inplace=True)
                TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
                lda_models['Model'][ind] = lda_model
                ind = ind+1
            end = time.time()
            print("Topic time", end-start)
            X.drop(some_list,axis=1,inplace=True)
            X.drop(remove_list,axis=1, inplace=True)
            lda_models.dropna(axis=0,inplace=True)
        except Exception as e:
            print('#### TEXT ENGINEERING HAD ERRORS ####', e)
            X.drop(some_list,axis=1,inplace=True)
            if(remove_list):
                X.drop(remove_list,axis=1,inplace=True)
            remove_list = []
            some_list = []
            TEXT_DF = pd.DataFrame(None)
            lda_models = pd.DataFrame(None)

    end2= time.time()

    print("total text analytics time taken =", end2-start1)
    print("Text Engineering Result", TEXT_DF)

    #TEXT_DF holds the columns obtained from Text Engineering and
    #X contains the columns after Text imputation

    ########################### TEXT ENGINEERING #############################

    ############# PEARSON CORRELATION ############
    print('\n #### PEARSON CORRELATION ####')
    corr = num_df.corr(method='pearson')
    # If correlation is found to be greater than 85 or equal to 85%, both positive and negative
    corr = corr[(corr >= 0.85)|(corr <= -0.85)]
    for column in corr.columns:
        corr.loc[column][column] = np.nan
    corr.dropna(axis=1,how='all',inplace=True)
    corr.dropna(axis=0,how='all',inplace=True)
    removed_cols = []
    if corr.shape!=(0,0):

        while corr.shape != (0,0):
            corr_dict = {}
            for column in corr.columns:
                corr_dict[corr[column].max()] = column
            try:
                val = max(corr_dict)
                corr.drop(corr_dict[val],inplace=True)
                corr.drop(corr_dict[val],axis=1,inplace=True)
                corr.dropna(axis=1,how='all',inplace=True)
                corr.dropna(axis=0,how='all',inplace=True)
                removed_cols.append(corr_dict[val])
                del corr_dict[val]
            except ValueError:
                break
    num_df.drop(removed_cols,axis=1,inplace=True)
    print('\n{} columns removed which were highly correlated'.format(len(removed_cols)))
    print('The columns removed are {}'.format(removed_cols))
    print(' #### DONE ####')
    ############# PEARSON CORRELATION ############

    num_df.reset_index(drop=True, inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    DATE_DF.reset_index(drop=True, inplace=True)
    TEXT_DF.reset_index(drop=True, inplace=True)
    print('num_df - {}'.format(num_df.shape))
    print('disc_df - {}'.format(disc_df.shape))
    print('DATE_DF - {}'.format(DATE_DF.shape))
    print('TEXT_DF - {}'.format(TEXT_DF.shape))
    concat_list = [num_df,disc_df,DATE_DF,TEXT_DF]
    X = pd.concat(concat_list,axis=1)
    
    # print("This is what the data looks like before going into transformations and encoding",X)
    X = drop_single_valued_features(X)

    ############# ENCODING ############
    print("Encoding categorical variables")
    LE = LabelEncoder()
    print('\n #### LABEL ENCODING ####')
    te_start = time.time()
    X_rec = X[disc_df.columns].apply(LE.fit_transform)
    for col in X_rec.columns:
        X[col] = X_rec[col]
    te_end = time.time()
    print(X.shape)
    print(X.dtypes)
    print('Target Encoding Time taken : {}'.format(te_end-te_start))
    ############# TRANSFORMATIONS ############

    ############# NORMALISATION AND TRANSFORMATIONS #####################
    TrainingColumns = X.columns
    print('\n #### NORMALIZATION ####')
    MM = MinMaxScaler(feature_range=(1, 2))
    X = pd.DataFrame(MM.fit_transform(X),columns=TrainingColumns)
    print(' #### DONE ####')
    print(X.shape)
    print('\n #### POWER TRANSFORMATIONS ####')
    PT = PowerTransformer(method = 'box-cox')
    X = pd.DataFrame(PT.fit_transform(X),columns=TrainingColumns)
    new_mm = MinMaxScaler()
    X = pd.DataFrame(new_mm.fit_transform(X),columns=TrainingColumns)
    print(' #### DONE ####')
    print(X.shape)
    ############# NORMALISATION AND TRANSFORMATIONS ##################### 

    ############# DIMENSIONALITY REDUCTION ##################### 
    n_comp = calculate_n_components(X)
    if n_comp == 1:
        n_comp = 2
    print(f"{n_comp} Principal Components will be generated in dimensionality reduction")
    # print("This is what the data looks like before going into dimensionality reduction",X)
    for col in disc_df.columns:
        X[col] =  X[col].astype('category') #FAMD requires the presence of both numeric and categorical variables
    X_reduced = famd(X,n_comp)


    ############# DIMENSIONALITY REDUCTION ##################### 


    print('\n #### SAVING INIT INFORMATION ####')
    init_info = {'NumericColumns':num_df.columns,'NumericMean':num_df.mean().to_dict(),'DiscreteColumns':disc_df.columns,
                'DateColumns':date_cols, 'PossibleDateColumns':possible_datecols,
                'DateFinalColumns':DATE_DF.columns,'DateMean':DATE_DF.mean().to_dict(),'MinMaxScaler':MM,'PowerTransformer':PT,'TargetLabelEncoder':LE,
                'TrainingColumns':TrainingColumns, 'init_cols':init_cols,
                'KEY':key,'X_train':X,'disc_cat':disc_cat,
                'some_list':some_list,'remove_list':remove_list,'lda_models':lda_models}
    print(' #### DONE ####')
    return init_info
