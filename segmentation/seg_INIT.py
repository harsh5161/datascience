import pandas as pd
import numpy as np
from segmentation.seg_all_other_functions import *
from segmentation.seg_modelling import *
from segmentation.seg_engineerings import *
from sklearn.preprocessing import LabelEncoder,PowerTransformer,MinMaxScaler
from category_encoders import TargetEncoder
import joblib
from segmentation.seg_Viz import *
from segmentation.seg_modelling import *
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

    print('Shape of X is {}'.format(X.shape))
    

    ######## LAT-LONG ENGINEERING #########
    print('\n#### LAT-LONG ENGINEERING RUNNING WAIT ####')

    lat,lon,lat_lon_cols  = findLatLong(X)
    print("lat columns are",lat)
    print("long columns are",lon)
    print("lat-long columns are",lat_lon_cols)
    if (lat and lon) or lat_lon_cols:
        print('Respective columns will undergo lat-long engineering and will be imputed in the function itself')
        try:
            print("Lat-Long Engineering Running...")
            LAT_LONG_DF = latlongEngineering(X,lat,lon,lat_lon_cols)
            print(LAT_LONG_DF)
            print(LAT_LONG_DF.shape)
            if lat: X.drop(lat,axis=1,inplace=True)
            if lon: X.drop(lon,axis=1,inplace=True)
            if lat_lon_cols: X.drop(lat_lon_cols,axis=1,inplace=True)
        except Exception as exceptionMessage:
            print('#### LAT-LONG ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            if lat: X.drop(lat,axis=1,inplace=True)
            if lon: X.drop(lon,axis=1,inplace=True)
            if lat_lon_cols: X.drop(lat_lon_cols,axis=1,inplace=True)
            LAT_LONG_DF = pd.DataFrame(None)
            lat = []
            lon = []
            lat_lon_cols = []
    else:
        print("No Latitude or Longitude Columns found")
        LAT_LONG_DF = pd.DataFrame(None)
        lat = []
        lon = []
        lat_lon_cols = []



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

    ######## EMAIL URL ENGINEERING ########
    obj_df  = X.select_dtypes('object') # Sending in only object dtype columns
    short_obj_df = obj_df.astype(str).sample(3000).dropna(how='all') if len(obj_df)>3000 else obj_df.astype(str).dropna(how='all')
    email_cols = identifyEmailUrlColumns(short_obj_df,email=True)
    if len(email_cols)>0:
        try:
            EMAIL_DF = emailUrlEngineering(X[email_cols])
            X.drop(email_cols,axis=1,inplace=True) # If any email columns found, we drop it after engineering
            EMAIL_DF.reset_index(drop=True)
        except Exception as e:
            print('### EMAIL ENGINEERING HAD ERRORS ###')
            print(f'The Exception message is {e}')
            X.drop(email_cols,axis=1,inplace=True)
            EMAIL_DF = pd.DataFrame(None)
            email_cols  = []
    else:
        print("No Email columns found")
        EMAIL_DF = pd.DataFrame(None)
        email_cols = []

    url_cols = identifyEmailUrlColumns(short_obj_df,email=False)
    if len(url_cols)>0:
        try:
            URL_DF = emailUrlEngineering(X[url_cols])
            X.drop(url_cols,axis=1) # If any email columns found, we drop it post engineering
            URL_DF.reset_index(drop=True)
        except Exception as e:
            print('### URL ENGINEERING HAD ERRORS ###')
            print(f'The Exception is as {e}')
            X.drop(url_cols,axis=1)
            URL_DF = pd.DataFrame(None)
            url_cols  = []
    else:
        print("No URL columns found")
        URL_DF = pd.DataFrame(None)
        url_cols = []
    ######## EMAIL URL ENGINEERING ########


    ######## COLUMN SEGREGATION ########
    print('\n ### Entering Segregation Zone ### \n')

    num_df, disc_df, useless_cols = Segregation(X)
    if not disc_df.empty:
        DISC_VAL = True
        disc_df = disc_df.astype('category')
        disc_cat = {}
        for column in disc_df:
            disc_cat[column] = disc_df[column].cat.categories
    else:
        DISC_VAL = False
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
    LAT_LONG_DF.reset_index(drop=True, inplace=True)
    EMAIL_DF.reset_index(drop=True, inplace=True)
    URL_DF.reset_index(drop=True, inplace= True)
    print('num_df - {}'.format(num_df.shape))
    print('disc_df - {}'.format(disc_df.shape))
    print('DATE_DF - {}'.format(DATE_DF.shape))
    print('TEXT_DF - {}'.format(TEXT_DF.shape))
    print('LAT_LONG_DF - {}'.format(LAT_LONG_DF.shape))
    print('EMAIL_DF - {}'.format(EMAIL_DF.shape))
    print('URL_DF -  {}'.format(URL_DF,shape))
    concat_list = [num_df,disc_df,DATE_DF,TEXT_DF,LAT_LONG_DF,EMAIL_DF,URL_DF]
    X = pd.concat(concat_list,axis=1)
    
    # print("This is what the data looks like before going into transformations and encoding",X)
    single_vals = drop_single_valued_features(X) #change this part to accomodate for the changes in num_df
    for single in single_vals:
        if single in num_df.columns:
            num_df.drop(single,axis=1,inplace=True)
        elif single in disc_df.columns:
            disc_df.drop(single,axis=1,inplace=True)
        elif single in DATE_DF.columns:
            DATE_DF.drop(single,axis=1,inplace=True)
        elif single in TEXT_DF.columns:
            TEXT_DF.drop(single, axis=1, inplace=True)
        elif single in LAT_LONG_DF.columns:
            LAT_LONG_DF.drop(single, axis=1, inplace=True)
        elif single in EMAIL_DF.columns:
            EMAIL_DF.drop(single, axis =1, inplace=True)
        elif single in URL_DF.columns:
            URL_DF.drop(single, axis=1, inplace=True)
        
        X.drop(single, axis=1, inplace=True)


    #Making a copy of the dataframe that will required in cluster profiling
    X_df = X.copy()
    ############# ENCODING ############
    if not disc_df.empty:
        print("\nEncoding categorical variables")
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
    else:
        print("No categorical columns found, so no encoding required")
        LE = None
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

    print(f'The columns that are going into the dimensionality reduction are as follows {X.columns}')

    ############# DIMENSIONALITY REDUCTION ##################### 
    n_comp = calculate_n_components(X)
    if n_comp == 1:
        n_comp = 2
    print(f"{n_comp} Principal Components will be generated in dimensionality reduction")
    # print("This is what the data looks like before going into dimensionality reduction",X)
    print("disc_df columns", disc_df.columns)
    for col in disc_df.columns:
        X[col] =  X[col].astype(str) #FAMD requires the presence of both numeric and categorical variables
    X_reduced = dimensionality_reduction(X,n_comp,DISC_VAL)
    # X_reduced = pd.DataFrame(X_reduced) #Convert to dataframe to be used in modelling
    # print(isinstance(X_reduced,pd.DataFrame))

    ############# DIMENSIONALITY REDUCTION ##################### 

    ############# CLUSTERING ##################### 

    algo = Segmentation()
    segdata= algo.clustering_algorithms(X_reduced,X_df)
    
    # Displaying Cluster magnitudes
    ClusterMags(segdata)
    
    
    # Cluster Profiling
    ClusterProfiling(segdata, num_df, disc_df)


    ############# CLUSTERING ##################### 
    print('\n #### SAVING INIT INFORMATION ####')
    init_info = {'NumericColumns':num_df.columns,'NumericMean':num_df.mean().to_dict(),'DiscreteColumns':disc_df.columns,
                'DateColumns':date_cols, 'PossibleDateColumns':possible_datecols,
                'DateFinalColumns':DATE_DF.columns,'DateMean':DATE_DF.mean().to_dict(),'MinMaxScaler':MM,'PowerTransformer':PT,'TargetLabelEncoder':LE,
                'TrainingColumns':TrainingColumns, 'init_cols':init_cols,
                'KEY':key,'X_train':X,'disc_cat':disc_cat,
                'some_list':some_list,'remove_list':remove_list,'lda_models':lda_models,'lat':lat,'lon':lon,'lat_lon_cols':lat_lon_cols, 'email_cols':email_cols,'url_cols':url_cols}
    print(' #### DONE ####')
    return init_info
