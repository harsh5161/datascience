import pandas as pd
import numpy as np
from all_other_functions import *
from modelling import *
from engineerings import *
from sklearn.preprocessing import LabelEncoder,PowerTransformer,MinMaxScaler
from category_encoders import TargetEncoder
import joblib
from Viz import *
from imblearn.over_sampling import RandomOverSampler
from tabulate import tabulate 

def INIT(df,info):
    print("Length of the dataframe entering INIT is: ", df.shape[0])
    target = info['target']
    key = info['key']
    cols = info['cols']
    cols.append(target)
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

    # Find Classification or Regression
    class_or_Reg = None #INIT
    print('Target Analysis')
    class_or_Reg = targetAnalysis(df[target])
    if not class_or_Reg:
        print('\nExecution stops as We cant deal with such a target')
        return None,None

    if class_or_Reg == 'Classification':
        # Remove Classes with less than 0.05% occurence
        returnValue = removeLowClass(df,target)
        if isinstance(returnValue,pd.DataFrame):
            df = returnValue
        elif not returnValue:
            return None,None
        else:
            pass


    print('{} column needs {}'.format(target,class_or_Reg))

    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    if not info['graph']:
        if key:
            x = df.drop(key,axis=1)
            eda_df = x.copy()
        else: 
            eda_df = df.copy()
    else:
        eda_df = pd.DataFrame()
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################

    # Remove all rows with Target Column Empty
    beforeIndex = df.index
    if class_or_Reg == 'Classification':
        df[target] = df[target].astype(str)
        df = df[df[target].str.strip().astype(bool)] #vectorized format to remove rows with target values that contain ''
    df.dropna(axis=0,subset=[target],inplace=True)
    afterIndex = df.index
    rowsRemoved = list(set(beforeIndex)-set(afterIndex))
    print('\n {} rows were removed since target had these missing'.format(len(rowsRemoved)))
    print(f'Target Unique values and count \n {df[target].value_counts()} \n Unique values \n {df[target].nunique()}')
    del beforeIndex,afterIndex
    if class_or_Reg == 'Classification':
        target_unique = df[target].unique().tolist()
        target_unique.sort()

    ############# TRAIN VALIDATION SPLIT ###########
    if class_or_Reg == 'Classification':
        LE = LabelEncoder()
        df[target] = LE.fit_transform(df[target])
        try:
            train,validation = train_test_split(df,test_size=0.2,random_state=1,stratify=df[target])
        except:
            train,validation = train_test_split(df,test_size=0.2,random_state=1)
    else:
        LE = None
        df[target] = df[target].astype(float)
        df[target].clip(lower=df[target].quantile(0.1),upper=df[target].quantile(0.9),inplace=True)
        try:
            train,validation = train_test_split(df,test_size=0.2,random_state=1,stratify=df[target])
        except:
            train,validation = train_test_split(df,test_size=0.2,random_state=1)
    ############# TRAIN VALIDATION SPLIT ###########

    init_cols = df.columns
    X = train.drop(target,axis=1)
    y = train[target]
    if key:
        X.drop(key,axis=1,inplace=True)
    del train
    del df

    # Remove columns and rows with more than 50% missing values
    print('\nRemoving Rows and Columns with more than 50% missing\n')
    X,y = DatasetSelection(X,y)

    print('After removal of highly missing rows and columns')
    MISSING = pd.DataFrame(((X.isnull().sum().sort_values(ascending=False)*100/len(X)).round(2)),columns=['Missing in %'])[:10]
    print(MISSING)

    # print("length of X going into sampling!!!!!!!!!!!!",len(X))
    # print("length of y going into sampling!!!!!!!!!!!!",len(y))
    # # Sampling Data
    # print('Sampling Data!')
    # X,y = data_model_select(X,y)
    # print('After sampling:')
    print('Shape of X_train is {}'.format(X.shape))
    print('Shape of y_train is {}'.format(y.shape))
    if class_or_Reg == 'Classification':
        print('printing target variable distribution for classification:\n')
        print(pd.Series(y).value_counts(normalize=True))



    ####### Logic to remove labels from validation that arent present in training ########## 
    print("Checking if there are labels present in validation that arent present in training")
    if class_or_Reg == 'Classification':
        stored_labels = y.value_counts().keys().to_list() #To add future logic if required to only use the levels present in training for scoring purpose
        # print("STORED LABELS ",stored_labels)
        init_len = len(validation)
        validation[target] = validation[target].apply(lambda x: format_y_labels(x,stored_labels))
        validation.dropna(axis=0,subset=[target],inplace=True)
        validation.reset_index(drop=True,inplace=True)
        print(f"Number of columns dropped from validation dataset due to mismatched target variable is : {init_len-len(validation)}")
    else:
        stored_labels = None



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
            DATE_DF,dropped_cols = date_engineering(X[date_cols],possible_datecols)
            if dropped_cols:
                for val in dropped_cols :
                    if val in date_cols[:]:
                        date_cols.remove(val)
                    if val in possible_datecols[:]:
                        possible_datecols.remove(val)
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
    print('\n#### EMAIL ENGINEERING RUNNING WAIT ####')
    obj_df  = X.select_dtypes('object') # Sending in only object dtype columns
    short_obj_df = obj_df.astype(str).sample(3000).dropna(how='all') if len(obj_df)>3000 else obj_df.astype(str).dropna(how='all')
    email_cols = identifyEmailColumns(short_obj_df)
    if len(email_cols)>0:
        try:
            EMAIL_DF = emailUrlEngineering(X[email_cols],email=True)
            X.drop(email_cols,axis=1,inplace=True) # If any email columns found, we drop it after engineering
            EMAIL_DF.reset_index(drop=True)
            print(EMAIL_DF)
            print(EMAIL_DF.shape)
            short_obj_df.drop(email_cols,axis=1,inplace=True)
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

    print('\n#### URL ENGINEERING RUNNING WAIT ####')
    url_cols = findURLS(short_obj_df)
    if len(url_cols)>0:
        try:
            URL_DF = URlEngineering(X[url_cols])
            X.drop(url_cols,axis=1,inplace=True) # If any email columns found, we drop it post engineering
            URL_DF.reset_index(drop=True)
            print(URL_DF)
            print(URL_DF.shape)
        except Exception as e:
            print('### URL ENGINEERING HAD ERRORS ###')
            print(f'The Exception is as {e}')
            X.drop(url_cols,axis=1,inplace=True)
            URL_DF = pd.DataFrame(None)
            url_cols  = []
    else:
        print("No URL columns found")
        URL_DF = pd.DataFrame(None)
        url_cols = []
    ######## EMAIL URL ENGINEERING ########
    # Additional Logic below because EMAIL and URL Engineering can choose not to work if  there's a lot of missing column
    if EMAIL_DF.empty:
        EMAIL_STATUS = True
    else:
        EMAIL_STATUS = False
    # if URL_DF.empty:
    #     URL_STATUS = True
    # else:
    #     URL_STATUS =  False

    X.reset_index(drop=True,inplace=True)
    DATE_DF.reset_index(drop=True, inplace=True)
    LAT_LONG_DF.reset_index(drop=True, inplace=True)
    EMAIL_DF.reset_index(drop=True, inplace=True)
    URL_DF.reset_index(drop=True, inplace=True)
    concat_list = [X,DATE_DF,LAT_LONG_DF,EMAIL_DF,URL_DF]
    X = pd.concat(concat_list,axis=1)

    ######## COLUMN SEGREGATION ########
    print('\n ### Entering Segregation Zone ### \n')

    num_df, disc_df, useless_cols = Segregation(X,y)
    if not disc_df.empty:
        disc_df = disc_df.astype('category')
        disc_cat = {}
        for column in disc_df:
            disc_cat[column] = disc_df[column].cat.categories
    else:
        disc_df = pd.DataFrame()
        disc_cat = {}
    print('Segregation Done!')
    print('Useless Columns are',useless_cols)

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
            text_analytics(sentiment_frame,new_frame,class_or_Reg,y,LE)
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
    disc_df.reset_index(drop=True,inplace=True)
    num_df.reset_index(drop=True,inplace=True)
    #This is the num_df,disc_df that we will be checking for in the scoring file (Before adding the TEXT_DF variables that are actually artificially created by us)
    NumColumns = num_df.columns
    NumMean = num_df.mean().to_dict()
    DiscColumns = disc_df.columns
    TEXT_DF.reset_index(drop=True, inplace=True)
    if not TEXT_DF.empty:
        for col in TEXT_DF.columns:
            if col.find("_Topic")!=-1:
                disc_df = pd.concat([disc_df,pd.DataFrame(TEXT_DF[col])],axis=1)
            else:
                num_df = pd.concat([num_df,pd.DataFrame(TEXT_DF[col])],axis=1)



    ############# OUTLIER WINSORIZING ###########
    print('\n#### OUTLIER WINSORIZING ####')
    bef_out = num_df.shape[0]
    num_df.clip(lower=num_df.quantile(0.1),upper=num_df.quantile(0.9),inplace=True,axis=1)
    print(f'No. of outliers handled : {bef_out-num_df.shape[0]}')
    print(' #### DONE ####')
    ############# OUTLIER WINSORIZING ###########

    ############# PEARSON CORRELATION ############
    print(f"The shape before Pearson's {num_df.shape}")
    print('\n #### PEARSON CORRELATION ####')
    # corr = num_df.corr(method='pearson')
    corr = np.corrcoef(num_df.values, rowvar=False) 
    corr = pd.DataFrame(corr, columns = num_df.columns.to_list())
    # print("Initial correlation matrix",corr)
    corr = corr.where(np.tril(np.ones(corr.shape),k=-1).astype(np.bool))
    # print("The Lower Triangular matrix is \n",corr)
    col_counter = {}
    for col in corr.columns:
        ser =  corr[col].apply(lambda x: findDefaulters(x)).to_list()
        # print(f"{col} : {ser.count(True)}")
        if ser.count(True) >0:
            col_counter[col] = ser.count(True)
    print("List of columns and how many columns they are corelated to",col_counter)
    if not col_counter:
        print("No columns are correlated")
    else:
        while col_counter:
            # print(f"Len of the col_counter",len(col_counter))
            num_df,col_counter = pearsonmaker(num_df,col_counter)

    # print(f'Pearsons Matrix \n {pd.DataFrame(num_df.corr())}')
    print(f"The shape after Pearson's {num_df.shape}")
    print(' #### DONE ####')
    PearsonsColumns = num_df.columns 
    ############# PEARSON CORRELATION ############

    y.reset_index(drop=True, inplace=True)
    num_df.reset_index(drop=True, inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    print('num_df - {}'.format(num_df.shape))
    print('disc_df - {}'.format(disc_df.shape))
    print('DATE_DF - {}'.format(DATE_DF.shape))
    print('TEXT_DF - {}'.format(TEXT_DF.shape))
    print('LAT_LONG_DF - {}'.format(LAT_LONG_DF.shape))
    print('EMAIL_DF - {}'.format(EMAIL_DF.shape))
    print('URL_DF - {}'.format(URL_DF.shape))
    if num_df.shape[1] != 0:    #Some datasets may contain only categorical data
        concat_list = [num_df,disc_df]
        X = pd.concat(concat_list,axis=1)
    else:
        X = disc_df
    X_old = X.copy()# X_old is before Target Encoding

    ############# TARGET ENCODING ############
    temp = disc_df.columns.tolist()
    X_1 = X.copy()
    TE = TargetEncoder(cols=disc_df.columns)
    print('\n #### TARGET ENCODING ####')
    te_start = time.time()
    X = TE.fit_transform(X,y)
    te_end = time.time()
    print(X.shape)
    print(y.shape)
    print('Target Encoding Time taken : {}'.format(te_end-te_start))
    ############# TARGET ENCODING ############
    X_2 = X.copy()
    ############# FEATURE SELECTION AND PLOTS ############
    if (len(X.columns)>=10):
        print('\n #### FEATURE SELECTION ####')
        fe_s = time.time()
        rem,feat_df = FeatureSelection(X,y,class_or_Reg)
        fe_e = time.time()
        print(X.shape)
        print(y.shape)
        print('Feature Selection Time taken : {}'.format(fe_e-fe_s))
        X.drop(rem,axis=1,inplace=True)
        X_old.drop(rem,axis=1,inplace=True)  # removing columns through feature selection without target encoding
        fe_s = time.time()

        try:
            featureSelectionPlot(feat_df[:15])
        except:
            print('\nFEATURE SELECTION PLOT DID NOT RUN SUCCESSFULLY!')
        fe_e = time.time()
        print('Feature Selection Plot Time taken : {}'.format(fe_e-fe_s))
        print(X.shape)
        print(y.shape)
    else:
        print('\n #### FEATURE SELECTION SKIPPED BECAUSE COLUMNS LESS THAN 10 ####')
    ############# FEATURE SELECTION AND PLOTS #####################
    ##################### Checking for constant columns ###################

    for col in X.columns:
        if X[col].nunique() == 1:
            X.drop(col,axis=1,inplace=True)
            X_old.drop(col,axis=1,inplace=True)
            print(f"Dropping column {col} because it only contains one value throughout the column")

    TrainingColumns = X.columns
    if TrainingColumns.empty:
        print("Error : There are no informative columns present in the dataset. Please collect more information.")
        return None, None 

    ##################### Checking for constant columns ###################

    ############# CART DECISION TREE VISUALIZATION #####################
    if not info['graph']:
        print('\n #### DECISION TREE VISUALIZATION ####')
        X_cart = X_old.copy() #making a specific copy for CART because X_old is also being used by sample equation
        X_cart.reset_index(drop=True, inplace=True)
        y_cart = y.copy()
        y_cart.reset_index(drop=True, inplace=True)
        if class_or_Reg=='Classification':
            print("Length of X_cart and y_cart",len(X_cart),"---",len(y_cart))
            ros = RandomOverSampler(sampling_strategy='minority')
            X_cart_res, y_cart_res = ros.fit_resample(X_cart,y_cart)
            print("Length of X_cart_res and y_cart_res",len(X_cart_res),"---",len(y_cart_res))
            passingList = y_cart_res.value_counts(normalize=True).values
            cart_list =[X_cart_res,y_cart_res]
            cart_df = pd.concat(cart_list,axis=1)
        else:
            passingList = []
            cart_list =[X_cart,y_cart]
            cart_df = pd.concat(cart_list,axis=1)
        try:
            cart_decisiontree(cart_df,target,class_or_Reg,passingList)
            if class_or_Reg  == 'Classification':
                print(f'{target_unique[0]} is alphabetically lower so its on the left and {target_unique[-1]} is alphabetically higher so its on the right')
        except Exception as e:
            print(e)
            print('#### CART VISUALIZATION DID NOT RUN AND HAD ERRORS ####')
    ############# CART DECISION TREE VISUALIZATION #####################   

    ############# RULE TREE VISUALIZATION #####################   

    if class_or_Reg == 'Classification':
        ros = RandomOverSampler(sampling_strategy='minority')
        X_rt, y_rt = ros.fit_resample(X,y)
        rule_val,rule_model = rules_tree(X_old,y_rt,class_or_Reg,X_rt,LE)
        feat = X_rt.columns.tolist()
    else:
        rule_val,rule_model = rules_tree(X_old,y,class_or_Reg,X,LE)
        feat = X.columns.tolist()

    imps = rule_model.feature_importances_
    indices = np.argsort(imps)
    if len(feat)>10:
        num = 10
    else:
        num = len(feat)
    feat = [feat[i] for i in indices[-num:]]
    feat = [x for x in feat[:] if x in disc_df.columns.tolist()]
    # print(f'Top important discrete features from Rule Tree are: {feat}')
    if rule_val:
        print('Rule Tree Generated')
    else:
        print('Rule Tree not Generated')
    ############# RULE TREE VISUALIZATION ##################### 
    ############# NORMALISATION AND TRANSFORMATIONS ##################### 
    print('\n #### SCALING ####')
    MM = MinMaxScaler(feature_range=(1, 2))
    X = pd.DataFrame(MM.fit_transform(X),columns=TrainingColumns)
    print(' #### SCALING DONE ####')
    print(X.shape)
    print(y.shape)
    print('\n #### POWER TRANSFORMATIONS ####')
    PT = PowerTransformer(method = 'box-cox')
    X = pd.DataFrame(PT.fit_transform(X),columns=TrainingColumns)
    new_mm = MinMaxScaler()
    X = pd.DataFrame(new_mm.fit_transform(X),columns=TrainingColumns)
    print(' #### POWER TRANSFORMATIONS DONE ####')
    print(X.shape)
    print(y.shape)
    ############# NORMALISATION AND TRANSFORMATIONS ##################### 
    ############# SAMPLE EQUATION ##################### 
    print('\n #### Printing Sample Equation of the DATA ####')
    # The few lines below consider only columns of object/category type that remained after feature selection
    try:
        vis_disc_cols = []
        for col in disc_df.columns:
            if col in X.columns:
                vis_disc_cols.append(col)
        selected_obj_cols = SampleEquation(X_old.copy(),y.copy(),class_or_Reg,vis_disc_cols,LE,feat)
        
    except Exception as e:
        print(e)
        print('SAMPLE EQUATION DID NOT RUN AND HAD ERRORS!!!')
    print(' #### DONE ####')

    print('\nThis is final shape of X_train : {}'.format(X.shape))
    print('This is final shape of Y_train : {}\n'.format(y.shape))
    ############# SAMPLE EQUATION ##################### 

    ############# SHAP ENCODINGS ##################### 

    for val in temp[:]:
        if val not in X.columns.tolist():
            temp.remove(val)
    encoded_disc = []
    for col in temp:
        try:
            print('Generating SHAP encodings...')
            encoding_df = pd.DataFrame()
            encoding_df[f'{col}'] = X_1[col].unique()
            encoding_df['Encoding'] = X_2[col].unique()
            encoding_df['Encoding'] = encoding_df['Encoding'].round(decimals=2)
            encoding_df = encoding_df.sort_values('Encoding',ignore_index=True,ascending=True)
            print (tabulate(encoding_df, headers='keys', tablefmt='psql', showindex=False)) #to output on python not for webapp
            if len(encoding_df)<5:
                encoded_disc.append(encoding_df)
        except:
            pass
    ############# SHAP ENCODINGS ##################### 

    ############# RT ENCODINGS ##################### 
    print("Generating Rule Tree Encodings.....")
    print(selected_obj_cols)
    for val in selected_obj_cols:
        try:
            rule_df = pd.DataFrame()
            rule_df[f'{val}'] = X_1[val].unique()
            rule_df['Encoding'] = X_2[val].unique()
            rule_df['Encoding'] = rule_df['Encoding'].round(decimals=2)
            rule_df = rule_df.sort_values('Encoding',ignore_index=True,ascending=True) 
            #extract rule_df here and embed onto webapp under ruletree
            print (tabulate(rule_df, headers='keys', tablefmt='psql', showindex=False)) #to output on python not for webapp
        except:
            pass
    ############# RT ENCODINGS ##################### 

    print('\n #### SAVING INIT INFORMATION ####')
    init_info = {'NumericColumns':NumColumns,'NumericMean':NumMean,'DiscreteColumns':DiscColumns, 'StoredLabels':stored_labels,
                'DateColumns':date_cols, 'PossibleDateColumns':possible_datecols,'PearsonsColumns':PearsonsColumns,
                'DateFinalColumns':DATE_DF.columns,'DateMean':DATE_DF.mean().to_dict(),
                'TargetEncoder':TE,'MinMaxScaler':MM,'PowerTransformer':PT,'TargetLabelEncoder':LE,'Target':target,
                 'TrainingColumns':TrainingColumns, 'init_cols':init_cols,
                'ML':class_or_Reg,'KEY':key,'X_train':X,'y_train':y,'disc_cat':disc_cat,'q_s':info['q_s'],
                'some_list':some_list,'remove_list':remove_list,'lda_models':lda_models,'lat':lat,'lon':lon,'lat_lon_cols':lat_lon_cols,
                 'email_cols':email_cols,'url_cols':url_cols,'EMAIL_STATUS':EMAIL_STATUS,'eda_df' : eda_df,'rule_model':rule_model,'encoded_disc':encoded_disc}
    print(' #### DONE ####')
    return init_info,validation
    
