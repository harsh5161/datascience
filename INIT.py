import pandas as pd
import numpy as np
from all_other_functions import *
from modelling import *
from engineerings import *
from sklearn.preprocessing import LabelEncoder,PowerTransformer,MinMaxScaler
from category_encoders import TargetEncoder
import joblib

def INIT(df,info):
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
    # ues = time.time()
    # if key:userInteractVisualization(df.drop(key,axis=1),target)
    # else:userInteractVisualization(df,target)
    # uee = time.time()
    # print('Bi/Uni Variate Plotter time taken : {}'.format(uee-ues))
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################
    ######################### UNIVARIATE and BIVARIATE GRAPHS #########################

    # Remove all rows with Target Column Empty
    beforeIndex = df.index
    df.dropna(subset=[target],inplace=True)
    afterIndex = df.index
    rowsRemoved = list(set(beforeIndex)-set(afterIndex))
    print('\n {} rows were removed since target had these missing'.format(len(rowsRemoved)))
    del beforeIndex,afterIndex

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

    # Separate comment columns before removing missing cols
    if commentCols:
        commentDF = X[commentCols]
        X.drop(commentCols,axis=1,inplace=True)

    # Remove columns and rows with more than 50% missing values
    print('\nRemoving Rows and Columns with more than 50% missing\n')
    X,y = DatasetSelection(X,y)
    if commentCols:
        commentDF = commentDF.loc[X.index,:]
        X = pd.concat([X,commentDF],axis=1)
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)

    # Sampling Data
    print('\nSampling Data!\n')
    X,y = data_model_select(X,y)
    print('After sampling:')
    print('Shape of X_train is {}'.format(X.shape))
    print('Shape of y_train is {}'.format(y.shape))
    if class_or_Reg == 'Classification':
        print('printing target variable distribution for classification:\n')
        print(pd.Series(y).value_counts(normalize=True))

    ######## DATE ENGINEERING #######
    print('\n#### DATE ENGINEERING RUNNING WAIT ####')
    date_cols = getDateColumns(X.sample(1500) if len(X) > 1500 else X)
    print('Date Columns found are {}'.format(date_cols))
    if date_cols:
        print('Respective columns will undergo date engineering and will be imputed in the function itself')
        print('\n#### DATE ENGINEERING RUNNING WAIT ####')
        try:
            DATE_DF = date_engineering(X[date_cols])
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
    # Feature Reduction and Segregation of discrete columns
    # joblib.dump(X,'seg');return None,None
    num_df, disc_df, useless_cols = Segregation(X)
    disc_df = disc_df.astype('category')
    disc_cat = {}
    for column in disc_df:
        disc_cat[column] = disc_df[column].cat.categories

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
        except:
            print('#### TEXT ENGINEERING HAD ERRORS ####')
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
    corr = corr[(corr >= 0.85)]
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

    y.reset_index(drop=True, inplace=True)
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
    X_old = X.copy() # X_old is before Target Encoding
    print('\n #### TRANSFORMATIONS ####')
    TE = TargetEncoder(cols=disc_df.columns)
    print('\n #### TARGET ENCODING ####')
    te_start = time.time()
    X = TE.fit_transform(X,y)
    te_end = time.time()
    print('Target Encoding Time taken : {}'.format(te_end-te_start))
    print('\n #### FEATURE SELECTION ####')
    fe_s = time.time()
    rem,feat_df = FeatureSelection(X,y,class_or_Reg)
    fe_e = time.time()
    print('Feature Selection Time taken : {}'.format(fe_e-fe_s))
    X.drop(rem,axis=1,inplace=True)
    fe_s = time.time()

    try:
        featureSelectionPlot(feat_df[:15])
    except:
        print('\nFEATURE SELECTION PLOT DID NOT RUN SUCCESSFULLY!')
    fe_e = time.time()
    print('Feature Selection Plot Time taken : {}'.format(fe_e-fe_s))

    print('\n #### DECISION TREE VISUALIZATION ####')
    try:
        Visualization(X,y,class_or_Reg,LE)
    except:
        print('#### VISUALIZATION DID NOT RUN AND HAD ERRORS ####')

    TrainingColumns = X.columns

    print('\n #### Printing Sample Equation of the DATA ####')
    try:
        SampleEquation(X_old[X.columns],y,class_or_Reg)
    except:
        print('SAMPLE EQUATION DID NOT RUN AND HAD ERRORS!!!')
    print(' #### DONE ####')

    print('\n #### NORMALIZATION ####')
    # MM = None
    MM = MinMaxScaler(feature_range=(1, 2))
    X = pd.DataFrame(MM.fit_transform(X),columns=TrainingColumns)
    print(' #### DONE ####')
    print('\n #### POWER TRANSFORMATIONS ####')
    PT = PowerTransformer(method = 'box-cox')
    X = pd.DataFrame(PT.fit_transform(X),columns=TrainingColumns)
    new_mm = MinMaxScaler()
    X = pd.DataFrame(new_mm.fit_transform(X),columns=TrainingColumns)
    # PT = None
    print(' #### DONE ####')

    print('\n #### SAVING MODEL INFORMATION ####')
    init_info = {'NumericColumns':num_df.columns,'NumericMean':num_df.mean().to_dict(),'DiscreteColumns':disc_df.columns,
                'DateColumns':date_cols,'DateFinalColumns':DATE_DF.columns,'DateMean':DATE_DF.mean().to_dict(),
                'TargetEncoder':TE,'MinMaxScaler':MM,'PowerTransformer':PT,'TargetLabelEncoder':LE,'Target':target,
                 'TrainingColumns':TrainingColumns, 'init_cols':init_cols,
                'ML':class_or_Reg,'KEY':key,'X_train':X,'y_train':y,'disc_cat':disc_cat,'q_s':info['q_s'],
                'some_list':some_list,'remove_list':remove_list,'lda_models':lda_models}
    print(' #### DONE ####')
    return init_info,validation
