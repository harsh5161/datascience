import time
import pandas as pd
import numpy as np
from xgboost import *
from tqdm.notebook import tqdm
from segmentation.seg_modelling import *
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import pydotplus
from category_encoders import TargetEncoder
from missingpy import MissForest
def targetAnalysis(df):
    print('\n### TARGET ANALYSIS ENTERED ###')
    Type = str(df.dtypes)
    # IF INT OR FLOAT IN TARGET, and IF NUMBER OF UNIQUE IS LESS, CLASSIFICATION, ELSE, REGRESSION
    print('Target has {} unique values'.format(df.nunique()))
    print('Printing % occurence of each class in target Column')
    print(df.value_counts(normalize=True))
    if ('int' in Type) or ('float' in Type):
        if df.nunique() < 5:
            return 'Classification'
        else:
            return 'Regression'

    else:
        if df.nunique() < 5:
            return 'Classification'
        else:
            return None
def ForestImputer(num_df,disc_df,target):
    num_df.reset_index(drop=True,inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    concat_list = [num_df,disc_df]
    df = pd.concat(concat_list,axis=1)
    #testing purposes
    # print("checking Initial Presence of Missing Per Column",df.isna().any())
    # print("Printing the total number of columns present in the dataframe",df.shape[1])
    # print("Printing the count of column with missing values",df.isna().any().sum())
    #testing purposes
    cat_list = disc_df.columns.to_list()
    num_list = num_df.columns.to_list()

    forester = 1 #MissForest imputation will be done
    if df.isna().any().sum() > 0.50 * len(df.columns):
        forester = 0 # Mean imputation will be done
        print("Dataframe has too many columns with null values, hence mean imputation will be done")
        df.fillna(value=df.mean(),inplace=True)
        print("Printing the missing values after mean imputation logic",df.isna().any().sum())

    TE = TargetEncoder(cols=cat_list) #target encoding t=categorical variables
    df1 = TE.fit_transform(df,target)

    class_or_reg = targetAnalysis(target)
    if class_or_reg == 'Classification':
        imputer = MissForest(max_iter=5,copy=True,max_depth=10,class_weight="balanced",n_estimators=100)
    elif class_or_reg == 'Regression':
        imputer = MissForest(max_iter=5,copy=True,max_depth=10,n_estimators=100)
    if forester ==1:
        start = time.time()
        print("MissForest imputation will be done...")
        X = imputer.fit_transform(df1)
        print("Time taken for the completion of MissForest is :",time.time()-start)
    elif forester ==0:
        X = df1
    df2 = pd.DataFrame(X,index=df1.index,columns=df1.columns) #converting numpy array back to dataframe after transformation call
    # print("Checking Final Presence of Missing Per Column",df2.isna().any())
    # print("!!!---!!!")
    for col in df.columns:
        if col in df2.columns:
            if col not in cat_list:
                df[col] = df2[col]
    numeric = df[num_list]
    return numeric

def Segregation(df,y):
    print('\n#### Entering Segregation ####\n')
    start = time.time()
    num = df._get_numeric_data().columns
    obj = list(set(df.columns)-set(num))

    nu = df[num].nunique()>8
    numeric = df[nu[nu == True].index]
    cat_num = df[list(set(num) - set(numeric.columns))]
    cat_num.fillna(cat_num.median(skipna=True),inplace=True)

    print('There are {} pure numeric columns'.format((len(numeric.columns))))
    print('There are {} categorical numeric columns\n'.format((len(cat_num.columns))))
    print('The pure numeric columns are {}'.format(numeric.columns))
    print('The categorical numeric columns are {}\n'.format(cat_num.columns))

    obj_df = pd.DataFrame(df[obj])

    unique = []

    # Function to group minor categories, if present
    def func(column):
        l=column.value_counts(normalize=True)
        minor=l[l<=0.005].index
        if len(minor) > 0:
            print('{} contains {} categories that is/are less than 0.5 percent'.format(column.name, len(minor)))
            if (column.nunique() - len(minor)) in range(1,61):
                return column.replace(minor,'others')
            else:
                unique.append(column.name)
        else:
            print('{} does not contain minor categories'.format(column.name))
            return column

    print('We found {} obj type columns!'.format(obj_df.shape[1]))
    obj_df.fillna('missing',inplace=True)
    print('Printing Cardinality info of All Object Type Columns!\n')
    print(obj_df.nunique())
    print('\n')

    # For each object type column, below are a sequence of conditions to determine if it's a unique column/not
    for col in obj_df:
        # If top 5 levels contribute to less than 10 percent of data
        if obj_df[col].value_counts(normalize=True)[:5].sum()<=0.1:
            print('{} has top 5 levels that contribute to less than 10% of data!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)

        # If number of unique entries is greater than or equal to 50000
        elif obj_df[col].nunique() >= 50000:
            print('{} has more than 50000 unique levels!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)

        # If Number of unique entries is greater than 75% of the total number of rows
        elif obj_df[col].nunique() > 0.75 * len(df):
            print('{} has more than 75% unique levels!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)
        # If none of the above is true, we try to group minor categories
        else:
            print('{} has top 5 levels that contribute to more than 10% of data!'.format(col))
            print('{} has {} levels before grouping'.format(col,obj_df[col].nunique()))
            # If number of levels is greater than 60, attempt grouping 
            if obj_df[col].nunique() > 60:
                print('Attempting grouping of minor levels of {} as the column has more than 60 levels'.format(col))
                obj_df[col] = func(obj_df[col])
                print('{} has {} levels after grouping\n'.format(col,obj_df[col].nunique()))
            else:
                print('{} is a discrete column!\n'.format(col))

    print('\nWe found {} unique columns!\n'.format(len(unique)))
    print('\nThe unique columns are {}'.format(unique))

    obj_df.drop(unique,axis=1,inplace=True)
    print('\nWe now have {} obj type discrete columns!'.format(obj_df.shape[1]))
    print('\nPrinting Cardinality info of obj Discrete Columns!\n')
    print(obj_df.nunique())
    disc = pd.concat([cat_num,obj_df],axis=1)
    if numeric.empty is False:
        numeric = ForestImputer(numeric,disc,y)
    print('\nPrinting Cardinality info of all Discrete Columns! That is categorical numerical + obj type discrete!\n')
    print(disc.nunique())
    end = time.time()
    print('\nSegregation time taken : {}'.format(end-start))
    return numeric,disc,unique

def DatasetSelection(X,Y):
  X1=X.copy()
  X2=X.copy()
  index=list(X.index)
  #Row then column
  X1.dropna(axis=0,thresh=0.5*len(X1.columns),inplace=True)#dropping the rows with many null values
  index1=list(X1.index)#storing the indices of the dataframe after the operation in index1
  X1.dropna(axis=1,thresh=0.5*len(X1),inplace=True)#dropping columns
  if len(X1.columns)==0:#in case if all columns get dropped then in result there should be no rows in the dataframe
    index1=[] #in this case list of row indices equal to null list
  Rowsdrop1=(list(set(index)-set(index1)))#storing the indices of the rows getting dropped above
  #column then row
  X2.dropna(axis=1,thresh=0.5*len(X2),inplace=True)#dropping the columns with many null values
  X2.dropna(axis=0,thresh=0.5*len(X2.columns),inplace=True)#dropping rows
  index2=list(X2.index)#storing its indices in a list
  if len(X2.columns)==0:
    index2=[]
  Rowsdrop2=(list(set(index)-set(index2)))#storing the indices of the rows getting dropped above
  if len(Rowsdrop1)<len(Rowsdrop2): #checking in which case is number of rows getting dropped is lesser
    Y.drop(Rowsdrop1,inplace=True)
    print("Columns are getting dropped first then columns")
    print("The columns getting dropped are {}".format(list(set(X.columns)-set(X1.columns))))
    print("Shape of the dataframe: {}".format(X1.shape))
    print("Shape of the target column {}".format(Y.shape))
    return X1,Y #returns resultant dataframe and target column
  else:
    Y.drop(Rowsdrop2,inplace=True)
    print("Rows are getting dropped first then rows")
    print("The columns getting dropped are {}".format(list(set(X.columns)-set(X2.columns))))
    print("Shape of the dataframe: {}".format(X2.shape))
    print("Shape of the target column {}".format(Y.shape))
    return X2,Y

def SampleEquation(X,Y,class_or_Reg,disc_df_columns,LE):
    obj_df = pd.DataFrame(X[disc_df_columns])     # collect all 'category' columns
    for col in obj_df.columns:                 # convert numeric category column type from object to numeric 
        obj_df[col]=pd.to_numeric(obj_df[col], errors = 'ignore')       
    num = obj_df._get_numeric_data().columns    
    obj = list(set(obj_df.columns)-set(num))
    obj_df=obj_df[obj]                         # only keep those category columns which are of type object(have non numeric values)
    if not obj_df.empty:
        X.drop(obj_df.columns,axis=1,inplace=True)         # drop non numerical category columns from X
        d = defaultdict(LabelEncoder)
        dummy=obj_df.copy()          # for table grid purpose
        obj_df = obj_df.apply(lambda x: d[x.name].fit_transform(x.astype(str)))  # label encode non numeric category columns
        print('LABEL ENCODED FOR SAMPLE EQUATION\n')
        X = pd.concat([X,obj_df],axis=1)       # add non numeric category columns back after encoding them
        
    from sklearn.feature_selection import f_classif, f_regression
    if class_or_Reg == 'Classification':# for classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.feature_selection import SelectKBest
        model=LogisticRegression(max_iter=400,class_weight='balanced')
        kb = SelectKBest( score_func=f_classif,k=8) #for selecting the 8 best features
        if len(X.columns)>8:#to limit the size of equation. restricting to be less than 9 variables
            kb.fit_transform(X,Y)
            new_features = []
            mask=kb.get_support() #This returns an array with true /false values with true for those columns which got selected
            for bool, feature in zip(mask,X.columns):#to extract column names from mask
                if bool:
                    new_features.append(feature)
            X=X[new_features]
           
        model.fit(X,Y)
        selected_features= X.columns
        Y_pred = model.predict(X)
        print("\nLOGISTIC REGRESSION EQUATION:\n\n")
        if Y.nunique()==2: #if there are only two classes
            for i in range(len(model.coef_)): # for dispaying the equation curresponding to all classes
                s=""
                for j in range(len(model.coef_[i])):
                    s=s+str(model.coef_[i][j])+"*"+X.columns[j]+" + "
                s=s+str(model.intercept_[i])

                lbls = LE.inverse_transform(model.classes_)       #to display class names of target instead of numbers
                print("ln(odds) = " + s)
                print("\n=> odds = exp ( "+s+" )")
                print("\nWhere, odds = P(class={}) / 1 - P(class={}) \n".format(lbls[i+1],lbls[i+1]))
                print("In simple terms Odds of an event happening is defined as the likelihood that an event will occur, expressed as a proportion of the likelihood that the event will not occur. For example - the odds of rolling four on a dice are 1/6 or 16.67%.")
                print("\nEstimated f1 score = ","{:.2%}".format(Decimal(str(f1_score(Y,Y_pred,average='weighted')))) )
                print("(F1 score is the harmonic mean of precision and recall, it tells how good the model is at predicting correctly and avoiding false predictions. Simply put, it is approximate accuracy.)")
                               
        else:#multiclass classification
            for i in range(len(model.coef_)): # for dispaying the equation curresponding to all classes
                s=""
                for j in range(len(model.coef_[i])):
                    s=s+str((model.coef_[i][j]))+"*"+X.columns[j]+" + "
                s=s+str(model.intercept_[i])
                lbls = LE.inverse_transform(model.classes_)    #to display class names of target instead of numbers
                print("Prediction of class "+ str(lbls[i])+":\n")
                print("ln(odds) = " + s)
                print("\n=> odds = exp ( "+s+" )")
                print("\nWhere, odds= P(class={}) / 1 - P(class={}) \n".format(lbls[i],lbls[i]))
            print("In simple terms Odds of an event happening is defined as the likelihood that an event will occur, expressed as a proportion of the likelihood that the event will not occur. For example - the odds of rolling four on a dice are 1/6 or 16.67%.")
            print("\nEstimated f1 score = ","{:.2%}".format(Decimal(str(f1_score(Y,Y_pred,average='weighted')))) )
            print("(F1 score is the harmonic mean of precision and recall, it tells how good the model is at predicting correctly and avoiding false predictions. Simply put, it is approximate accuracy.)")
            
    else:#regression problem
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        from sklearn.linear_model import LinearRegression
        model=LinearRegression()
        if len(X.columns)>8:#Executing forward feature selection
            sfs = SFS(model,
               k_features=8,
               forward=True,
               floating=False,
               scoring = 'r2',
               cv = 0)

            sfs.fit(X,Y)
            X=X[list(sfs.k_feature_names_)]

        model.fit(X,Y)
        selected_features= X.columns
        coeff=model.coef_
        equation=""
        for i in range(len(coeff)):
            equation= equation+str(coeff[i])+"*"+X.columns[i]+" + "
        equation=equation+str(model.intercept_)
        print("\nLINEAR REGRESSION EQUATION:\n\n")
        print('Predicted value = {}'.format(equation))
        print("\nR squared =", round(model.score(X,Y), 3))
        print("(The closer R squared is to 1, the better the model is)")
    
    dum2=pd.DataFrame()  
#     list_dfs=[]
    selected_obj_cols=list(set(selected_features)&set(obj_df.columns)) 
    if len(selected_obj_cols)!=0:  # to only print those encoded columns which are included in equation
            print("\nWhere the columns are encoded like this:\n")             
            for i in selected_obj_cols: 
                    dum=dummy.drop_duplicates(subset=[i])      
                    dum2=obj_df.drop_duplicates(subset=[i])   
                    dum2.rename(columns = {i:str(i)+" encoded"}, inplace = True)      
                    dum3=(pd.concat([dum[i],dum2[str(i)+" encoded"]],axis=1)).sort_values (str(i)+" encoded")
#                     list_dfs.append(dum3)
                    from tabulate import tabulate
                    pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql', showindex = False)
                    print(pdtabulate(dum3))

#     from IPython.display import display,HTML
#     def multi_column_df_display(list_dfs, cols=3):        #funtction to display encoded variable tables in grid form
#         html_table = "<table style='width:100%; border:0px'>{content}</table>"
#         html_row = "<tr style='border:0px'>{content}</tr>"
#         html_cell = "<td style='width:{width}%;vertical-align:top;text-align:center;border:0px'>{{content}}</td>"
#         html_cell = html_cell.format(width=100/cols)

#         cells = [ html_cell.format(content=df.to_html(index=False)) for df in list_dfs ]
#         cells += (cols - (len(list_dfs)%cols)) * [html_cell.format(content="")] # pad
#         rows = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,len(cells),cols)]
#         display(HTML(html_table.format(content="".join(rows))))
        
#     if list_dfs:    # only display table grid if any columns were encoded
#         multi_column_df_display(list_dfs)


def featureSelectionPlot(feat_df):
    f = 20
    plt.figure(figsize=(8,8))
    plt.title('Feature Importance Plot',fontsize=f)
    sns.barplot(x='scores2',y='col_name',data=feat_df,palette="YlGn_r")
    plt.xlabel('Importance',fontsize=f)
    plt.ylabel('Feature',fontsize=f)
    plt.xticks(fontsize=12,rotation=90)
    plt.yticks(fontsize=12)
    plt.show()

def FeatureSelection(X,y,class_or_Reg):
    print(X.shape)
    if class_or_Reg == 'Classification':
        print('runnning classifier selector')
        classes_num = y.nunique() #Checking Number of classes in Target
        if classes_num == 2:
            print("\nBinary Classification")
#             k = y.value_counts()
#             if k[0]>k[1]: impact_ratio = k[0]/k[1]
#             else: impact_ratio = k[1]/k[0]
#             selector = XGBClassifier(n_estimators =100, max_depth= 5, scale_pos_weight=impact_ratio, n_jobs=-1);
            selector = lgb.LGBMClassifier(class_weight='balanced',n_estimators=100,random_state=1,objective='binary')
        else:
            print("\nMulticlass Classification")

            #Creating weight array for balancing
#             class_weights = list(class_weight.compute_class_weight('balanced', np.unique(y),y))
#             class_w=pd.Series(class_weights,index=np.unique(y))
#             w_array = np.ones(y.shape[0], dtype = 'float')
#             for i,val in enumerate(y):
#               w_array[i] = class_w[val]

#             selector = XGBClassifier(n_estimators =100, sample_weight = w_array, max_depth= 5, n_jobs=-1);
            selector = lgb.LGBMClassifier(class_weight='balanced',n_estimators=100,random_state=1,objective='multiclass',num_class=classes_num,metric='multi_logloss')
    else :
#         selector = XGBRegressor(n_estimators =100, max_depth= 5, n_jobs=-1);
        selector = lgb.LGBMRegressor(n_estimators=100,random_state=1)
        print('runnning regressor selector')

    for i in tqdm(range(10)):
        selector.fit(X, y)

    # all columns container
    cols = pd.DataFrame(X.columns)

    # Getting importance scores of all the features
    k = selector.feature_importances_
    k = k.reshape(X.shape[1],1)
    k = pd.DataFrame(k)

    # threshold one(This thres is able to select only top best features which are very few)
    thresh1 = k.mean(); l = k>thresh1
    sheet1 = pd.concat([cols, k, l], axis =1)
    sheet1.columns = ['col_name','scores1','t/f']
    new_1 = sheet1.loc[(sheet1['t/f'] == False)]

    # threshold two(The mean of the remaining features is used as a thres)
    thresh2 = new_1['scores1'].mean(); l2 = k>thresh2
    print('\nthresh2: {}'.format(thresh2))
    sheet2 = pd.concat([cols, k, l2], axis =1)
    sheet2.columns = ['col_name','scores2','t/f']
    new_2 = sheet2.loc[(sheet2['t/f'] == True)]

    # Final Score Sheet
    new_2 = new_2.sort_values('scores2', ascending=False)
    print('\nThe final score sheet of {} selected columns with importances:\n' .format(new_2.shape[0]))
    print(new_2)

    rejected_cols = set(X.columns) - set(new_2.col_name)
    print('\n{} columns are eliminated during Feature Selection which are:\n{}' .format(len(rejected_cols), rejected_cols))
    return list(rejected_cols),new_2.drop(['t/f'],axis=1)

def dataHandler(dx):
        for col in dx.columns:
            if 'Unnamed' in col:
                if len(dx[col].value_counts())<0.5*dx.shape[0]:
                    dx.drop(col,axis=1,inplace=True)
        # to handel cases when some blank rows or other information above the data table gets assumed to be column name
        if (len([col for col in dx.columns if 'Unnamed' in col]) > 0.5*dx.shape[1]  ):#Checking for unnamed columns
            colNew = dx.loc[0].values.tolist()           # Getting the values in the first row of the dataframe into a list
            dx.columns = colNew                          #Making values stored in colNew as the new column names
            dx = dx.drop(labels=[0])                     #dropping the row whose values we made as the column names
            dx.reset_index(drop=True, inplace=True)      #resetting index to the normal pattern 0,1,2,3...
        else:
            return dx

        new_column_names=dx.columns.values.tolist() # Following three lines of code are for counting the number of null values in our new set of column names
        new_column_names=pd.DataFrame(new_column_names)
        null_value_sum=new_column_names.isnull().sum()[0]
        if null_value_sum<0.5*dx.shape[1]: # if count of null values are less than a certain ratio of total no of columns
            return dx
        while(null_value_sum>=0.5*dx.shape[1]):
            colNew = dx.loc[0].values.tolist()
            dx.columns = colNew
            dx = dx.drop(labels=[0])
            dx.reset_index(drop=True, inplace=True)
            new_column_names=dx.columns.values.tolist()
            new_column_names=pd.DataFrame(new_column_names)
            null_value_sum=new_column_names.isnull().sum()[0]
        return dx

def removeLowClass(df,target):
    if df[target].nunique() == 2:
        print('\nTarget has 2 Levels! No classes will be removed')
        return 1
    elif df[target].nunique() < 2:
        print('\nTarget has less than 2 Levels! Classification will not be performed')
        return None
    else:
        print('Dropping levels in target with less than 0.5%')
        vc = df[target].value_counts(normalize=True)<0.005
        classes = vc[vc==True].index.to_list()
        if df[target].nunique() - len(classes) < 2:
            print('{} levels are left. Classification will not be performed'.format(df[target].nunique() - len(classes)))
            print('Atleast 2 levels are required')
            return None
        else:
            print('\nLevels {} will be dropped!\n'.format(classes))
            print(df[target].value_counts(normalize=True))
            df[target].replace(classes,np.nan,inplace=True)
            df.dropna(subset=[target],inplace=True)
            return df

def model_training(X_train,y_train,X_test,y_test,class_or_Reg,priorList,q_s):
  # Selecting best model
  if class_or_Reg == 'Classification':
    Classification=classification()
    name,mod,acc,par,model_info = Classification.best_model_class(X_train, X_test, y_train.values, y_test.values,priorList,q_s)
  else:#Regression
    regression=Regression()
    name,mod,acc,par,model_info = regression.best_model_reg(X_train, X_test, y_train, y_test,q_s)
  print('Accuracy :',acc)
  return mod,model_info

def data_model_select(X_train,y_train):
  if len(X_train) <= 10000:
    input_X_train = X_train
    input_y_train = y_train
    print('Less than 10k rows, no sampling needed')
  elif len(X_train) > 10000 and len(X_train) <= 100000:
    input_X_train = X_train.sample(frac=0.8, random_state=1)
    input_y_train = y_train.sample(frac=0.8, random_state=1)
    print('Sampling 80% of the data')
  elif len(X_train) > 100000 and len(X_train) <= 1000000:
    input_X_train = X_train.sample(frac=0.7, random_state=1)
    input_y_train = y_train.sample(frac=0.7, random_state=1)
    print('Sampling 70% of the data')
  elif len(X_train) >1000000:
    input_X_train = X_train.sample(frac=0.5, random_state=1)
    input_y_train = y_train.sample(frac=0.5, random_state=1)
    print('Sampling 50% of the data')
  return input_X_train,input_y_train

def removeOutliers(df):
    df_z = (df - df.mean())/df.std()
    indices = df_z[(df_z>4)|(df_z<-4)].dropna(how='all').index
    print('{} rows contain outliers and will be removed'.format(len(indices)))
    return indices

def getDF(df,model):
    try:
        mdf = df[model['init_cols'].drop(model['Target'])]
        print('Columns Match!')
        return mdf
    except KeyError as e:
        print('We could not find the column/columns ' + str(e) + ' in the current file!')
        print('The column names don\'t match with the ones that were present during Training')
        print('Kindly Check for spelling, upper/lower cases and missing columns if any!')
        return None

def userInteractVisualization(df,key):
    import plotly.offline as py
    import plotly.figure_factory as ff
    import plotly.graph_objs as gobj
    py.init_notebook_mode(connected=True)

    import plotly.express as px
    from jupyter_dash import JupyterDash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import dash_table
    
    df_copy=df.copy()
    
    # if there is a primary then dropping that
    if key:
        df= df.drop(key,axis=1)
    
    # making a list of numeric and object columns
    numlist = list(df.select_dtypes(include=['int64','float64']).columns)
    for col in numlist:
        if df[col].nunique()<100:
            numlist.remove(col)
    objectlist = list(df.select_dtypes(include=['object']).columns)
    
    UNBRAND_CONFIG = dict(modeBarButtonsToRemove=['sendDataToCloud'], displaylogo=False, showLink=False)
    # Build App
    app = JupyterDash(__name__)
    
    # function to plot object columns
    def plot_objcols(col_name, bar= True, pie= False):
        values_count = pd.DataFrame(df[col_name].value_counts())
        values_count.columns = ['count']
        # convert the index column into a regular column.
        values_count[col_name] = [ str(i) for i in values_count.index ]
        # add a column with the percentage of each data point to the sum of all data points.
        values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
        # change the order of the columns.
        values_count = values_count.reindex([col_name,'count','percent'],axis=1)
        values_count.reset_index(drop=True,inplace=True)


        # add a font size for annotations0 which is relevant to the length of the data points.
    #         font_size = 20 - (.25 * len(values_count[col_name]))
        font_size = 14

        if bar == True:
            trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )

            annotations0 = [ dict(x = xi,
                                 y = yi, 
                                 showarrow=False,
                                 font={'size':font_size},
                                 text = "{:,}".format(yi),
                                 xanchor='center',
                                 yanchor='bottom' )
                           for xi,yi,_ in values_count.values ]

            annotations1 = [ dict( x = xi,
                                  y = yi/2,
                                  showarrow = False,
                                  text = "{}%".format(pi),
                                  xanchor = 'center',
                                  yanchor = 'middle',
                                  font = {'color':'yellow'})
                             for xi,yi,pi in values_count.values if pi > 10 ]

            annotations = annotations0 + annotations1                       

            layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                                 titlefont = {'size': 20},
                                 yaxis = {'title':'count'},
                                 xaxis = {'type':'category'},
                                annotations = annotations  )
            figure = gobj.Figure( data = trace0, layout = layout)

        elif pie== True:
            figure = px.pie(values_count, names=values_count[col_name], values=values_count['count'],title=col_name.replace('_',' ').capitalize())

        return figure

    # function to plot numeric columns
    def plot_numcols(col_name, hist = True, box=False, violin=False):
        series = df[col_name]
        # remove zero values items [ indicates NA values.]
        series = series[ series != 0 ]

        if hist==True:
            trace0 = gobj.Histogram( x = series,
                                    histfunc = 'avg', 
                                    histnorm = 'probability density',
                                    opacity=.75,
                                   marker = {'color':'#EB89B5'})

            layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                                titlefont = {'size':20},
                                yaxis = {'title':'Probability/Density'},
                                xaxis = {'title':col_name, 'type':'-'}
                                 )
            figure = gobj.Figure(data = trace0, layout = layout)

        if box==True:
            figure = gobj.Figure(data=gobj.Box(x=series, y0= col_name), layout = gobj.Layout(title=col_name.replace('_',' ').capitalize()))

        if violin==True:
            figure = gobj.Figure(data=gobj.Violin(x=series, box_visible=True, line_color='black',
                                   meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                                   y0=col_name),  
                                 layout = gobj.Layout(title=col_name.replace('_',' ').capitalize()))


        return figure

    # function to display table
    def get_data_table():
        data_table = dash_table.DataTable(
            id='datatable-data',
            data=df_copy.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df_copy.columns],
    #         style_table={'overflowY': 'scroll'},
            fixed_rows={'headers': True},
            style_cell={'width': '150px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
        return data_table

    app.layout = html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', 
                 children=[
            dcc.Tab(label='Data Table', value='tab-1', children=[html.Br(),
                                                                 get_data_table()]),
            dcc.Tab(label='Numeric Variables', value='tab-2', children= [
                html.Br(),
                html.Div(children=["Select a chart type:"]),
                dcc.RadioItems(id='dynamic-choice1', 
                               options=[{'label': 'Histogram', 'value': 'hist'}, 
                                        {'label': 'Box Plot', 'value': 'box'},
                                        {'label': 'Violin Plot', 'value': 'violin'}],
                               value='hist'),
                html.Div(children=[html.Br(),"Select numeric columns to visualize:"]),
                dcc.Dropdown(
                    id='dynamic-dropdown1',
                    options=[{'label': s, 'value': s} for s in numlist],
                    multi=True,
                    value="",
                    placeholder= "Select columns to visualize"
                ),
                html.Div(id='container1', children=[])]
                   ),
            dcc.Tab(label='Categorical Variables', value='tab-3', children = [
                html.Br(),
                html.Div(children=["Select a chart type:"]),
                dcc.RadioItems(id='dynamic-choice2', 
                               options=[{'label': 'Bar Chart', 'value': 'bar'}, 
                                        {'label': 'Pie Chart', 'value': 'pie'}],
                               value='bar'),
                html.Div(children=[html.Br(),"Select categorical columns to visualize:"]),
                dcc.Dropdown(
                    id='dynamic-dropdown2',
                    options=[{'label': s, 'value': s} for s in objectlist],
                    multi=True,
                    value="",
                    placeholder= "Select columns to visualize"
                ),
                html.Div(id='container2', children=[]) ]
                   )  
                 ])
        ])

    # Define callback to update numeric variable graphs
    @app.callback(
        Output('container1','children'), 
        [Input('dynamic-choice1', 'value'), 
         Input('dynamic-dropdown1', 'value')] 
    )       
    def update_graph(chart_choice, col_list):
        if chart_choice == 'hist':
            graphnum={}
            for col in col_list :
                graphnum[col] = dcc.Graph(
                        id='graph'+str(col),
                        figure= plot_numcols( col ),       
                        config=UNBRAND_CONFIG
                    )
        elif chart_choice== 'box':
            graphnum={}
            for col in col_list :
                graphnum[col] = dcc.Graph(
                        id='graph'+str(col),
                        figure= plot_numcols( col,0,1,0 ),        
                        config=UNBRAND_CONFIG
                    )
        elif chart_choice== 'violin':
            graphnum={}
            for col in col_list :
                graphnum[col] = dcc.Graph(
                        id='graph'+str(col),
                        figure= plot_numcols( col,0,0,1 ),        
                        config=UNBRAND_CONFIG
                    )
        return [graphnum[col] for col in col_list]


    # Define callback to update categorical variable graphs
    @app.callback(
        Output('container2','children'), 
        [Input('dynamic-choice2', 'value'), 
         Input('dynamic-dropdown2', 'value')] 
    )       
    def update_graph(chart_choice, col_list):
        if chart_choice == 'bar':
            graphobj={}
            for col in col_list :
                graphobj[col] = dcc.Graph(
                        id='graph1'+str(col),
                        figure= plot_objcols(col),              
                        config=UNBRAND_CONFIG
                    ) 
        elif chart_choice== 'pie':
            graphobj={}
            for col in col_list :
                graphobj[col] = dcc.Graph(
                        id='graph1'+str(col),
                        figure= plot_objcols(col,0,1),             
                        config=UNBRAND_CONFIG
                    ) 
        return [graphobj[col] for col in col_list]

    # Run app and display result inline in the notebook
    app.run_server(mode='inline')
    
        