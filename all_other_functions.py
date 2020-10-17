import time
import pandas as pd
import numpy as np
from xgboost import *
from tqdm.notebook import tqdm
from modelling import *
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import pydotplus


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

def Segregation(df):
    print('\n#### Entering Segregation ####\n')
    start = time.time()
    num = df._get_numeric_data().columns
    obj = list(set(df.columns)-set(num))

    nu = df[num].nunique()>8
    numeric = df[nu[nu == True].index]
    cat_num = df[list(set(num) - set(numeric.columns))]
    numeric.fillna(numeric.mean(),inplace=True)
    cat_num.fillna('missing',inplace=True)

    print('There are {} pure numeric columns'.format((len(numeric.columns))))
    print('There are {} categorical numeric columns\n'.format((len(cat_num.columns))))
    print('The pure numeric columns are {}'.format(numeric.columns))
    print('The categorical numeric columns are {}\n'.format(cat_num.columns))

    obj_df = pd.DataFrame(df[obj])

    unique = []

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
    for col in obj_df:
        if obj_df[col].value_counts(normalize=True)[:5].sum()<=0.1:
            print('{} has top 5 levels that contribute to less than 10% of data!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)
        elif obj_df[col].nunique() >= 50000:
            print('{} has more than 50000 unique levels!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)
        elif obj_df[col].nunique() > 0.75 * len(df):
            print('{} has more than 75% unique levels!'.format(col))
            print('{} is unique\n'.format(col))
            unique.append(col)
        else:
            print('{} has top 5 levels that contribute to more than 10% of data!'.format(col))
            print('{} has {} levels before grouping'.format(col,obj_df[col].nunique()))
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

def bivar_ploter(df1,targ,base_var):  #!! targ stores column name and base_var stores target name!!

      l=[]
      for b in set(df1[targ]):l.append((df1[df1[targ]==b].groupby(base_var).count()[targ]).rename(b))
      c=pd.concat(l,axis=1)
      if(df1[targ].nunique()>5):
          a=list(c.sum(axis=0).sort_values(ascending=False)[:4].index)
          c=pd.concat([c[a],pd.Series(c[list(set(c.columns)-set(a))].sum(axis=1),name='Others')],axis=1)
      if(df1[base_var].dtype==np.object or df1[base_var].nunique()/len(df1)>0.1):
          if(df1[base_var].nunique()<10):
                a=c.plot(kind='bar')
                plt.xlabel(base_var)
                plt.ylabel("Frequency")
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)
                plt.title(targ)
                plt.show()
          else:
            a=c.loc[list(c.sum(axis=1).sort_values().index)[-10:]].plot(kind='bar')
            plt.xlabel(base_var)
            plt.ylabel("Frequency")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)
            plt.title(targ)
            plt.show()
           
      else:
          a=c.plot(kind='line',alpha=0.5)
          plt.xlabel(base_var)
          plt.ylabel("Frequency")
          plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)
          plt.title(targ)
          plt.show()
          
      return a

def userInteractVisualization(df,targ):
        df1 =  df.copy() #df.sample(n=1000,random_state=1) if len(df)>1000
        B=list(df1.columns)
        B.remove(targ)
        l=[]
        numlist = list(df1.select_dtypes(include=['int64','float64']).columns)
        for col in numlist:
            if df1[col].nunique()<100:
                numlist.remove(col)
        objectlist = list(df1.select_dtypes(include=['object']).columns)
        start = time.time()
        if numlist:
            print("Generating Histograms for numeric columns")
            try:
                for c in numlist:
                    df1.hist( column=c, bins=15, color=np.random.rand(3,))
                    plt.show()
                
            except:
                pass

        if len(objectlist)>2:
            i=0
            print("Generating Bivariates for object columns")
            for c in B:
                    #Plots for cat features done if top 10 unique_values account for >10% of data (else visulaisation is significant)
                    if(df1[c].dtype==np.object and np.sum(df1[c].value_counts(normalize=True).iloc[:min(10,df1[c].nunique())])<0.10):continue
                    if(df1[c].dtype ==np.object):
                        try:
                            bivar_ploter(df1,c, targ);
                            i=i+1
                        except:
                            pass

        print(f'\t Done with Histogram and Bivar plotting in time {time.time() - start} seconds ')

