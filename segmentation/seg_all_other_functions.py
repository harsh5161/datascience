import time
import pandas as pd
import numpy as np
from xgboost import *
from tqdm.notebook import tqdm
from seg_modelling import *
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import pydotplus
from category_encoders import TargetEncoder
from missingpy import MissForest
import prince
import operator 
def ForestImputer(num_df,disc_df):
    print("Check",disc_df)
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
    print(f"The number of columns with missing values are {df.isna().any().sum()}")
    forester = 1 #MissForest imputation will be done
    if df.isna().any().sum() > 0.50 * len(df.columns):
        forester = 0 # Mean imputation will be done
        print("Dataframe has too many columns with null values, hence mean imputation will be done")
        df.fillna(value=df.mean(),inplace=True)
        print("Printing the missing values after mean imputation",df.isna().any().sum())

    # print("Before Label Encoding")
    # print(df)
    if not disc_df.empty:
        LE = LabelEncoder()#target encoding the categorical variables
        df_new = df[cat_list].apply(LE.fit_transform)
        df1 = df.copy()
        for col in df_new.columns:
            df1[col] = df_new[col]
    else:
        df1= df.copy()

    # print("After Label Encoding")
    # print(df1)

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

def Segregation(df):
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
        if len(numeric)<50000 and len(df.columns)<100:
            print(40*'#')
            print("MissForest Imputation can be attempted")
            numeric = ForestImputer(numeric,disc)
        else:
            print(40*'#')
            print("Mean Imputation will be done")
            numeric.fillna(numeric.mean(),inplace=True)
    else:
        print("Mean Imputation will be done")
        numeric.fillna(numeric.mean(),inplace=True)
    print('\nPrinting Cardinality info of all Discrete Columns! That is categorical numerical + obj type discrete!\n')
    print(disc.nunique())
    end = time.time()
    print('\nSegregation time taken : {}'.format(end-start))
    return numeric,disc,unique

def DatasetSelection(X):
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
    print("Rows are getting dropped first then columns")
    print("The columns getting dropped are {}".format(list(set(X.columns)-set(X1.columns))))
    print("Shape of the dataframe: {}".format(X1.shape))
    return X1,Y #returns resultant dataframe and target column
  else:
    print("Columns are getting dropped first then rows")
    print("The columns getting dropped are {}".format(list(set(X.columns)-set(X2.columns))))
    print("Shape of the dataframe: {}".format(X2.shape))
    return X2


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
def drop_single_valued_features(df):
    print("Dropping columns that only contain one value")
    req_list = []
    for col in df.columns:
        if df[col].nunique() ==1 :
            print(f"Dropping {col}...")
            # df.drop(col,axis=1,inplace=True)
            req_list.append(col)
    return req_list
def calculate_n_components(df): #Dont delete the comments in this functions
    print("Calculating number of components....")
    #Generating a covariance matrix
    covMatrix = np.cov(df.T,bias=True)
    eigen_vals, eigen_vecs = np.linalg.eig(covMatrix)
    #Making a list of (eigenvalue,eigenvevtor) tuples
    # eig_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

    #Sorting the paired tuples in descending order or eigen values
    # eig_pairs.sort(key = lambda x: x[0],reverse=True)

    tot = sum(eigen_vals)
    var_exp = [(i/tot)*100 for i in sorted(eigen_vals,reverse=True)]
    var_exp_new = []
    print(type(var_exp))
    if np.iscomplex(var_exp).tolist().count(True) == len(var_exp):
        for i in range(len(var_exp)):
            var_exp_new[i] = var_exp[i].real 
        var_exp = var_exp_new
        cum_var_exp = np.cumsum(var_exp_new).tolist()   
    else:
        cum_var_exp = np.cumsum(var_exp).tolist()
    print(type(cum_var_exp))
    print(f"The variance captured by each component is \n {var_exp}")
    print(40* "-")
    print(f"The cumulative variance we capture as we travel through each components are \n {cum_var_exp}")
    try:
        for i in var_exp:
            if i <10.0:
                current =  var_exp.index(i)
                if var_exp[current] - var_exp[current+1] < 1.0:
                    n_components = current+1
                    trigger = 'Triggered'
                    retained = current
                    break
        print(f"Try block {trigger}")
    except:
        for i in cum_var_exp:
            if i>90.0:
                trigger = 'Triggered'
                n_components = cum_var_exp.index(i)+1
                retained = cum_var_exp.index(i)
                break
        print(f"Except block {trigger}")
    try:
        if cum_var_exp[retained] <60.0:  #In case the PCs do not explain at least 60% of the cumulative variance then we take PCs that explain at least that much variance
            for i in cum_var_exp:
                if i >60.0:
                    n_components = cum_var_exp.index(i)+1
                    break 
            print("Initial PCs could not explain a lot of variance so PCs that account for at least 60% of the cumulative variance is taken")
    except:
        if cum_var_exp[retained].real <60.0:  #In case the PCs do not explain at least 60% of the cumulative variance then we take PCs that explain at least that much variance
            for i in cum_var_exp:
                if i.real >60.0:
                    n_components = cum_var_exp.index(i)+1
                    break 
            print("Initial PCs could not explain a lot of variance so PCs that account for at least 60% of the cumulative variance is taken")
    return n_components
def dimensionality_reduction(df,n,DISC_VAL):
    if DISC_VAL:
        print("Mixed Variables of Numeric and Categorical found, applying FAMD Dimensionality Reduction Technique")
        try:
            FAMD = prince.FAMD(n_components= n,random_state=42,engine='sklearn')
            X = FAMD.fit_transform(df)
        except AssertionError:
            FAMD = prince.FAMD(n_components= n-1,random_state=42,engine='sklearn')
            X = FAMD.fit_transform(df)
        print("Inside FAMD")
        print(X)
        # print("inverting famd") #You invert PCA type analysis by giving the cluster centers as the input to the famd.inverse_transform function, this may not be possible 
        # print(FAMD.inverse_transform(X_FAMD))
    else:
        print("Only Numeric Variables foundm applying PCA Dimensionality Reduction Technique")
        PCA = prince.PCA(n_components= n,random_state=42,engine='sklearn')
        X = PCA.fit_transform(df)
        print("Inside PCA")
        print(X)
    return X

def profiler(segdata,req,num_df,disc_df):
        temp = segdata.copy()
        num_temp = num_df.copy()
        disc_temp = disc_df.copy()
        for column in segdata.columns:
            if column != 'Segments (Clusters)' and column not in req:
                temp.drop(column,axis=1,inplace=True)
        for col in num_temp.columns:
            if col != 'Segments (Clusters)' and col not in req:
                try:
                    num_temp.drop(col,axis=1,inplace=True)
                except:
                    pass
        for col in disc_temp.columns:
            if col != 'Segments (Clusters)' and col not in req:
                try:
                    disc_temp.drop(col,axis=1,inplace=True)
                except:
                    pass
        # print("Num",num_temp)
        # print("Disc",disc_temp)
        return temp,num_temp,disc_temp

def findDefaulters(x):
    if np.isnan(x):
        return False
    else:
        if abs(float(x)) >= 0.85:
            return True
        else:
            return False


def pearsonmaker(numeric_df,column_counter): #LowerTriangularMatrix, Dictionary with related columns, the column with the maximum value
    req_cols = []
    high = 0.85
    # corr = numeric_df.corr(method='pearson')
    corr = np.corrcoef(numeric_df.values, rowvar=False) 
    corr = pd.DataFrame(corr, columns = numeric_df.columns.to_list())
    # print("Initial correlation matrix",corr)
    corr = corr.where(np.tril(np.ones(corr.shape),k=-1).astype(np.bool))

    if column_counter is False:
        print("No columns are correlated")
        return numeric_df, {}
    else:
        maxi_col = max(column_counter.items(), key=operator.itemgetter(1))[0]


    val = column_counter[maxi_col]
    count = sum(x == val for x in column_counter.values())
    # print(f"Value of count {count}")
    if count == 1 :
        # Logic when only one column has the highest 
        drop_col = maxi_col 

    elif count > 1:
        #Logic when more than one column has equal number of highly correlated columns
        for k,v in column_counter.items():
            if v == val:
                req_cols.append(k)
        for col in corr.columns:
            # if col in req_cols:
            #     print(f"Max value of the column {col} :{corr[col].max()}")
            if col in req_cols and corr[col].max() > high:
                    high = corr[col].abs().max()
                    drop_col = col 

    # print(f"Column counter is {column_counter}")
    try:
        print(f"Dropping {drop_col} due to high correlation")
    except UnboundLocalError:
        return numeric_df, {}
    numeric_df.drop(drop_col,axis=1,inplace=True)
    del column_counter[drop_col]
    return numeric_df,column_counter

def randomSample(df):
    data = df.copy()
    return df.sample(n=50000,random_state=42)