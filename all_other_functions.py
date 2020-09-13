import time
import pandas as pd
import numpy as np
from xgboost import *
from tqdm.notebook import tqdm
from modelling import *
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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

    nu = df[num].nunique()>5
    numeric = df[nu[nu == True].index]
    cat_num = df[list(set(num) - set(numeric.columns))]
    numeric.fillna(numeric.mean(),inplace=True)
    cat_num.fillna('missing',inplace=True)

    obj_df = pd.DataFrame(df[obj])

    unique = []

    def func(column):
        l=column.value_counts(normalize=True)
        minor=l[l<=0.005].index
        if len(minor) > 0:
            print('\n{} contains {} categories that is/are less than 0.5 percent'.format(column.name, len(minor)))
            if (column.nunique() - len(minor)) in range(1,60):
                return column.replace(minor,'others')
            else:
                unique.append(column.name)
        else:
            print('\n{} does not contain minor categories'.format(column.name))
            return column

    print('We found {} obj type columns!'.format(obj_df.shape[1]))
    obj_df.fillna('missing',inplace=True)
    print('Printing Cardinality info of All Object Type Columns!\n')
    print(obj_df.nunique())
    for col in obj_df:
        obj_df[col] = func(obj_df[col])
    print('Grouped all minor levels of columns!')
    obj_df.drop(unique,axis=1,inplace=True)
    print('\nWe found {} obj type discrete columns!'.format(obj_df.shape[1]))
    print('\nPrinting Cardinality info of obj Discrete Columns!\n')
    print(obj_df.nunique())
    print('\nWe found {} unique columns!\n'.format(len(unique)))
    print('\n The useless columns are {}'.format(unique))
    end = time.time()
    print('Segregation time taken : {}'.format(end-start))
    return numeric,pd.concat([cat_num,obj_df],axis=1),unique

def Visualization(X,Y,class_or_Reg):
    import pydotplus
    if class_or_Reg == 'Classification':
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)#encoding the target variable
        Yt=pd.DataFrame(Y)
        clf = DecisionTreeClassifier(max_depth = 3, min_samples_split=2, min_samples_leaf=0.01, random_state = 1)
        clf.fit(X, Y)
        class_names=list(le.inverse_transform(sorted(Yt[Yt.columns[0]].unique())))
        for i in range(len(class_names)):
            class_names[i]=str(class_names[i])
        print("value=[n1,n2,n3...] where n1,n2,n3 are the number of samples of the classes in the order     \nvalue="+str(le.inverse_transform(sorted(Yt[Yt.columns[0]].unique()))))
        tree.plot_tree(clf,
                     feature_names =X.columns, #the list of all column names
                     class_names=class_names, #list of the class names
                     filled = True,
                     impurity=False,
                     rounded=True,
                     fontsize=10);
        dot_data = tree.export_graphviz(clf, out_file=None,
                            feature_names=X.columns,
                            filled=True, impurity=False, rounded=True,
                            special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('CART.png')
        graph.write_svg("CART.svg")

    else:
        from sklearn.tree import DecisionTreeRegressor
        from sklearn import tree
        import matplotlib.pyplot as plt
        clf = DecisionTreeRegressor(max_depth = 3, min_samples_split=2, min_samples_leaf=0.01, random_state = 0)
        clf.fit(X, Y)
        tree.plot_tree(clf,
                   feature_names =X.columns,
                      filled = True,
                     impurity=False,
                     rounded=True,
                     fontsize=10);
        dot_data = tree.export_graphviz(clf, out_file=None,
                             feature_names=X.columns,
                                filled=True, impurity=False, rounded=True,
                                special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('CART.png')
        graph.write_svg("CART.svg")

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

def SampleEquation(X,Y,class_or_Reg):
    num_df = X.select_dtypes('number')
    obj_df = X.select_dtypes('category')
    d = defaultdict(LabelEncoder)
    obj_df = obj_df.apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    print('LABEL ENCODED FOR SAMPLE EQUATION\n')
    X = pd.concat([num_df,obj_df],axis=1)
    from sklearn.feature_selection import f_classif, f_regression
    if class_or_Reg == 'Classification':# for classification
        from sklearn.linear_model import LogisticRegression
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
        if Y.nunique()==2: #if there are only two classes
            for i in range(len(model.coef_)): # for dispaying the equation curresponding to all classes
                s=""
                for j in range(len(model.coef_[i])):
                    s=s+str(model.coef_[i][j])+"*"+X.columns[j]+" + "
                s=s+str(model.intercept_[i])

                print("Power term = "+s+"\n")
                print("Probability(Y=1) = exp(Power term)/(exp(Power term) + 1)\n")
        else:#multiclass classification
            for i in range(len(model.coef_)): # for dispaying the equation curresponding to all classes
                s=""
                for j in range(len(model.coef_[i])):
                    s=s+str((model.coef_[i][j]))+"*"+X.columns[j]+" + "
                s=s+str(model.intercept_[i])

                print("Prediction of class "+ str(model.classes_[i])+"\n\n")
                print("Power term= " + s)
                print("\nPrediction(class={}) = exp(Power term)/(exp(Power term) + 1)\n".format(model.classes_[i]))
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
        coeff=model.coef_
        equation=""
        for i in range(len(coeff)):
            equation= equation+str(coeff[i])+"*"+X.columns[i]+" + "
        equation=equation+str(model.intercept_)

        print('Linear Equation is : {}'.format(equation))
    if len(obj_df.columns)!=0:
        new_df = pd.DataFrame()
        print('\nWHERE, the encoded information is as follows : \n')
        for k,v in d.items():
            new_df[str(k)] = pd.Series(v.classes_)
            new_df[str(k) + ' encoded info'] = pd.Series(d[k].transform(new_df[k].dropna()))
        print(new_df)
        print('\n')

def featureSelectionPlot(feat_df):
    f = 20
    plt.figure(figsize=(8,8))
    plt.title('Feature Importance Plot',fontsize=f)
    sns.barplot(x='scores2',y='col_name',data=feat_df,palette="Blues_d")
    plt.xlabel('Importance',fontsize=f)
    plt.ylabel('Feature',fontsize=f)
    plt.xticks(fontsize=12,rotation=90)
    plt.yticks(fontsize=12)
    plt.show()

def FeatureSelection(X,y,class_or_Reg):
    n_est = 20
    if class_or_Reg == 'Classification':
        selector = XGBClassifier(n_estimators =n_est, max_depth= 6, n_jobs=-1)
        print('runnning classifier selector')
    else :
        selector = XGBRegressor(n_estimators =n_est, max_depth= 6, n_jobs=-1)
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
    input_X_train,input_y_train=data_model_select(X_train,y_train)
    name,mod,acc,par,model_info = Classification.best_model_class(input_X_train, X_test, input_y_train.values, y_test.values,priorList)
  else:#Regression
    regression=Regression()
    input_X_train,input_y_train=data_model_select(X_train,y_train)
    name,mod,acc,par,model_info = regression.best_model_reg(input_X_train, X_test, input_y_train, y_test)
  print('Accuracy :',acc)
  return mod,model_info

def data_model_select(X_train,y_train):
  if len(X_train) <= 10000:
    input_X_train = X_train
    input_y_train = y_train
  elif len(X_train) > 10000 & len(X_train) <= 100000:
    input_X_train = X_train.sample(frac=0.8, random_state=1)
    input_y_train = y_train.sample(frac=0.8, random_state=1)
  elif len(X_train) > 100000 & len(X_train) < 1000000:
    input_X_train = X_train.sample(frac=0.7, random_state=1)
    input_y_train = y_train.sample(frac=0.7, random_state=1)
  else:
    input_X_train = X_train.sample(frac=0.5, random_state=1)
    input_y_train = y_train.sample(frac=0.5, random_state=1)
  return input_X_train,input_y_train

def removeOutliers(df):
    df_z = (df - df.mean())/df.std()
    indices = df_z[(df_z>4)|(df_z<-4)].dropna(how='all').index
    print('{} rows contain outliers and will be removed'.format(len(indices)))
    return indices

def getDF(df,model):
    try:
        df = df[model['init_cols']]
        print('Columns Match!')
        return df
    except KeyError as e:
        print('We could not find the column/columns ' + str(e) + ' in the current file!')
        print('The column names don\'t match with the ones that were present during Training')
        print('Kindly Check for spelling, upper/lower cases and missing columns if any!')
        return None

def importModel(path):
    print('IMPORTING MODEL INFORMATION')
    return joblib.load(path)
