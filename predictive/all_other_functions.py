import time
import pandas as pd
import numpy as np
from xgboost import *
from tqdm.notebook import tqdm
from modelling import *
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.utils import class_weight
import pydotplus
from category_encoders import TargetEncoder
from missingpy import MissForest
import operator
import json
import graphviz
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
import shap
from engineerings import numeric_engineering
import gc
from sklearn import tree
from openpyxl import Workbook

def targetAnalysis(df):
    '''
    Function that returns if a particular column can undergo Classification, Regression and None
    '''
    print(">>>>>>[[Target Analysis]]>>>>>")
    
    Type = str(df.dtypes)
    # IF INT OR FLOAT IN TARGET, and IF NUMBER OF UNIQUE IS LESS, CLASSIFICATION, ELSE, REGRESSION
    
    if ('int' in Type) or ('float' in Type):
        if df.nunique() <= 5:
            return 'Classification'
        else:
            return 'Regression'

    else:
        if df.nunique() <= 5:
            return 'Classification'
        else:
            try:
                df.astype(float)
                return 'Regression'
            except:
                return None


def ForestImputer(num_df, disc_df, target):
    '''
    MissForest Imputation Function : Sometimes MissForest Imputation will not be done to speed up the application
    '''
    print(">>>>>>[[Imputation]]>>>>>")
    num_df.reset_index(drop=True, inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    concat_list = [num_df, disc_df]
    df = pd.concat(concat_list, axis=1)
    
    cat_list = disc_df.columns.to_list()
    num_list = num_df.columns.to_list()

    forester = 1  # MissForest imputation will be done
    if df.isna().any().sum() > 0.50 * len(df.columns):
        forester = 0  # Mean imputation will be done
        print("Dataframe has too many columns with null values, hence mean imputation will be done")
        df.fillna(value=df.mean(), inplace=True)
        print("Printing the missing values after mean imputation",
              df.isna().any().sum())

    # target encoding t=categorical variables
    TE = TargetEncoder(cols=cat_list)
    df1 = TE.fit_transform(df, target)

    class_or_reg = targetAnalysis(target)
    if class_or_reg == 'Classification':
        imputer = MissForest(max_iter=5, copy=True, max_depth=10,
                             class_weight="balanced", n_estimators=100)
    elif class_or_reg == 'Regression':
        imputer = MissForest(max_iter=5, copy=True,
                             max_depth=10, n_estimators=100)
    if forester == 1:
        start = time.time()
        print("MissForest imputation will be done...")
        X = imputer.fit_transform(df1)
        print("Time taken for the completion of MissForest is :", time.time()-start)
    elif forester == 0:
        X = df1
    # converting numpy array back to dataframe after transformation call
    df2 = pd.DataFrame(X, index=df1.index, columns=df1.columns)
    
    for col in df.columns:
        if col in df2.columns:
            if col not in cat_list:
                df[col] = df2[col]
    numeric = df[num_list]
    return numeric


def Segregation(df, y):
    '''
    Grouping low categories together in a column
    '''
    print("\n\n")
    print(">>>>>>[[Segregation Zone]]>>>>>")
    start = time.time()
    num = df._get_numeric_data().columns
    obj = list(set(df.columns)-set(num))

    nu = df[num].nunique() > 8
    numeric = df[nu[nu == True].index]
    cat_num = df[list(set(num) - set(numeric.columns))]
    cat_num.fillna(cat_num.median(skipna=True), inplace=True)

    print('There are {} pure numeric columns'.format((len(numeric.columns))))
    print('There are {} categorical numeric columns\n'.format(
        (len(cat_num.columns))))
    print('The pure numeric columns are {}'.format(numeric.columns))
    print('The categorical numeric columns are {}\n'.format(cat_num.columns))

    obj_df = pd.DataFrame(df[obj])

    unique = []

    # Function to group minor categories, if present
    def func(column):
        l = column.value_counts(normalize=True)
        minor = l[l <= 0.005].index
        if len(minor) > 0:
            print(
                '{} contains {} categories that is/are less than 0.5 percent'.format(column.name, len(minor)))
            if (column.nunique() - len(minor)) in range(1, 61):
                return column.replace(minor, 'others')
            else:
                unique.append(column.name)
        else:
            print('{} does not contain minor categories'.format(column.name))
            return column

    print('We found {} obj type columns!'.format(obj_df.shape[1]))
    obj_df.fillna('missing', inplace=True)
    

    print("Starting grouping of levels in categorical variables to reduce dimensionality.")
    # For each object type column, below are a sequence of conditions to determine if it's a unique column/not
    for col in obj_df:
        # If top 5 levels contribute to less than 10 percent of data
        if obj_df[col].value_counts(normalize=True)[:5].sum() <= 0.1:
            
            unique.append(col)

        # If number of unique entries is greater than or equal to 50000
        elif obj_df[col].nunique() >= 50000:
            unique.append(col)

        # If Number of unique entries is greater than 75% of the total number of rows
        elif obj_df[col].nunique() > 0.75 * len(df):
            unique.append(col)
        # If none of the above is true, we try to group minor categories
        else:
            # If number of levels is greater than 60, attempt grouping
            if obj_df[col].nunique() > 60:
                obj_df[col] = func(obj_df[col])
            else:
                pass

    print('We found {} unique columns!\n'.format(len(unique)))
    print('The unique columns are {}'.format(unique))

    obj_df.drop(unique, axis=1, inplace=True)
    print('We now have {} obj type discrete columns!'.format(
        obj_df.shape[1]))
    disc = pd.concat([cat_num, obj_df], axis=1)
    if numeric.empty is False:
        if len(numeric) < 50000 and len(df.columns) < 100:
            print("MissForest Imputation can be attempted")
            numeric = ForestImputer(numeric, disc, y)
        else:
            print("Mean Imputation will be done")
            numeric.fillna(numeric.mean(), inplace=True)
    else:
        print("Mean Imputation will be done")
        numeric.fillna(numeric.mean(), inplace=True)
    end = time.time()
    print('\nSegregation Completed: {}'.format(end-start))
    return numeric, disc, unique


def DatasetSelection(X, Y):
    '''
    We either drop rows with null values first and then columns or vice-versa depending whichever logic results in more data being retained
    '''
    print(">>>>>>[[Removing bad data]]>>>>>")
    X1 = X.copy()
    X2 = X.copy()
    index = list(X.index)
    # Row then column
    # dropping the rows with many null values
    X1.dropna(axis=0, thresh=0.5*len(X1.columns), inplace=True)
    # storing the indices of the dataframe after the operation in index1
    index1 = list(X1.index)
    X1.dropna(axis=1, thresh=0.5*len(X1), inplace=True)  # dropping columns
    if len(X1.columns) == 0:  # in case if all columns get dropped then in result there should be no rows in the dataframe
        index1 = []  # in this case list of row indices equal to null list
    # storing the indices of the rows getting dropped above
    Rowsdrop1 = (list(set(index)-set(index1)))
    # column then row
    # dropping the columns with many null values
    X2.dropna(axis=1, thresh=0.5*len(X2), inplace=True)
    X2.dropna(axis=0, thresh=0.5*len(X2.columns),
              inplace=True)  # dropping rows
    index2 = list(X2.index)  # storing its indices in a list
    if len(X2.columns) == 0:
        index2 = []
    # storing the indices of the rows getting dropped above
    Rowsdrop2 = (list(set(index)-set(index2)))
    # checking in which case is number of rows getting dropped is lesser
    if len(Rowsdrop1) < len(Rowsdrop2):
        Y.drop(Rowsdrop1, inplace=True)
        print("The columns getting dropped are {}".format(
            list(set(X.columns)-set(X1.columns))))
        print("Shape of the dataframe: {}".format(X1.shape))
        print("Shape of the target column {}".format(Y.shape))
        print("\n\n")
        return X1, Y  # returns resultant dataframe and target column
    else:
        Y.drop(Rowsdrop2, inplace=True)
        # print("Rows are getting dropped first then rows")
        print("The columns getting dropped are {}".format(
            list(set(X.columns)-set(X2.columns))))
        print("Shape of the dataframe: {}".format(X2.shape))
        print("Shape of the target column {}".format(Y.shape))
        print("\n\n")
        return X2, Y


def SampleEquation(X, Y, class_or_Reg, disc_df_columns, LE, feat,features_created):
    '''
    Sample Equation is generated by this function, Logistic Regression is used for Classification and Linear Regression is used for Regression and the outputs are saved in an xlsx file
    '''
    X = X.copy()
    obj_df = pd.DataFrame(X[disc_df_columns])
    for col in features_created:
        if col in obj_df.columns:
            obj_df.drop(col,axis=1,inplace=True)
        if col in X.columns:
            X.drop(col,axis=1,inplace=True)
    try:
        for col in obj_df.columns:                 # convert numeric category column type from object to numeric
            obj_df[col] = pd.to_numeric(obj_df[col], errors='ignore')
    except:
        pass
    num = obj_df._get_numeric_data().columns
    obj = list(set(obj_df.columns)-set(num))
    feat = list(set(feat[:])-set(num))
    # only keep those category columns which are of type object(have non numeric values)
    obj_df = obj_df[obj]
    if not obj_df.empty:
        # drop non numerical category columns from X
        X.drop(obj_df.columns, axis=1, inplace=True)
        d = defaultdict(LabelEncoder)
        dummy = obj_df.copy()          # for table grid purpose
        # label encode non numeric category columns
        obj_df = obj_df.apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        print('Encoding Data for Sample Equation\n')
        # add non numeric category columns back after encoding them
        X = pd.concat([X, obj_df], axis=1)

    from sklearn.feature_selection import f_classif, f_regression
    if class_or_Reg == 'Classification':  # for classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.feature_selection import SelectKBest
        model = LogisticRegression(max_iter=400, class_weight='balanced')
        # for selecting the 8 best features
        kb = SelectKBest(score_func=f_classif, k=8)
        if len(X.columns) > 8:  # to limit the size of equation. restricting to be less than 9 variables
            kb.fit_transform(X, Y)
            new_features = []
            # This returns an array with true /false values with true for those columns which got selected
            mask = kb.get_support()
            # to extract column names from mask
            for bool, feature in zip(mask, X.columns):
                if bool:
                    new_features.append(feature)
            X = X[new_features]

        model.fit(X, Y)
        selected_features = X.columns
        Y_pred = model.predict(X)
        print("Generating Classification Equation...")
        if Y.nunique() == 2:  # if there are only two classes
            # for dispaying the equation curresponding to all classes
            for i in range(len(model.coef_)):
                s = ""
                for j in range(len(model.coef_[i])):
                    s = s+str(round(model.coef_[i][j],6))+"*"+X.columns[j]+" + "
                s = s+str(round(model.intercept_[i],6))

                # to display class names of target instead of numbers
                lbls = LE.inverse_transform([1])
                print("ln(odds) = " + s)
                print("\n=> odds = exp ( "+s+" )")
                print(
                    "\nWhere, odds = P(class={}) / 1 - P(class={}) \n".format(lbls, lbls))
                print("In simple terms Odds of an event happening is defined as the likelihood that an event will occur, expressed as a proportion of the likelihood that the event will not occur. For example - the odds of rolling four on a dice are 1/6 or 16.67%.")
                print("\nEstimated f1 score = ", "{:.2%}".format(
                    Decimal(str(f1_score(Y, Y_pred, average='weighted')))))
                print("(F1 score is the harmonic mean of precision and recall, it tells how good the model is at predicting correctly and avoiding false predictions. Simply put, it is approximate accuracy.)")

        else:  # multiclass classification
            # for dispaying the equation curresponding to all classes
            for i in range(len(model.coef_)):
                s = ""
                for j in range(len(model.coef_[i])):
                    s = s+str(round(model.coef_[i][j],6))+"*"+X.columns[j]+" + "
                s = s+str(round(model.intercept_[i],6))
                # to display class names of target instead of numbers
                lbls = LE.inverse_transform(model.classes_)
                print("Prediction of class " + str(lbls[i])+":\n")
                print("ln(odds) = " + s)
                print("\n=> odds = exp ( "+s+" )")
                print(
                    "\nWhere, odds= P(class={}) / 1 - P(class={}) \n".format(lbls[i], lbls[i]))
            print("In simple terms Odds of an event happening is defined as the likelihood that an event will occur, expressed as a proportion of the likelihood that the event will not occur. For example - the odds of rolling four on a dice are 1/6 or 16.67%.")
            print("\nEstimated f1 score = ", "{:.2%}".format(
                Decimal(str(f1_score(Y, Y_pred, average='weighted')))))
            print("(F1 score is the harmonic mean of precision and recall, it tells how good the model is at predicting correctly and avoiding false predictions. Simply put, it is approximate accuracy.)")

    else:  # regression problem
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        if len(X.columns) > 8:  # Executing forward feature selection
            sfs = SFS(model,
                      k_features=8,
                      forward=True,
                      floating=False,
                      scoring='r2',
                      cv=0)

            sfs.fit(X, Y)
            X = X[list(sfs.k_feature_names_)]

        model.fit(X, Y)
        selected_features = X.columns
        coeff = model.coef_
        equation = ""
        for i in range(len(coeff)):
            equation = equation+str(round(coeff[i],6))+"*"+X.columns[i]+" + "
        equation = equation+str(round(model.intercept_,6))
        print("Generating Regression Equation...")
        print('Predicted value = {}'.format(equation))
        print("\nR squared =", round(model.score(X, Y), 3))
        print("(The closer R squared is to 1, the better the model is)")

    dum2 = pd.DataFrame()
    selected_obj_cols = list(set(selected_features) & set(obj_df.columns))
    wb = Workbook()
    ws=wb.active
    if len(selected_obj_cols) != 0:  # to only print those encoded columns which are included in equation
        with pd.ExcelWriter('sample_equation_encodings.xlsx') as writer:
            writer.book= wb
            writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)
            for i in selected_obj_cols:
                dum = dummy.drop_duplicates(subset=[i])
                dum2 = obj_df.drop_duplicates(subset=[i])
                dum2.rename(columns={i: str(i)+" encoded"}, inplace=True)
                dum3 = (pd.concat([dum[i], dum2[str(i)+" encoded"]],
                        axis=1)).sort_values(str(i)+" encoded")
                dum3.reset_index(drop=True,inplace=True)
                from tabulate import tabulate
                def pdtabulate(df): return tabulate(
                    df, headers='keys', tablefmt='psql', showindex=False)
                dum3.to_excel(writer, sheet_name = f'{i}')
                writer.save()
                try:
                    std=wb.get_sheet_by_name('Sheet')
                    wb.remove_sheet(std)
                except:
                    pass
                # Json variable to show the tables in a new format in the front end
                json_var = dum3.to_json()
    return list(set(feat))


def featureSelectionPlot(feat_df):
    '''
    this function doesn't matter because the output of this function is only visible in the DS code, but isnt actually being used by the WebApp
    '''
    f = 20
    plt.figure(figsize=(8, 8))
    plt.title('Feature Importance Plot', fontsize=f)
    sns.barplot(x='scores2', y='col_name', data=feat_df, palette="YlGn_r")
    plt.xlabel('Importance', fontsize=f)
    plt.ylabel('Feature', fontsize=f)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.show()
    plt.close('all')


def FeatureSelection(X, y, class_or_Reg):
    '''
    Feature Selection is done using LGBM 
    '''
    print(X.shape)
    if class_or_Reg == 'Classification':

        classes_num = y.nunique()  # Checking Number of classes in Target
        if classes_num == 2:
            print("\nBinary Classification Selector Used")
            selector = lgb.LGBMClassifier(class_weight='balanced', max_depth=16,
                                          num_leaves=30, n_estimators=100, random_state=42, objective='binary')
        else:
            print("\nMulticlass Classification Selector Used")

            selector = lgb.LGBMClassifier(class_weight='balanced', max_depth=16, num_leaves=30, n_estimators=100,
                                          random_state=42, objective='multiclass', num_class=classes_num, metric='multi_logloss')

        print("Classifer Selector Done...")
    else:
        selector = lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.01,
                                     n_estimators=1000, random_state=42, subsample=0.8, num_leaves=30, max_depth=16)

        print("Regression Selector Done...")
    for i in tqdm(range(10)):
        selector.fit(X, y)
    # all columns container
    cols = pd.DataFrame(X.columns)

    # Getting importance scores of all the features
    k = selector.feature_importances_
    k = k.reshape(X.shape[1], 1)
    k = pd.DataFrame(k)
    gc.collect()
    # threshold one(This thres is able to select only top best features which are very few)
    thresh1 = k.mean()
    l = k > thresh1
    sheet1 = pd.concat([cols, k, l], axis=1)
    sheet1.columns = ['col_name', 'scores1', 't/f']
    new_1 = sheet1.loc[(sheet1['t/f'] == False)]

    # threshold two(The mean of the remaining features is used as a thres)
    thresh2 = new_1['scores1'].mean()
    l2 = k > thresh2
    sheet2 = pd.concat([cols, k, l2], axis=1)
    sheet2.columns = ['col_name', 'scores2', 't/f']
    new_2 = sheet2.loc[(sheet2['t/f'] == True)]

    # Final Score Sheet
    new_2 = new_2.sort_values('scores2', ascending=False)
    print('\nThe final score sheet of {} selected columns with importances:\n' .format(
        new_2.shape[0]))
    print(new_2)

    rejected_cols = set(X.columns) - set(new_2.col_name)
    print('\n{} columns are eliminated during Feature Selection which are:\n{}' .format(
        len(rejected_cols), rejected_cols))
    return list(rejected_cols), new_2.drop(['t/f'], axis=1),selector,X.columns.tolist()


def removeLowClass(df, target):
    '''
    Additional logic to remove categories from a multiclass target that constitute a very small part of the dataset
    '''
    if df[target].nunique() == 2:
        print('\nTarget has 2 Levels! No classes will be removed')
        return 1
    elif df[target].nunique() < 2:
        print('\nTarget has less than 2 Levels! Classification will not be performed')
        return None
    else:
        print('Dropping levels in target with less than 0.5%')
        vc = df[target].value_counts(normalize=True) < 0.006
        classes = vc[vc == True].index.to_list()
        if df[target].nunique() - len(classes) < 2:
            print('{} levels are left. Classification will not be performed'.format(
                df[target].nunique() - len(classes)))
            print('Atleast 2 levels are required')
            return None
        else:
            print('\nLevels {} will be dropped!\n'.format(classes))
            print(df[target].value_counts(normalize=True))
            df[target].replace(classes, np.nan, inplace=True)
            df.dropna(subset=[target], inplace=True)
            return df


def model_training(X_train, y_train, X_test, y_test, class_or_Reg, priorList, q_s):
    '''
    Modelling Process helper function, no idea why he added this here so if you are refactoring - move this inside the modelling.py
    '''
    # Selecting best model
    if class_or_Reg == 'Classification':
        Classification = classification()
        name, mod, acc, par, model_info, exp_mod, exp_name, feat_mod, feat_name = Classification.best_model_class(
            X_train, X_test, y_train.values, y_test.values, priorList, q_s)
    else:  # Regression
        regression = Regression()
        name, mod, acc, par, model_info, exp_mod, exp_name, feat_mod, feat_name = regression.best_model_reg(
            X_train, X_test, y_train, y_test, q_s)
    print('Accuracy :', acc)
    return mod, model_info, exp_mod, exp_name, feat_mod, feat_name


def data_model_select(X_train, y_train):
    '''
    Deprecated
    '''
    if len(X_train) <= 10000:
        input_X_train = X_train
        input_y_train = y_train
        print('Less than 10k rows, no sampling needed')
    elif len(X_train) > 10000 and len(X_train) <= 100000:
        input_X_train = X_train.sample(frac=0.8, random_state=42)
        input_y_train = y_train.sample(frac=0.8, random_state=42)
        print('Sampling 80% of the data')
    elif len(X_train) > 100000 and len(X_train) <= 1000000:
        input_X_train = X_train.sample(frac=0.7, random_state=42)
        input_y_train = y_train.sample(frac=0.7, random_state=42)
        print('Sampling 70% of the data')
    elif len(X_train) > 1000000:
        input_X_train = X_train.sample(frac=0.5, random_state=42)
        input_y_train = y_train.sample(frac=0.5, random_state=42)
        print('Sampling 50% of the data')
    return input_X_train, input_y_train


def removeOutliers(df):
    '''
    Deprecated
    '''
    df_z = (df - df.mean())/df.std()
    indices = df_z[(df_z > 4) | (df_z < -4)].dropna(how='all').index
    print('{} rows contain outliers and will be removed'.format(len(indices)))
    return indices


def getDF(df, model):
    try:
        mdf = df[model['init_cols'].drop(model['Target'])]
        print('Columns Match!')
        return mdf
    except KeyError as e:
        print('We could not find the column/columns ' +
              str(e) + ' in the current file!')
        print(
            'The column names don\'t match with the ones that were present during Training')
        print('Kindly Check for spelling, upper/lower cases and missing columns if any!')
        return None


# !! targ stores column name and base_var stores target name!!
def bivar_ploter(df1, targ, base_var):
    '''
    Only actually used as a placeholder. Webapp generates these plots in plotly separately, don't remove because of legacy code in the sever.
    '''

    l = []
    for b in set(df1[targ]):
        l.append((df1[df1[targ] == b].groupby(
            base_var).count()[targ]).rename(b))
    c = pd.concat(l, axis=1)
    if(df1[targ].nunique() > 5):
        a = list(c.sum(axis=0).sort_values(ascending=False)[:4].index)
        c = pd.concat([c[a], pd.Series(
            c[list(set(c.columns)-set(a))].sum(axis=1), name='Others')], axis=1)
    if(df1[base_var].dtype == np.object or df1[base_var].nunique()/len(df1) > 0.1):
        if(df1[base_var].nunique() < 10):
            a = c.plot(kind='bar')
            plt.xlabel(base_var)
            plt.ylabel("Frequency")
            plt.legend(loc='center left', bbox_to_anchor=(
                1, 0.7), fancybox=True, shadow=True)
            plt.title(targ)
            plt.show()
            plt.close()
        else:
            a = c.loc[list(c.sum(axis=1).sort_values().index)
                      [-10:]].plot(kind='bar')
            plt.xlabel(base_var)
            plt.ylabel("Frequency")
            plt.legend(loc='center left', bbox_to_anchor=(
                1, 0.7), fancybox=True, shadow=True)
            plt.title(targ)
            plt.show()
            plt.close()
    else:
        a = c.plot(kind='line', alpha=0.5)
        plt.xlabel(base_var)
        plt.ylabel("Frequency")
        plt.legend(loc='center left', bbox_to_anchor=(
            1, 0.7), fancybox=True, shadow=True)
        plt.title(targ)
        plt.show()
        plt.close()
    return a


def userInteractVisualization(df1, targ):
    B = list(df1.columns)
    B.remove(targ)
    l = []
    numlist = list(df1.select_dtypes(include=['int64', 'float64']).columns)
    for col in numlist:
        if df1[col].nunique() < 100:
            numlist.remove(col)
    objectlist = list(df1.select_dtypes(include=['object']).columns)
    start = time.time()
    df1[numlist] = df1[numlist].fillna(df1.mode().iloc[0])
    for col in numlist:
        df1[col] = df1[col].clip(lower=df1[col].quantile(
            0.1), upper=df1[col].quantile(0.9))

    if numlist:
        print("Generating Histograms for numeric columns")
        try:
            for c in numlist:
                df1.hist(column=c, bins=15, color=np.random.rand(3,))
                plt.show()
                plt.close()
        except:
            pass

    if len(objectlist) > 2:
        i = 0
        print("Generating Bivariates for object columns")
        for c in B:
            # Plots for cat features done if top 10 unique_values account for >10% of data (else visulaisation is significant)
            if(df1[c].dtype == np.object and np.sum(df1[c].value_counts(normalize=True).iloc[:min(10, df1[c].nunique())]) < 0.10):
                continue
            if(df1[c].dtype == np.object):
                try:
                    bivar_ploter(df1, c, targ)
                    i = i+1
                except:
                    pass

    print(
        f'\t Done with Histogram and Bivar plotting in time {time.time() - start} seconds ')


def findDefaulters(x):
    if np.isnan(x):
        return False
    else:
        if abs(float(x)) >= 0.85:
            return True
        else:
            return False


# LowerTriangularMatrix, Dictionary with related columns, the column with the maximum value
def pearsonmaker(numeric_df, column_counter):
    '''
    Actual logic of pearson correlation, where the threshold is 0.85 and we are recursively recalculating the correlation matrix until we remove collinearity
    '''
    req_cols = []
    high = 0.85
    corr = np.corrcoef(numeric_df.values, rowvar=False)
    corr = pd.DataFrame(corr, columns=numeric_df.columns.to_list())
    corr = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(np.bool))

    if column_counter is False:
        return numeric_df, {}
    else:
        maxi_col = max(column_counter.items(), key=operator.itemgetter(1))[0]

    val = column_counter[maxi_col]
    count = sum(x == val for x in column_counter.values())
    if count == 1:
        # Logic when only one column has the highest
        drop_col = maxi_col

    elif count > 1:
        # Logic when more than one column has equal number of highly correlated columns
        for k, v in column_counter.items():
            if v == val:
                req_cols.append(k)
        for col in corr.columns:
            if col in req_cols and corr[col].max() > high:
                high = corr[col].abs().max()
                drop_col = col

    try:
        print(f"Dropping {drop_col} due to high correlation")
    except UnboundLocalError:
        return numeric_df, {}
    numeric_df.drop(drop_col, axis=1, inplace=True)
    del column_counter[drop_col]
    return numeric_df, column_counter


def format_y_labels(x, stored_labels):
    if x in stored_labels:
        return x
    else:
        return np.nan


def rules_tree(X, y, mode, X_transformed, LE,features_created):
    '''
    Decision Tree Classifier or Regressor is used to generate human-readable rules from the tree
    '''
    X_inside = X.copy()
    X_transformed_inside = X_transformed.copy()
    for col in features_created:
     if col in X_inside.columns:
        X_inside.drop(col,axis=1,inplace=True)
     if col in X_transformed_inside:
        X_transformed_inside.drop(col,axis=1,inplace=True)
    # print('Trying to generate a rule tree...')
    if mode == 'Classification':
        text_selector = DecisionTreeClassifier(class_weight='balanced', max_depth=4, min_samples_split=int(
            0.05*len(X_transformed_inside)), ccp_alpha=0.001,random_state=42)
        y.fillna(y.mode()[0], inplace=True)
        y = y.astype('int')
        y = LE.inverse_transform(y)
    else:
        text_selector = DecisionTreeRegressor(
            max_depth=4, min_samples_split=int(0.05*len(X_transformed_inside)), ccp_alpha=0.001,random_state=42)
    for i in tqdm(range(10)):
        text_selector.fit(X_transformed_inside, y)

    text_rules = export_text(
        text_selector, feature_names=X_inside.columns.to_list(), show_weights=True, decimals=2)
    joblib.dump(text_rules, 'text_rule.txt')
    return text_rules, text_selector


def featureimportance(exp_mod, exp_name, num_features, features):
    '''
    This is the feature importance plot that actually gets generated for the webapp
    '''
    print(">>>>>>[[Feature Importance Plot]]>>>>>")
    r = random.random()
    b = random.random()
    g = random.random()
    colors = (r, g, b)
    importances = exp_mod.feature_importances_
    indices = np.argsort(importances)
    feature_importances = (exp_mod.feature_importances_ /
                           sum(exp_mod.feature_importances_))*100
    print("printing feature importances",len(feature_importances))
    results = pd.DataFrame(
        {'Features': features, 'Importances': feature_importances})
    results.sort_values(by='Importances', inplace=True)
    results = results.iloc[len(results)-10:,:] if len(results) > 10 else results.copy()
    plt.figure(figsize=(10, 10))
    plt.title(
        f'Feature Importances for {exp_name} Explainable Model based on Test Data')
    # only plot the customized number of features
    fmt = '%.0f%%'
    plt.barh(results['Features'], results['Importances'],
             color=colors, align='center')
    plt.yticks(range(num_features), [features[i]
               for i in indices[-num_features:]])
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))
    plt.xlabel('Relative Importance')
    plt.show()
    plt.close('all')
    return [features[i] for i in indices[-num_features:]]


def ruleTesting(X_test, y_test, mode, model, LE,features_created):
    '''
    Performance of the Decision Tree used in RuleTree is generated here, actually verify if this has been migrated to the webapp
    '''
    X_test = X_test.copy()
    y_test = y_test.copy()
    for col in features_created:
        if col in X_test:
            X_test.drop(col,axis=1,inplace=True)
    if mode == 'Classification':
        result = pd.DataFrame(index=range(1), columns=[
                              'Machine Learning Model', 'Accuracy%', 'Precision', 'Recall', 'Weighted F1'])
        predictions = model.predict(X_test)
        try:
            predictions = [int(i) for i in predictions[:]]
        except:
            predictions = [int(i)
                           for i in LE.transform(predictions[:]).tolist()]
        result['Machine Learning Model'] = 'Decision Tree Classifier'
        result['Accuracy%'] = "{:.2%}".format(
            Decimal(str(accuracy_score(y_test, predictions))))
        result['Precision'] = precision_score(
            y_test, predictions, average='weighted')
        result['Recall'] = recall_score(
            y_test, predictions, average='weighted')
        result['Weighted F1'] = f1_score(
            y_test, predictions, average='weighted')
    else:
        result = pd.DataFrame(index=range(1), columns=[
                              'Machine Learning Model', 'RMSE', 'MSE', 'MAE'])
        predictions = model.predict(X_test)
        result['Machine Learning Model'] = 'Decision Tree Regressor'
        result['RMSE'] = sqrt(mean_squared_error(y_test, predictions))
        result['MSE'] = mean_squared_error(y_test, predictions)
        result['MAE'] = mean_absolute_error(y_test, predictions)

    return result


def inputCap(df, target):
    '''
    A feature added a long time ago to limit the user data to 1 million rows, think about removing this some time in the future when the project is better rounded
    '''
    dfsamp = df.sample(n=1000, random_state=42) if len(df) > 1000 else df.copy()
    dfsamp = numeric_engineering(dfsamp)
    dfsamp = dfsamp.dropna(axis=0, subset=[target])
    class_or_Reg = targetAnalysis(dfsamp[target])
    if class_or_Reg == 'Classification':
        if len(df) > 1000000:
            df_train, _ = train_test_split(
                df, train_size=1000000, random_state=42, stratify=df[target])
            return df_train
        else:
            return df
    elif class_or_Reg == 'Regression':
        dfr = df.sample(n=1000000, random_state=42) if len(
            df) > 1000000 else df.copy()
        return dfr
    elif class_or_Reg is None:
        return pd.DataFrame()
