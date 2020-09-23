        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

        ###################################### C L A S S I F I C A T I O N #####################################

        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

import pandas as pd
import os
import numpy as np
import random
from pprint import pprint
from itertools import combinations
import ast # ast.literal_eval(str(best))
from time import process_time
import time
from decimal import Decimal
import math

# Model
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

#Hyperopt
import hyperopt
from hyperopt import *
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

#sklearn library
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import classification_report
from sklearn.utils import compute_sample_weight
from sklearn.utils import class_weight

import xgboost as xgb
from scipy.stats import ks_2samp

class classification:

  #This funciton takes input of training and testing datasets and give out the best model's Name, model with best parameter(can be used directly to score data using 'predcit' function), accuracy on the test dataset and parameters (not usefful)
  ###############################################################################################################################
  def best_model_class(self,X_train ,X_test, y_train, y_test,priorList,q_s,MAX_EVALS=15,CV=5):
      df=pd.DataFrame()
      print("Q_S value passed is!!!!",q_s)
      if q_s ==True:print('QUICK RESULTS')#QUICK RESULTS
      else:print('HYPER OP')
      class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(y_train),
                                             y_train))
      class_w = pd.Series(class_weights,index=np.unique(y_train))
      w_array = np.ones(y_train.shape[0], dtype = 'float')
      for i,val in enumerate(y_train):
        w_array[i] = class_w[val]

      maxval = priorList.max()
      print(maxval)
      minval = priorList.min()
      print(minval)
      myval = math.ceil(maxval/minval)
      print(myval)

      print("PRIOR LIST IS",priorList)
      flag = 1
      check = 1
      if len(priorList) == 2:
        check =1  #binary classification problem
        for val in priorList:
          if val <= 0.25:
            flag = 0
            check =1  #binary classification problem
      elif len(priorList) >2:
        check =0 #multiclassification problem
        for val in priorList:
          if val <= 0.15:
            flag = 0

      if q_s ==True:  #QUICK RESULTS
        ind=0
        best = {}
        #XGBoost
        #######################################################################
        df.loc[ind,'Name']='XGBoost'
        if check == 1:
            df.loc[ind,'model']=xgb.XGBClassifier(n_estimators=100,eta= 0.1,max_depth=16,min_child_weight=2,gamma=5,subsample=0.1,scale_pos_weight=1,eval_metric='logloss')
        elif check ==0:
            df.loc[ind,'model']=xgb.XGBClassifier(n_estimators=100,eta= 0.1,max_depth=16,min_child_weight=2,gamma=5,subsample=0.1,objective="multi:softmax",scale_pos_weight=1,eval_metric='mlogloss',num_class=len(priorList))

        df.loc[ind,'param']=str(best)
        Start=time.time()
        eval_set = [(X_test, y_test)]
        if check ==1:
            df.loc[ind,'model'].fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set,verbose=False)
        elif check==0:
            df.loc[ind,'model'].fit(X_train, y_train,sample_weight = w_array, eval_metric="mlogloss", eval_set=eval_set,verbose=False)

        xgb_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, xgb_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, xgb_pred))))
        if check==1:    # if binary classification
          df.loc[ind, 'Precision']=precision_score(y_test, xgb_pred)
          df.loc[ind, 'Recall']=recall_score(y_test, xgb_pred)
          df.loc[ind, 'F1']=f1_score(y_test, xgb_pred)
          df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, xgb_pred)
            
        elif check==0:   # if multiclass classification
          df.loc[ind, 'Precision']=precision_score(y_test, xgb_pred,average='weighted')
          df.loc[ind, 'Recall']=recall_score(y_test, xgb_pred,average='weighted')
          df.loc[ind, 'F1']=f1_score(y_test, xgb_pred,average='weighted')
          #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, xgb_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, xgb_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, xgb_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, xgb_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("XGB val done")
        ind=ind+1
        ########################################################################################################

        ##Catboost
        ########################################################################################################
        df.loc[ind,'Name']='CatBoost'
        if check==1:
            df.loc[ind,'model']=cb.CatBoostClassifier(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,auto_class_weights="Balanced")
        elif check==0:
            df.loc[ind,'model']=cb.CatBoostClassifier(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,auto_class_weights="Balanced",loss_function='MultiClass')
        df.loc[ind,'param']=str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train,eval_set=eval_set,verbose=False)
        catboost_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, catboost_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, catboost_pred))))
        if check==1:
          df.loc[ind, 'Precision']=precision_score(y_test, catboost_pred)
          df.loc[ind, 'Recall']=recall_score(y_test, catboost_pred)
          df.loc[ind, 'F1']=f1_score(y_test, catboost_pred)
          df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, catboost_pred)
        elif check==0:
          df.loc[ind, 'Precision']=precision_score(y_test, catboost_pred,average='weighted')
          df.loc[ind, 'Recall']=recall_score(y_test, catboost_pred,average='weighted')
          df.loc[ind, 'F1']=f1_score(y_test, catboost_pred,average='weighted')
          #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_testbin, catboost_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, catboost_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, catboost_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, catboost_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("CAT val done")
        ind=ind+1
        ########################################################################################################


        ##LGBM
        ########################################################################################################
        df.loc[ind,'Name']='Light GBM'
        if check==1:
            df['model'][ind]=lgb.LGBMClassifier(boosting_type='gbdt',class_weight='balanced',learning_rate=0.1,n_estimators=100,random_state=1,subsample=1.0,num_leaves=31,max_depth=16,objective='binary')
        elif check==0:
            df['model'][ind]=lgb.LGBMClassifier(boosting_type='gbdt',class_weight='balanced',learning_rate=0.1,n_estimators=100,random_state=1,subsample=1.0,num_leaves=31,max_depth=16,objective='multiclass',num_class=len(priorList),metric='multi_logloss')
        df.loc[ind,'param']= str(best)
        Start=time.time()
        if check==1:
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
        elif check==0:
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="multi_logloss", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
        lightgbm_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, lightgbm_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, lightgbm_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, lightgbm_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, lightgbm_pred)
            df.loc[ind, 'F1']=f1_score(y_test, lightgbm_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, lightgbm_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, lightgbm_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, lightgbm_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, lightgbm_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, lightgbm_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, lightgbm_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, lightgbm_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, lightgbm_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("LGBM val done")
        ind=ind+1



        ##Random forest
        ########################################################################################################
        df.loc[ind,'Name']='Random Forest'
        df['model'][ind]=RandomForestClassifier(n_estimators=100,max_depth=16,class_weight='balanced')
        df.loc[ind,'param']= str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        randomforest_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, randomforest_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, randomforest_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, randomforest_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, randomforest_pred)
            df.loc[ind, 'F1']=f1_score(y_test, randomforest_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, randomforest_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, randomforest_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, randomforest_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, randomforest_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, randomforest_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, randomforest_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, randomforest_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, randomforest_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("RF val done")
        ind=ind+1

        ########################################################################################################


        ##ExtraTreesClassifier(2) Finding out accuracy on the test dataset
        ########################################################################################################
        df.loc[ind,'Name']='Extra Trees Classifier'
        df['model'][ind]=ExtraTreesClassifier(n_estimators=100,max_depth=16,class_weight='balanced')
        df.loc[ind,'param']=str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        extra_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, extra_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, extra_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, extra_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, extra_pred)
            df.loc[ind, 'F1']=f1_score(y_test, extra_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, extra_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, extra_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, extra_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, extra_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, extra_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, extra_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, extra_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, extra_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("ET val done")
        ind=ind+1
        #########################################################################################################

        #NaiveBayes
        ########################################################################################################

        if(flag == 1):
            best = {'priors': priorList}
            df.loc[ind,'Name']='Naive Bayes(Bayesisan Statistics)'
            df.loc[ind,'model']=GaussianNB(priors = priorList)
            df.loc[ind,'param']=str(best)
            Start=time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            naive_pred = df.loc[ind,'model'].predict(X_test)
            End=time.time()
            df.loc[ind,'accuracy']=accuracy_score(y_test, naive_pred)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, naive_pred))))
            if check==1:
                df.loc[ind, 'Precision']=precision_score(y_test, naive_pred)
                df.loc[ind, 'Recall']=recall_score(y_test, naive_pred)
                df.loc[ind, 'F1']=f1_score(y_test, naive_pred)
                df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, naive_pred)
            elif check==0:
                df.loc[ind, 'Precision']=precision_score(y_test, naive_pred,average='weighted')
                df.loc[ind, 'Recall']=recall_score(y_test, naive_pred,average='weighted')
                df.loc[ind, 'F1']=f1_score(y_test, naive_pred,average='weighted')
                #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, naive_pred,average='weighted',multi_class='ovo')
            df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, naive_pred)
            df.loc[ind, 'MCC']=matthews_corrcoef(y_test, naive_pred)
            #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, naive_pred)
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("Naive Bayes done")
            ind=ind+1


        #Logistic Regression
        ##########################################################################################################

        df.loc[ind,'Name']='Logistic Regression'
        df.loc[ind,'model']=LogisticRegression(class_weight='balanced',solver='saga',penalty='l2',random_state=1,max_iter=1000,multi_class ='auto')
        df.loc[ind,'param']=""
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        log_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, log_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, log_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, log_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, log_pred)
            df.loc[ind, 'F1']=f1_score(y_test, log_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, log_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, log_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, log_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, log_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, log_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, log_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, log_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, log_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("LR val done")
        ind=ind+1




        #Neural net
        ########################################################################################################

        if(flag == 1):
            df.loc[ind,'Name']='Neural Network'
            best={'hidden_layer_sizes':(50,),'solver':'sgd','learning_rate':'adaptive','max_iter':1000,'early_stopping':True}
            df.loc[ind,'model']=MLPClassifier(**best)
            df.loc[ind,'param']=str(best)
            Start=time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            neural_pred = df.loc[ind,'model'].predict(X_test)
            End=time.time()
            df.loc[ind,'accuracy']=accuracy_score(y_test, neural_pred)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, neural_pred))))
            if check==1:
                df.loc[ind, 'Precision']=precision_score(y_test, neural_pred)
                df.loc[ind, 'Recall']=recall_score(y_test, neural_pred)
                df.loc[ind, 'F1']=f1_score(y_test, neural_pred)
                df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, neural_pred)
            elif check==0:
                df.loc[ind, 'Precision']=precision_score(y_test, neural_pred,average='weighted')
                df.loc[ind, 'Recall']=recall_score(y_test, neural_pred,average='weighted')
                df.loc[ind, 'F1']=f1_score(y_test, neural_pred,average='weighted')
                #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, neural_pred,average='weighted',multi_class='ovo')
            df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, neural_pred)
            df.loc[ind, 'MCC']=matthews_corrcoef(y_test, neural_pred)
            #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, neural_pred)
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("NN done")
            ind=ind+1


        #SVC
        #########################################################################################################

        df.loc[ind,'Name']='Support Vector Machine'
        df.loc[ind,'model']= svm.SVC(kernel='linear',max_iter=1000,class_weight='balanced',probability=True,random_state=1)
        df.loc[ind,'param']= str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        support_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, support_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, support_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, support_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, support_pred)
            df.loc[ind, 'F1']=f1_score(y_test, support_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, support_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, support_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, support_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, support_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, support_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, support_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, support_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, support_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("SVC val done ")
        ind=ind+1


      elif q_s == False:
        ind=0
        #XGBoost
        #########################################################################################################################
        ##XGBoost(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
        ########################################################################################################
        def objective(params):
              print(params)
              xg = xgb.XGBClassifier(**params)
              result=cross_val_score(xg,X=X_train,y=y_train,cv=CV,scoring='accuracy',error_score=np.nan)
              print("XGB train done")
              print(result.min()*100)
              return (1-result.min())
        Start=time.time()
        sample_weight = compute_sample_weight('balanced', y_train)
        Space = {
              'n_estimators': 100, #scope.int(hp.quniform('n_estimators', 50,500,50)),
              'eta': hp.uniform('eta', 0.01,0.2 ),
              'max_depth': 16, #scope.int(hp.quniform('max_depth',2,16,1 )),
              'min_child_weight':  scope.int(hp.quniform('min_child_weight',1,15,1 )),
              'colsample_bytree': hp.uniform('colsample_bytree', 0.2,1.0 ),
              'gamma': scope.int(hp.quniform('gamma', 0,15,1)),
              'subsample': hp.uniform('subsample',  0.2,1.0  ),
              # 'sample_weight':sample_weight
              }
        if check ==1:
            Space['eval_metric'] = 'logloss'
            if myval >2:
                Space['scale_pos_weight'] = hp.choice('scale_pos_weight',[1,myval-1,myval,myval+1])
            else:
                Space['scale_pos_weight'] = hp.choice('scale_pos_weight',[1,myval])
        elif check==0:
            Space['eval_metric'] = 'mlogloss'
            Space['objective'] = 'multi:softmax'
            Space['num_class'] = len(priorList)


        bayes_trials = Trials()
        print("Moving into HyperOp")
        best = fmin(fn=objective, space = Space, algo = hyperopt.tpe.suggest,max_evals=MAX_EVALS, trials = bayes_trials)
        print("HyperOP done for XGB")

        best['n_estimators']=100 #int(best['n_estimators'])
        best['max_depth']=20 #int(best['max_depth'])
        best['min_child_weight']=int(best['min_child_weight'])
        best['gamma'] = int(best['gamma'])
        if check==1:
             best['eval_metric']='logloss'
        elif check==0:
             best['eval_metric']='mlogloss'
             best['objective'] = 'multi:softmax'

        best['subsample'] = float(best['subsample'])
        if check ==1:
            if myval >2:
                wea = [1,myval-1,myval,myval+1]
                best['scale_pos_weight'] = wea[best['scale_pos_weight']]
            else:
                wea = [1,myval]
                best['scale_pos_weight'] = wea[best['scale_pos_weight']]
        # best['sample_weight']=sample_weight
        print('XGB done')
        ########################################################################################################


        ##XGBoost(2) Finding out accuracy on the test dataset
        ########################################################################################################
        df.loc[ind,'Name']='XGBoost'
        df.loc[ind,'model']=xgb.XGBClassifier(**best)
        df.loc[ind,'param']=str(best)
        eval_set = [(X_test, y_test)]
        if check ==1:
            df.loc[ind,'model'].fit(X_train, y_train, eval_metric="logloss",early_stopping_rounds=30, eval_set=eval_set,verbose=False)
        elif check==0:
            df.loc[ind,'model'].fit(X_train, y_train,sample_weight = w_array,early_stopping_rounds=30, eval_metric="mlogloss", eval_set=eval_set,verbose=False)

        xgb_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, xgb_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, xgb_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, xgb_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, xgb_pred)
            df.loc[ind, 'F1']=f1_score(y_test, xgb_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, xgb_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, xgb_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, xgb_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, xgb_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, xgb_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, xgb_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, xgb_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, xgb_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("XGB val done")
        ind=ind+1
        ########################################################################################################

        #Catboost
        #########################################################################################################################
        df.loc[ind,'Name']='CatBoost'
        if check==1:
            df.loc[ind,'model']=cb.CatBoostClassifier(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,auto_class_weights="Balanced")
        elif check==0:
            df.loc[ind,'model']=cb.CatBoostClassifier(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,auto_class_weights="Balanced",loss_function='MultiClass')

        df.loc[ind,'param']=str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train,eval_set=eval_set,verbose=False)
        catboost_pred = df.loc[ind,'model'].predict(X_test).tolist()
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, catboost_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, catboost_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, catboost_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, catboost_pred)
            df.loc[ind, 'F1']=f1_score(y_test, catboost_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, catboost_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, catboost_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, catboost_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, catboost_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, catboost_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, catboost_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, catboost_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, catboost_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("CAT val done")
        ind=ind+1
        ########################################################################################################


        #LightGBM(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
        ########################################################################################################


        def objective(params):
            print('\n',params)
            lb = lgb.LGBMClassifier(**params)
            result = cross_val_score(lb,X=X_train,y=y_train,cv=CV,scoring='accuracy',error_score=np.nan)
            print("LGBM train done")
            print("\n",result.min()*100)
            return (1-result.min())

        Start=time.time()
        Space = {
                'boosting_type': 'gbdt',
                'learning_rate': hp.uniform('learning_rate',0.01,0.2),
                'class_weight': 'balanced',
                'n_estimators': 100, #scope.int(hp.quniform('n_estimators',50,1250,75)),
                'random_state':1,
                'subsample': hp.uniform('subsample',  0.1,1.0  ),
                'num_leaves': scope.int(hp.quniform('num_leaves',29,43,1)),
                'max_depth': 16, # scope.int(hp.quniform('max_depth',2,16,1 )),
                'min_child_weight':  scope.int(hp.quniform('min_child_weight',1,16,1 ))
              }

        if check==1:
            Space['objective'] = 'binary'
        elif check==0:
            Space['objective'] = 'multiclass'
            Space['num_class'] = len(priorList)
            Space['metric'] = 'multi_logloss'

        bayes_trials = Trials()
        print("Moving into HyperOp")
        best = fmin(fn=objective, space = Space, algo = hyperopt.tpe.suggest,max_evals=MAX_EVALS, trials = bayes_trials)
        print("HyperOP done for LGBM")

        best['boosting_type'] = 'gbdt'
        best['learning_rate'] = float(best['learning_rate'])
        best['class_weight'] = 'balanced'
        best['n_estimators'] = 100 #int(best['n_estimators'])
        best['random_state'] = 1
        best['subsample'] = float(best['subsample'])
        best['num_leaves'] = int(best['num_leaves'])
        best['min_child_weight']=int(best['min_child_weight'])
        best['max_depth'] = 16 #int(best['max_depth'])
        if check==1:
            best['objective'] = 'binary'
        elif check==0:
            best['objective'] = 'multiclass'
            best['num_class'] = len(priorList)
            best['metric'] = 'multi_logloss'

        print("LGBM done")

        ########################################################################################################
        ##LGBM(2) Finding out accuracy on the test dataset
        ########################################################################################################
        eval_set = [(X_test, y_test)]
        df.loc[ind,'Name']='Light GBM'
        df['model'][ind]=lgb.LGBMClassifier(**best)
        df.loc[ind,'param']= str(best)
        if check==1:
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
        elif check==0:
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="multi_logloss", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
        lightgbm_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, lightgbm_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, lightgbm_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, lightgbm_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, lightgbm_pred)
            df.loc[ind, 'F1']=f1_score(y_test, lightgbm_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, lightgbm_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, lightgbm_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, lightgbm_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, lightgbm_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, lightgbm_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, lightgbm_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, lightgbm_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, lightgbm_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("LGBM val done")
        ind=ind+1



        #Random Forest
        ########################################################################################################
        ##Random Forest(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
        ########################################################################################################
        def objective(params):
            print(params)
            rf = RandomForestClassifier(**params)
            result=cross_val_score(rf,X=X_train,y=y_train,cv=CV,scoring='accuracy',error_score=np.nan)
            print("RF train done")
            print(result.min()*100)
            return (1-result.min())
        Start=time.time()
        DSpace = {
                  'n_estimators': 100, # scope.int(hp.quniform('n_estimators', 100, 1200,50)),
                  "max_depth": 16, # scope.int(hp.quniform('max_depth',2,20,1)),
                  'max_features': hp.choice('max_features',['auto', 'sqrt','log2']),
                  'min_samples_split': scope.int(hp.quniform('min_samples_split',2,15,1)),
                  'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1,20,1)),
                  'oob_score':False,
                  'bootstrap':  hp.choice('bootstrap',[True, False]),
                  'class_weight':'balanced'
                  }

        bayes_trials = Trials()
        print("Moving into HyperOp")
        try:
            best = fmin(fn = objective, space = DSpace, algo = hyperopt.tpe.suggest,max_evals = MAX_EVALS, trials = bayes_trials)
            print("HyperOP done for RF")
        except:
            print("Hyperparameter tuning failed")
            best.clear()
            best['class_weight']='balanced'
        else:
            best['n_estimators']=100 #int(best['n_estimators'])
            best['max_depth']= 16 #int(best['max_depth'])
            best['min_samples_split']=int(best['min_samples_split'])
            best['min_samples_leaf']=int(best['min_samples_leaf'])
            fea=['auto', 'sqrt','log2']
            best['max_features']=fea[best['max_features']]
            best['oob_score']= False
            boot=[True, False]
            best['bootstrap']=boot[best['bootstrap']]
            best['class_weight']='balanced'
            print("HyperOP done for RF")


        print("RF done")
        ########################################################################################################



        ##Random forest(2) Finding out accuracy on the test dataset
        ########################################################################################################
        et_dict = best.copy()
        df.loc[ind,'Name']='Random Forest'
        df['model'][ind]=RandomForestClassifier(**best)
        df.loc[ind,'param']= str(best)
        df.loc[ind,'model'].fit(X_train, y_train)
        randomforest_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, randomforest_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, randomforest_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, randomforest_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, randomforest_pred)
            df.loc[ind, 'F1']=f1_score(y_test, randomforest_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, randomforest_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, randomforest_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, randomforest_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, randomforest_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, randomforest_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, randomforest_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, randomforest_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, randomforest_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("RF val done")
        ind=ind+1

        ########################################################################################################

        #ExtraTreesClassifier
        ########################################################################################################
        df.loc[ind,'Name']='Extra Trees Classifier'
        df['model'][ind]=ExtraTreesClassifier(**et_dict)
        df.loc[ind,'param']=str(et_dict)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        extra_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, extra_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, extra_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, extra_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, extra_pred)
            df.loc[ind, 'F1']=f1_score(y_test, extra_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, extra_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, extra_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, extra_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, extra_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, extra_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, extra_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, extra_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, extra_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("ET val done")
        ind=ind+1
        #########################################################################################################

        #NaiveBayes
        ########################################################################################################

        if(flag == 1):
            best = {'priors': priorList}
            df.loc[ind,'Name']='Naive Bayes(Bayesian Statistics)'
            df.loc[ind,'model']=GaussianNB(priors = priorList)
            df.loc[ind,'param']=str(best)
            Start=time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            naive_pred = df.loc[ind,'model'].predict(X_test)
            End=time.time()
            df.loc[ind,'accuracy']=accuracy_score(y_test, naive_pred)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, naive_pred))))
            if check==1:
                df.loc[ind, 'Precision']=precision_score(y_test, naive_pred)
                df.loc[ind, 'Recall']=recall_score(y_test, naive_pred)
                df.loc[ind, 'F1']=f1_score(y_test, naive_pred)
                df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, naive_pred)
            elif check==0:
                df.loc[ind, 'Precision']=precision_score(y_test, naive_pred,average='weighted')
                df.loc[ind, 'Recall']=recall_score(y_test, naive_pred,average='weighted')
                df.loc[ind, 'F1']=f1_score(y_test, naive_pred,average='weighted')
                #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, naive_pred,average='weighted',multi_class='ovo')
            df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, naive_pred)
            df.loc[ind, 'MCC']=matthews_corrcoef(y_test, naive_pred)
            #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, naive_pred)
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("Naive Bayes done")
            ind=ind+1


        #Logistic regression
        ########################################################################################################
        df.loc[ind,'Name']='Logistic Regression'
        df.loc[ind,'model']=LogisticRegression(class_weight='balanced',solver='saga',penalty='l2',random_state=1,max_iter=1000,multi_class ='auto')
        df.loc[ind,'param']=""
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        log_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, log_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, log_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, log_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, log_pred)
            df.loc[ind, 'F1']=f1_score(y_test, log_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, log_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, log_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, log_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, log_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, log_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, log_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, log_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, log_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("LR val done")
        ind=ind+1


        #Neural net
        ########################################################################################################

        if(flag == 1):
                df.loc[ind,'Name']='Neural Network'
                best={'hidden_layer_sizes':(50,),'solver':'sgd','learning_rate':'adaptive','max_iter':1000,'early_stopping':True}
                df.loc[ind,'model']=MLPClassifier(**best)
                df.loc[ind,'param']=str(best)
                Start=time.time()
                df.loc[ind,'model'].fit(X_train, y_train)
                neural_pred = df.loc[ind,'model'].predict(X_test)
                End=time.time()
                df.loc[ind,'accuracy']=accuracy_score(y_test, neural_pred)*100
                df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, neural_pred))))
                if check==1:
                    df.loc[ind, 'Precision']=precision_score(y_test, neural_pred)
                    df.loc[ind, 'Recall']=recall_score(y_test, neural_pred)
                    df.loc[ind, 'F1']=f1_score(y_test, neural_pred)
                    df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, neural_pred)
                elif check==0:
                    df.loc[ind, 'Precision']=precision_score(y_test, neural_pred,average='weighted')
                    df.loc[ind, 'Recall']=recall_score(y_test, neural_pred,average='weighted')
                    df.loc[ind, 'F1']=f1_score(y_test, neural_pred,average='weighted')
                    #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, neural_pred,average='weighted',multi_class='ovo')
                df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, neural_pred)
                df.loc[ind, 'MCC']=matthews_corrcoef(y_test, neural_pred)
                #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, neural_pred)
                df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
                print("NN done")
                ind=ind+1

        #Support Vector Machine(linear)
        ########################################################################################################

        df.loc[ind,'Name']='Support Vector Machine'
        df.loc[ind,'model']= svm.SVC(kernel='linear',max_iter=1000,class_weight='balanced',probability=True,random_state=1)
        df.loc[ind,'param']= str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        support_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=accuracy_score(y_test, support_pred)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, support_pred))))
        if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, support_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, support_pred)
            df.loc[ind, 'F1']=f1_score(y_test, support_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, support_pred)
        elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, support_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, support_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, support_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, support_pred,average='weighted',multi_class='ovo')
        df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, support_pred)
        df.loc[ind, 'MCC']=matthews_corrcoef(y_test, support_pred)
        #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, support_pred)
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("SVC val done ")
        ind=ind+1



      #Ensemble
      ########################################################################################################
      ##Ensemble(1) Finding all possible combination of above model and find out the best combination based on testing data accuracy
      ########################################################################################################
      lev=len(np.unique(y_test))
      arr1=np.empty((len(y_test),lev,0))
      for i in range(0,len(df)):
          arr1=np.dstack((arr1,df.loc[i,'model'].predict_proba(X_test)))

      max_f1=0
      max_seq=0
      for i in range(2,len(df)+1):
          comb=list(combinations(enumerate(np.rollaxis(arr1,axis=2,start=0)), i))
          for j in range(0,len(comb)):
              m=np.empty((len(y_test),lev,0))
              for x in range(0,len(comb[j])):
                  m=np.dstack((m,comb[j][x][1]))
              arr=np.mean(m,axis=2)
              clas=np.argmax(arr,axis=1)
              f1=f1_score(y_test, clas,average='weighted')*100
              seq=np.array(comb[j])[:,0]
              if f1>max_f1:
                  max_f1=f1
                  max_seq=seq

      print("this is what you are printing",max_seq)
      ########################################################################################################

      ##Ensemble(2) List of the best combination from the above method
      ########################################################################################################
      name=''
      df_en=pd.DataFrame(index = range(1000), columns=['Name','model'])
      for i in range(0,len(max_seq)):
          df_en.at[i,'Name']= df.at[max_seq[i],'Name']
          val = df.at[max_seq[i],'model']
          df_en['model'][i] = val
          name=name+df['Name'][max_seq[i]]+'+'

      df_en.dropna(axis=0,inplace=True)
      ########################################################################################################


      ##Ensemble(3) Making an esemble model of the best combination
      ########################################################################################################
      df.loc[ind,'Name']=('Ensemble '+'(' + name[:-1] + ')')
      df.loc[ind,'model']=VotingClassifier(df_en.values, voting='soft',n_jobs=-1)
      df.loc[ind,'param']="Default"
      Start=time.time()
      df.loc[ind,'model'].fit(X_train, y_train)
      ensemble_pred = df.loc[ind,'model'].predict(X_test)
      End=time.time()
      df.loc[ind,'accuracy']=accuracy_score(y_test, ensemble_pred)*100
      df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(accuracy_score(y_test, ensemble_pred))))
      if check==1:
            df.loc[ind, 'Precision']=precision_score(y_test, ensemble_pred)
            df.loc[ind, 'Recall']=recall_score(y_test, ensemble_pred)
            df.loc[ind, 'F1']=f1_score(y_test, ensemble_pred)
            df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, ensemble_pred)
      elif check==0:
            df.loc[ind, 'Precision']=precision_score(y_test, ensemble_pred,average='weighted')
            df.loc[ind, 'Recall']=recall_score(y_test, ensemble_pred,average='weighted')
            df.loc[ind, 'F1']=f1_score(y_test, ensemble_pred,average='weighted')
            #df.loc[ind, 'ROC_AUC_score']=roc_auc_score(y_test, ensemble_pred,average='weighted',multi_class='ovo')
      df.loc[ind, 'Kappa']=cohen_kappa_score(y_test, ensemble_pred)
      df.loc[ind, 'MCC']=matthews_corrcoef(y_test, ensemble_pred)
      #df.loc[ind, 'KS_statistic'],df.loc[ind, 'KS_p-value']=ks_2samp(y_test, ensemble_pred)
      df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
      ind=ind+1

      ##Best Model
      ########################################################################################################
      best_info=df.sort_values('F1',ignore_index=True,ascending=False).loc[0,:]
      best_name=best_info['Name']
      best_mod=best_info['model']
      best_acc=best_info['accuracy']
      best_param=best_info['param']
      ########################################################################################################

      return best_name,best_mod, best_acc, best_param,df



      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################

      ###################################### R E G R E S S I O N #############################################

      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################
      ########################################################################################################


import pandas as pd
import os
import numpy as np
import random
from pprint import pprint
from itertools import combinations
import ast # ast.literal_eval(str(best))
from time import process_time
import time
from math import sqrt
from decimal import Decimal

# Model
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

#Hyperopt
import hyperopt
from hyperopt import *
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample



#sklearn library
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import classification_report

from RegscorePy import aic,bic

  ###############################################################################################################################
class Regression:

  #This funciton takes input of training and testing datasets and give out the best model's Name, model with best parameter(can be used directly to score data using 'predcit' function), accuracy on the test dataset and parameters (not usefful)
  ###############################################################################################################################
  def best_model_reg(self,X_train , X_test, y_train, y_test,q_s,MAX_EVALS=15,CV=5):
      df=pd.DataFrame()
      print("The value of Q_S is ",q_s)
      if q_s ==True:print('QUICK RESULTS')#QUICK RESULTS
      else:print('HYPER OP')
      if q_s ==True:  #QUICK RESULTS
        ind=0
        best = {}
        #XGBoost
        #######################################################################
        df.loc[ind,'Name']='XGBoost'
        df.loc[ind,'model']=xgb.XGBRegressor(n_estimators=100,eta=0.1,max_depth=16,min_child_weight=2,gamma=5,subsample=0.1,objective="reg:squarederror",eval_metric='rmse')
        df.loc[ind,'param']=str(best)
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        xgb_reg_prob1 = df.loc[ind,'model'].predict(X_test).tolist()
        print(type(xgb_reg_prob1))
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, xgb_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, xgb_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, xgb_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, xgb_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, xgb_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, xgb_reg_prob1,X_train.shape[1])
        #print("aic done")
        df.loc[ind,'BIC']=bic.bic(y_test, xgb_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

        print("XGB Validation done")
        ind=ind+1
        ########################################################################################################

        ##Catboost
        ########################################################################################################
        df.loc[ind,'Name']='CatBoost'
        df.loc[ind,'model']=cb.CatBoostRegressor(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,silent=True)
        df.loc[ind,'param']=str(best)
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        cat_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, cat_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, cat_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, cat_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, cat_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, cat_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, cat_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, cat_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("CAT Validation done")
        ind=ind+1
        ########################################################################################################


        ##LGBM
        ########################################################################################################
        eval_set = [(X_test, y_test)]
        df.loc[ind,'Name']='Light GBM'
        df['model'][ind]=lgb.LGBMRegressor(boosting_type='gbdt',learning_rate=0.1,n_estimators=100,random_state=1,subsample=1.0,num_leaves=31,max_depth=16)
        df.loc[ind,'param']= str(best)
        Start=time.time()
        df.loc[ind,'model'].fit(X_train, y_train,verbose=False)
        lightgbm_pred = df.loc[ind,'model'].predict(X_test)
        End=time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, cat_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, cat_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, cat_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, cat_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, cat_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, cat_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, cat_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("LGBM val done")
        ind=ind+1
        ########################################################################################################


        ##Random forest
        ########################################################################################################
        df.loc[ind,'Name']='Random Forest'
        df['model'][ind]=RandomForestRegressor(n_estimators=100,max_depth=20)
        df.loc[ind,'param']=str(best)
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        random_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, random_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, random_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, random_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, random_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, random_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, random_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, random_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("RF Validation done")
        ind=ind+1

        ########################################################################################################


        ##ExtraTreesClassifier(2) Finding out accuracy on the test dataset
        ########################################################################################################
        df.loc[ind,'Name']='ExtraTrees Regressor'
        df['model'][ind]=ExtraTreesRegressor(n_estimators=100,max_depth=20)
        df.loc[ind,'param']=str(best)
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        extra_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, extra_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, extra_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, extra_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, extra_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, extra_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, extra_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, extra_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("ET Validation done")
        ind=ind+1
        #########################################################################################################


        #Linear Regression
        ##########################################################################################################

        df.loc[ind,'Name']='Linear Regression'
        df.loc[ind,'model']=LinearRegression()
        df.loc[ind,'param']=None
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        logr_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, logr_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, logr_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, logr_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, logr_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, logr_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, logr_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, logr_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

        print("linear reg done")
        ind=ind+1

        #Ridge Regression
        ##########################################################################################################

        df.loc[ind,'Name']='Ridge Regression'
        df.loc[ind,'model']=Ridge()
        df.loc[ind,'param']=None
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        ridge_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, ridge_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, ridge_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, ridge_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, ridge_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, ridge_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, ridge_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, ridge_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
        print("ridge reg done")
        ind=ind+1

        #Neural net
        ########################################################################################################

        best={'hidden_layer_sizes':(50,),'solver':'sgd','learning_rate':'adaptive','max_iter':1000,'early_stopping':True,'n_iter_no_change':30}
        df.loc[ind,'Name']='Neural Network'
        df.loc[ind,'model']=MLPRegressor(**best)
        df.loc[ind,'param']=str(best)
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        mlpc_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        try:
            df.loc[ind,'accuracy']=r2_score(y_test, mlpc_reg_prob1)*100
        except:
            print("Neural Net threw an error")
        else:
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, mlpc_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, mlpc_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, mlpc_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, mlpc_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, mlpc_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, mlpc_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

            print("neural net done")
            ind=ind+1

        #SVC
        #########################################################################################################

        df.loc[ind,'Name']='Support Vector Machine'
        df.loc[ind,'model']=svm.SVR(kernel='linear',max_iter=1000)
        df.loc[ind,'param']=None
        Start = time.time()
        df.loc[ind,'model'].fit(X_train, y_train)
        svc_reg_prob1 = df.loc[ind,'model'].predict(X_test)
        End = time.time()
        df.loc[ind,'accuracy']=r2_score(y_test, svc_reg_prob1)*100
        df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, svc_reg_prob1))))
        df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, svc_reg_prob1))
        df.loc[ind,'MSE'] = mean_squared_error(y_test, svc_reg_prob1)
        df.loc[ind,'MAE']=mean_absolute_error(y_test, svc_reg_prob1)
        #df.loc[ind,'AIC']=aic.aic(y_test, svc_reg_prob1,X_train.shape[1])
        df.loc[ind,'BIC']=bic.bic(y_test, svc_reg_prob1,X_train.shape[1])
        df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

        print("SVC done")
        ind=ind+1

      elif q_s==False:
            ind = 0
            #XGBoost
            #########################################################################################################################
            ##XGBoost(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
            ########################################################################################################
            def objective(params):
                  print(params)
                  xg = xgb.XGBRegressor(**params)
                  result=cross_val_score(xg,X=X_train,y=y_train,cv=CV,scoring='r2',error_score=np.nan)
                  print("XGB Training Done")
                  return (1-result.min())

            Start=time.time()
            Space = {
                  'n_estimators': 100, # scope.int(hp.quniform('n_estimators', 50,1250,75)),
                  'eta': hp.uniform('eta', 0.01,0.2 ),
                  'max_depth': 20, # scope.int(hp.quniform('max_depth',2,16,1 )),
                  'min_child_weight':  scope.int(hp.quniform('min_child_weight',1,15,1 )),
                  'colsample_bytree': hp.uniform('colsample_bytree', 0.2,1.0 ),
                  'gamma': scope.int(hp.quniform('gamma', 0,15,1)),
                  'eval_metric': 'rmse',
                  'objective': 'reg:linear',
                  'subsample': hp.uniform('subsample',  0.2,1.0  )
              }


            bayes_trials = Trials()
            best = fmin(fn = objective, space = Space, algo = hyperopt.tpe.suggest,max_evals=MAX_EVALS, trials = bayes_trials)
            print("XGB hyperop done")


            best['n_estimators']=100 #int(best['n_estimators'])
            best['max_depth']=20 #int(best['max_depth'])
            best['gamma'] = int(best['gamma'])
            best['subsample']= float(best['subsample'])
            best['min_child_weight']=int(best['min_child_weight'])
            best['objective']='reg:squarederror'
            best['eval_metric']='rmse'
            ########################################################################################################


            ##XGBoost(2) Finding out accuracy on the test dataset
            ########################################################################################################
            eval_set = [(X_test, y_test)]
            df.loc[ind,'Name']='XGBoost'
            df.loc[ind,'model']=xgb.XGBRegressor(**best)
            df.loc[ind,'param']=str(best)
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="rmse", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
            xgb_reg_prob1 = df.loc[ind,'model'].predict(X_test).tolist()
            print(type(xgb_reg_prob1))
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, xgb_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, xgb_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, xgb_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, xgb_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, xgb_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, xgb_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, xgb_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

            print("XGB Validation done")
            ind=ind+1
            ########################################################################################################


            #Catboost
            #########################################################################################################################
            ##Catboost
            ########################################################################################################
            df.loc[ind,'Name']='CatBoost'
            df.loc[ind,'model']=cb.CatBoostRegressor(depth=10,iterations=1000,learning_rate=0.1,rsm=1.0,silent=True)
            df.loc[ind,'param']=str(best)
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            cat_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, cat_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, cat_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, cat_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, cat_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, cat_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, cat_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, cat_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("CAT Validation done")
            ind=ind+1
            ########################################################################################################


            #LightGBM
            ########################################################################################################
            ##LightGBM(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
            ########################################################################################################

            def objective(params):
                  print(params)
                  xg = lgb.LGBMRegressor(**params)
                  result=cross_val_score(xg,X=X_train,y=y_train,cv=CV,scoring='r2',error_score=np.nan)
                  print("XGB Training Done")
                  return (1-result.min())

            Start=time.time()
            Space = {
                'boosting_type': 'gbdt',
                'learning_rate': hp.uniform('learning_rate',0.01,0.2),
                'n_estimators': 100, # scope.int(hp.quniform('n_estimators',50,1250,75)),
                'random_state':1,
                'subsample': hp.uniform('subsample',  0.1,1.0  ),
                'num_leaves': scope.int(hp.quniform('num_leaves',29,43,1)),
                'max_depth': 16, #scope.int(hp.quniform('max_depth',2,16,1 )),
                'min_child_weight':  scope.int(hp.quniform('min_child_weight',1,16,1 ))
              }

            bayes_trials = Trials()
            print("Moving into HyperOp")
            best = fmin(fn=objective, space = Space, algo = hyperopt.tpe.suggest,max_evals=MAX_EVALS, trials = bayes_trials)
            print("HyperOP done for LGBM")

            best['boosting_type'] = 'gbdt'
            best['learning_rate'] = float(best['learning_rate'])
            best['n_estimators'] = 100 #int(best['n_estimators'])
            best['random_state'] = 1
            best['subsample'] = float(best['subsample'])
            best['num_leaves'] = int(best['num_leaves'])
            best['min_child_weight']=int(best['min_child_weight'])
            best['max_depth'] = 16 #int(best['max_depth'])

            print("LGBM done")

            ##LightGBM(2) Finding out accuracy on the test dataset
            ########################################################################################################
            eval_set = [(X_test, y_test)]
            df.loc[ind,'Name']='Light GBM'
            df['model'][ind]=lgb.LGBMRegressor(**best)
            df.loc[ind,'param']= str(best)
            df.loc[ind,'model'].fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set,early_stopping_rounds=30,verbose=False)
            lightgbm_pred = df.loc[ind,'model'].predict(X_test)
            End=time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, cat_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, cat_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, cat_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, cat_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, cat_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, cat_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, cat_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("LGBM val done")
            ind=ind+1


            #Random forest
            #########################################################################################################################
            ##Random forest(1) Finding Best hyperparamter using Bayesian Hyperparameter Optimization
            ########################################################################################################
            def objective(params):
                  print(params)
                  rf = RandomForestRegressor(**params)
                  result=cross_val_score(rf,X=X_train,y=y_train,cv=CV,scoring='r2',error_score=np.nan)
                  print("Random Forest Training done")
                  return (1-result.min())
            Start=time.time()
            Space = {
                      'n_estimators': 100, #scope.int(hp.quniform('n_estimators', 100,1200,50)),
                      "max_depth": 20, # scope.int(hp.quniform('max_depth',2,30,1)),
                      'max_features': hp.choice('max_features',['auto', 'sqrt','log2']),
                      'min_samples_split': scope.int(hp.quniform('min_samples_split',2,15,1)),
                      'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1,20,1)),
                      'oob_score':False,
                      'bootstrap':  hp.choice('bootstrap',[True, False])
                  }

            bayes_trials = Trials()
            try:
                best = fmin(fn = objective, space = Space, algo = hyperopt.tpe.suggest,max_evals = MAX_EVALS, trials = bayes_trials)
                print("HyperOP done for RF")
            except:
                print("Hyperparameter tuning failed")
                best.clear()
                best['oob_score']=False
            else:
                best['n_estimators']=100 #int(best['n_estimators'])
                best['max_depth']= 20 #int(best['max_depth'])
                best['min_samples_split']=int(best['min_samples_split'])
                best['min_samples_leaf']=int(best['min_samples_leaf'])
                fea=['auto', 'sqrt','log2']
                best['max_features']=fea[best['max_features']]
                best['oob_score']= False
                boot=[True, False]
                best['bootstrap']=boot[best['bootstrap']]
                print("RF Hyperop done")

            print("RF done")
            ########################################################################################################


            ##Random forest(2) Finding out accuracy on the test dataset
            ########################################################################################################
            et_dict = best.copy()
            df.loc[ind,'Name']='Random Forest'
            df['model'][ind]=RandomForestRegressor(**best)
            df.loc[ind,'param']=str(best)
            df.loc[ind,'model'].fit(X_train, y_train)
            random_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, random_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, random_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, random_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, random_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, random_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, random_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, random_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("RF Validation done")
            ind=ind+1

            ########################################################################################################


            #ExtraTrees Regression
            #########################################################################################################################
            df.loc[ind,'Name']='ExtraTrees Regressor'
            df['model'][ind]=ExtraTreesRegressor(**et_dict)
            df.loc[ind,'param']=str(best)
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            extra_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, extra_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, extra_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, extra_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, extra_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, extra_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, extra_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, extra_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("ETextra_reg_prob1 Validation done")
            ind=ind+1

            ########################################################################################################


            #Ridge Regression
            ########################################################################################################
            df.loc[ind,'Name']='Ridge Regression'
            df.loc[ind,'model']=Ridge()
            df.loc[ind,'param']=None
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            ridge_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, ridge_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, ridge_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, ridge_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, ridge_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, ridge_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, ridge_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, ridge_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
            print("ridge reg done")
            ind=ind+1

            #Linear regression
            ########################################################################################################
            df.loc[ind,'Name']='Linear Regression'
            df.loc[ind,'model']=LinearRegression()
            df.loc[ind,'param']=None
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            logr_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, logr_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, logr_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, logr_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, logr_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, logr_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, logr_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, logr_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

            print("linear reg done")
            ind=ind+1

            #Neural net
            ########################################################################################################
            best={'hidden_layer_sizes':(50,),'solver':'sgd','learning_rate':'adaptive','max_iter':1000,'early_stopping':True,'n_iter_no_change':30}
            df.loc[ind,'Name']='Neural Net'
            df.loc[ind,'model']=MLPRegressor(**best)
            df.loc[ind,'param']=str(best)
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            mlpc_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            try:
                df.loc[ind,'accuracy']=r2_score(y_test, mlpc_reg_prob1)*100
            except:
                print("Neural Net threw an error")
            else:
                df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, mlpc_reg_prob1))))
                df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, mlpc_reg_prob1))
                df.loc[ind,'MSE'] = mean_squared_error(y_test, mlpc_reg_prob1)
                df.loc[ind,'MAE']=mean_absolute_error(y_test, mlpc_reg_prob1)
                #df.loc[ind,'AIC']=aic.aic(y_test, mlpc_reg_prob1,X_train.shape[1])
                df.loc[ind,'BIC']=bic.bic(y_test, mlpc_reg_prob1,X_train.shape[1])
                df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

                print("neural net done")
                ind=ind+1


            #Support Vector Machine
            ########################################################################################################
            df.loc[ind,'Name']='Support Vector Machine'
            df.loc[ind,'model']=svm.SVR(kernel='linear',max_iter=1000)
            df.loc[ind,'param']=None
            Start = time.time()
            df.loc[ind,'model'].fit(X_train, y_train)
            svc_reg_prob1 = df.loc[ind,'model'].predict(X_test)
            End = time.time()
            df.loc[ind,'accuracy']=r2_score(y_test, svc_reg_prob1)*100
            df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, svc_reg_prob1))))
            df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, svc_reg_prob1))
            df.loc[ind,'MSE'] = mean_squared_error(y_test, svc_reg_prob1)
            df.loc[ind,'MAE']=mean_absolute_error(y_test, svc_reg_prob1)
            #df.loc[ind,'AIC']=aic.aic(y_test, svc_reg_prob1,X_train.shape[1])
            df.loc[ind,'BIC']=bic.bic(y_test, svc_reg_prob1,X_train.shape[1])
            df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))

            print("SVC done")
            ind=ind+1



      #Ensemble
      ########################################################################################################
      ##Ensemble(1) Finding all possible combination of above model and find out the best combination based on testing data accuracy
      ########################################################################################################
      arr1=np.empty((len(y_test),0))
      for i in range(0,len(df)):
          arr1=np.hstack((arr1,np.reshape(df.loc[i,'model'].predict(X_test),(len(y_test),1))))

      min_rmse=100000
      max_seq=0
      for i in range(2,len(df)+1):
          comb=list(combinations(enumerate(arr1.T), i))
          for j in range(0,len(comb)):
              m=np.empty((len(y_test),0))
              for x in range(0,len(comb[j])):
                  m=np.hstack((m,np.reshape(comb[j][x][1],(len(y_test),1))))
              arr=np.mean(m,axis=1)
              rmse= sqrt(mean_squared_error(y_test, arr))
              seq=np.array(comb[j])[:,0]
              if rmse<min_rmse:
                  min_rmse = rmse
                  max_seq=seq

      print("this is what you are printing",max_seq)

      ##############################################################################

      ##Ensemble(2) List of the best combination from the above method
      ########################################################################################################
      name=''
      df_en=pd.DataFrame(index = range(1000), columns=['Name','model'])
      for i in range(0,len(max_seq)):
        df_en.at[i,'Name']= df.at[max_seq[i],'Name']
        val = df.at[max_seq[i],'model']
        df_en['model'][i] = val
        name=name+df['Name'][max_seq[i]]+'+'


      df_en.dropna(axis=0,inplace=True)
      ########################################################################################################


      ##Ensemble(3) Making an esemble model of the best combination
      ########################################################################################################
      df.loc[ind,'Name']=('Ensemble '+'(' + name[:-1] + ')')
      df.loc[ind,'model']=VotingRegressor(df_en.values,n_jobs=-1)
      df.loc[ind,'param']="Defualt"
      Start = time.time()
      df.loc[ind,'model'].fit(X_train, y_train)
      ensemble_pred = df.loc[ind,'model'].predict(X_test)
      End = time.time()
      df.loc[ind,'accuracy']=r2_score(y_test, ensemble_pred)*100
      df.loc[ind,'Accuracy%']="{:.2%}".format(Decimal(str(r2_score(y_test, ensemble_pred))))
      df.loc[ind,'RMSE']=sqrt(mean_squared_error(y_test, ensemble_pred))
      df.loc[ind,'MSE']=mean_squared_error(y_test, ensemble_pred)
      df.loc[ind,'MAE']=mean_absolute_error(y_test, ensemble_pred)
      #df.loc[ind,'AIC']=aic.aic(y_test, ensemble_pred,X_train.shape[1])
      df.loc[ind,'BIC']=bic.bic(y_test, ensemble_pred,X_train.shape[1])
      df.loc[ind,'Total time (hh:mm:ss)']= time.strftime("%H:%M:%S", time.gmtime(End-Start))
      ind=ind+1

      best_info=df.sort_values('RMSE',ignore_index=True,ascending=True).loc[0,:]
      best_name=best_info['Name']
      best_mod=best_info['model']
      best_acc=best_info['accuracy']
      best_param=best_info['param']

      return best_name,best_mod, best_acc, best_param,df
