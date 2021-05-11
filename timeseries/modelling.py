from metrics import evaluate
from plots import bar_metrics
import time
import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb


class Modelling:
    def __init__(self, X_train, X_test, y_train, y_test, resultsDict, predictionsDict):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def naiveModel(self):
        print("Naive Modelling Running...")
        mean = y_test.mean()
        mean = np.array([mean for u in range(len(X_test))])
        resultsDict['Naive mean'] = evaluate(y_test, mean)
        predictionsDict['Naive mean'] = mean

    def bayesianRegression(self):
        print("Bayesian Model Running...")
        reg = linear_model.BayesianRidge()
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['BayesianRidge'] = evaluate(y_test, yhat)
        predictionsDict['BayesianRidge'] = yhat

    def lassoRegression(self):
        print("Lasso Model Running...")
        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['Lasso'] = evaluate(y_test, yhat)
        predictionsDict['Lasso'] = yhat

    def randomForest(self):
        print("Random Forest Running...")
        reg = RandomForestRegressor(max_depth=5, random_state=0)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['Randomforest'] = evaluate(y_test, yhat)
        predictionsDict['Randomforest'] = yhat

    def XGB(self):
        print("XGB Running...")
        reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        reg.fit(X_train, y_train,
                verbose=False)  # Change verbose to True if you want to see it train
        yhat = reg.predict(X_test)
        resultsDict['XGBoost'] = evaluate(y_test, yhat)
        predictionsDict['XGBoost'] = yhat

    def LGBM(self):
        print("LGBM Running...")
        lightGBM = lgb.LGBMRegressor()
        lightGBM.fit(X_train, y_train)
        yhat = lightGBM.predict(X_test)
        resultsDict['Lightgbm'] = evaluate(y_test, yhat)
        predictionsDict['Lightgbm'] = yhat

    def SVM(self):
        print('SVM Running...')
        reg = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['SVM RBF'] = evaluate(y_test, yhat)
        predictionsDict['SVM RBF'] = yhat

    def KNN(self):
        print("KNN Running...")
        reg = KNeighborsRegressor(n_neighbors=2)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['Kneighbors'] = evaluate(y_test, yhat)
        predictionsDict['Kneighbors'] = yhat

    def modeller(self):
        current = time.time()
        naiveModel()
        bayesianRegression()
        lassoRegression()
        randomForest()
        XGB()
        LGBM()
        SVM()
        KNN()
        print(f'Total Modelling Time Taken : {time.time()-current}')
