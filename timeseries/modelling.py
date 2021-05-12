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
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
        mean = self.y_test.mean()
        mean = np.array([mean for u in range(len(self.X_test))])
        self.resultsDict['Naive mean'] = evaluate(self.y_test, mean)
        self.predictionsDict['Naive mean'] = mean

    def bayesianRegression(self):
        print("Bayesian Model Running...")
        reg = linear_model.BayesianRidge()
        reg.fit(self.X_train, self.y_train)
        yhat = reg.predict(self.X_test)
        self.resultsDict['BayesianRidge'] = evaluate(self.y_test, yhat)
        self.predictionsDict['BayesianRidge'] = yhat

    def lassoRegression(self):
        print("Lasso Model Running...")
        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(self.X_train, self.y_train)
        yhat = reg.predict(self.X_test)
        self.resultsDict['Lasso'] = evaluate(self.y_test, yhat)
        self.predictionsDict['Lasso'] = yhat

    def randomForest(self):
        print("Random Forest Running...")
        reg = RandomForestRegressor(max_depth=5, random_state=0)
        reg.fit(self.X_train, self.y_train)
        yhat = reg.predict(self.X_test)
        self.resultsDict['Randomforest'] = evaluate(self.y_test, yhat)
        self.predictionsDict['Randomforest'] = yhat

    def XGB(self):
        print("XGB Running...")
        reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        reg.fit(self.X_train, self.y_train,
                verbose=False)  # Change verbose to True if you want to see it train
        yhat = reg.predict(self.X_test)
        self.resultsDict['XGBoost'] = evaluate(self.y_test, yhat)
        self.predictionsDict['XGBoost'] = yhat

    def LGBM(self):
        print("LGBM Running...")
        lightGBM = lgb.LGBMRegressor()
        lightGBM.fit(self.X_train, self.y_train)
        yhat = lightGBM.predict(self.X_test)
        self.resultsDict['Lightgbm'] = evaluate(self.y_test, yhat)
        self.predictionsDict['Lightgbm'] = yhat

    def SVM(self):
        print('SVM Running...')
        reg = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        reg.fit(self.X_train, self.y_train)
        yhat = reg.predict(self.X_test)
        self.resultsDict['SVM RBF'] = evaluate(self.y_test, yhat)
        self.predictionsDict['SVM RBF'] = yhat

    def KNN(self):
        print("KNN Running...")
        reg = KNeighborsRegressor(n_neighbors=2)
        reg.fit(self.X_train, self.y_train)
        yhat = reg.predict(self.X_test)
        self.resultsDict['Kneighbors'] = evaluate(self.y_test, yhat)
        self.predictionsDict['Kneighbors'] = yhat

    def Ensemble(self):
        print('Trying XGB + Light Ensemble...')
        self.predictionsDict['EnsembleXG+LIGHT'] = (
            self.predictionsDict['XGBoost'] + self.predictionsDict['Lightgbm'])/2
        self.resultsDict['EnsembleXG+LIGHT'] = evaluate(
            self.y_test.values, self.predictionsDict['EnsembleXG+LIGHT'])
        print('Trying RF + XGBoost Ensemble...')
        self.predictionsDict['EnsembleRF+XG'] = (
            self.predictionsDict['Randomforest'] + self.predictionsDict['XGBoost'])/2
        self.resultsDict['EnsembleRF+XG'] = evaluate(
            self.y_test.values, self.predictionsDict['EnsembleRF+XG'])
        print('Trying RF + Light Ensemble...')
        self.predictionsDict['EnsembleRF+LIGHT'] = (
            self.predictionsDict['Randomforest'] + self.predictionsDict['Lightgbm'])/2
        self.resultsDict['EnsembleRF+LIGHT'] = evaluate(
            self.y_test.values, self.predictionsDict['EnsembleRF+LIGHT'])
        print('Trying XG + RF + Light Ensemble...')
        self.predictionsDict['EnsembleXG+LIGHT+RF'] = (
            self.predictionsDict['XGBoost'] + self.predictionsDict['Lightgbm'] + self.predictionsDict['Randomforest'])/3
        self.resultsDict['EnsembleXG+LIGHT+RF'] = evaluate(
            self.y_test.values, self.predictionsDict['EnsembleXG+LIGHT+RF'])

    def modeller(self):
        current = time.time()
        self.naiveModel()
        self.bayesianRegression()
        self.lassoRegression()
        self.randomForest()
        self.XGB()
        self.LGBM()
        self.SVM()
        self.KNN()
        self.Ensemble()
        print(f'Total Modelling Time Taken : {time.time()-current}')
