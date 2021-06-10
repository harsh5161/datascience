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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from tqdm import tqdm, tqdm_notebook
import pmdarima as pm


class Modelling:
    def __init__(self, X_train, X_test, y_train, y_test, resultsDict, predictionsDict,period):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict
        self.period = period

    def getForecasts(self,model,train,trainY,testX):
        # print(train.dtypes)
        model.fit(train,trainY)
        yhat = model.predict(testX)
        return yhat[0]
    
    def naiveModel(self,winner=False):
        if winner:
            return None
        print("Naive Modelling Running...")
        mean = self.y_test.mean()
        mean = np.array([mean for u in range(len(self.X_test))])
        self.resultsDict['Naive mean'] = evaluate(self.y_test, mean)
        self.predictionsDict['Naive mean'] = mean

    def bayesianRegression(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = linear_model.BayesianRidge()
        if winner:
            return reg
        print("Bayesian Model Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['BayesianRidge'] = evaluate(self.y_test, predictions)
        self.predictionsDict['BayesianRidge'] = predictions

    def lassoRegression(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = linear_model.Lasso(alpha=0.1)
        if winner:
            return reg
        print("Lasso Model Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['Lasso'] = evaluate(self.y_test, predictions)
        self.predictionsDict['Lasso'] = predictions

    def randomForest(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = RandomForestRegressor(max_depth=5, random_state=42,n_estimators=50)
        if winner:
            return reg
        print("Random Forest Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['Randomforest'] = evaluate(self.y_test, predictions)
        self.predictionsDict['Randomforest'] = predictions

    def XGB(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        if winner:
            return reg
        print("XGB Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['XGBoost'] = evaluate(self.y_test, predictions)
        self.predictionsDict['XGBoost'] = predictions

    def LGBM(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = lgb.LGBMRegressor(n_estimators=50)
        if winner:
            return reg
        print("LGBM Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['Lightgbm'] = evaluate(self.y_test, predictions)
        self.predictionsDict['Lightgbm'] = predictions

    def SVM(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1,max_iter=2)
        if winner:
            return reg
        print('SVM Running...')
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['SVM RBF'] = evaluate(self.y_test, predictions)
        self.predictionsDict['SVM RBF'] = predictions

    def KNN(self,winner=False):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        reg = KNeighborsRegressor(n_neighbors=2)
        if winner:
            return reg
        print("KNN Running...")
        predictions = list()
        X_history = X_train.to_numpy()
        print(X_history)
        y_history = y_train.to_numpy()
        X_test = X_test.to_numpy()
        for i in range(len(X_test)):
            yhat = self.getForecasts(reg,X_history,y_history,X_test[i].reshape(1,-1))
            predictions.append(yhat)
            X_history = np.append(X_history,np.array([X_test[i]]),axis=0)
            y_history = np.append(y_history,np.array([y_test[i]]),axis=0)
        self.resultsDict['Kneighbors'] = evaluate(self.y_test, predictions)
        self.predictionsDict['Kneighbors'] = predictions

    def HWES(self,winner=False):
        print("HWES Running...")
        yhat = list()
        for t in tqdm(range(len(self.y_test))):
            temp_train = self.y_train[:len(self.y_train)+t]
            model = ExponentialSmoothing(self.y_train)
            model_fit = model.fit()
            predictions = model_fit.predict(
                start=len(temp_train), end=len(temp_train))
            yhat = yhat + [predictions]

        yhat = pd.concat(yhat)
        self.resultsDict['HWES'] = evaluate(self.y_test, yhat.values)
        self.predictionsDict['HWES'] = yhat.values

    def SARIMAX(self):
        print("SARIMAX Running...")
        autoModel = pm.auto_arima(self.y_train, trace=True, error_action='ignore',
                                  suppress_warnings=True, seasonal=True, m=self.period, stepwise=True)
        autoModel.fit(self.y_train)

        order = autoModel.order
        seasonalOrder = autoModel.seasonal_order
        yhat = list()
        for t in tqdm(range(len(self.y_test))):
            temp_train = self.y_train[:len(self.y_train)+t]
            model = SARIMAX(temp_train, order=order,
                            seasonal_order=seasonalOrder)
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(
                start=len(temp_train), end=len(temp_train), dynamic=False)
            yhat = yhat + [predictions]

        yhat = pd.concat(yhat)
        self.resultsDict['AutoSARIMAX {0},{1}'.format(
            order, seasonalOrder)] = evaluate(self.y_test, yhat.values)
        self.predictionsDict['AutoSARIMAX {0},{1}'.format(
            order, seasonalOrder)] = yhat.values

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
        self.HWES()
        self.bayesianRegression()
        self.lassoRegression()
        self.randomForest()
        self.XGB()
        self.LGBM()
        self.KNN()
        # self.SARIMAX()
        # self.Ensemble()
        print(f'Total Modelling Time Taken : {time.time()-current}')

    def getWinnerModel(self,winnerName):
        switcher = {
            'Naive Mean' : self.naiveModel(winner=True),
            'Lasso' : self.lassoRegression(winner=True),
            'Lightgbm': self.LGBM(winner=True),
            'Randomforest' : self.randomForest(winner=True),
            'XGBoost' : self.XGB(winner=True),
            'Kneighbours' : self.KNN(winner=True),
            'BayesianRidge' : self.bayesianRegression(winner=True)
        }
        return switcher[winnerName]
    
    def scoring(self,model,predictions):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        # X_test = self.X_test.copy()
        # y_test = self.y_test.copy()
        