from metrics import evaluate
from plots import bar_metrics


class Modelling:
    def __init__(self, X_train, X_test, y_train, y_test, resultsDict, predictionsDict):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def naiveModel(self):
        mean = y_test.mean()
        mean = np.array([mean for u in range(len(X_test))])
        resultsDict['Naive mean'] = evaluate(y_test, mean)
        predictionsDict['Naive mean'] = mean

    def bayesianRegression(self):
        reg = linear_model.BayesianRidge()
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['BayesianRidge'] = evaluate(y_test, yhat)
        predictionsDict['BayesianRidge'] = yhat

    def lassoRegression(self):
        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['Lasso'] = evaluate(y_test, yhat)
        predictionsDict['Lasso'] = yhat

    def randomForest(self):
        reg = RandomForestRegressor(max_depth=2, random_state=0)
        reg.fit(X_train, y_train)
        yhat = reg.predict(X_test)
        resultsDict['Randomforest'] = evaluate(y_test, yhat)
        predictionsDict['Randomforest'] = yhat

    def XGB(self):
        reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        reg.fit(X_train, y_train,
                verbose=False)  # Change verbose to True if you want to see it train
        yhat = reg.predict(X_test)
        resultsDict['XGBoost'] = evaluate(y_test, yhat)
        predictionsDict['XGBoost'] = yhat

    def LGBM(self):
        lightGBM = lgb.LGBMRegressor()
        lightGBM.fit(X_train, y_train)
        yhat = lightGBM.predict(X_test)
        resultsDict['Lightgbm'] = evaluate(y_test, yhat)
        predictionsDict['Lightgbm'] = yhat
