from pysurvival.models.parametric import *
from pysurvival.models.survival_forest import *
from Metrics import pysurv_eval
import warnings
import time


warnings.filterwarnings('ignore')

seed = 42

# This will work only for Uncensored and Right censored data
# PySurvival doesn't has censorship attribute


class pysurv_Modelling:
    def __init__(self, X_train, X_test, resultsDict, predictionsDict, T_col, E_col, modelsDict):
        self.X_train = X_train
        self.X_test = X_test
        self.T_col = T_col
        self.E_col = E_col
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict
        self.modelsDict = modelsDict

    def randomForest(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = RandomSurvivalForestModel()
        if winner:
            return model
        print('RandomSurvivalForestModel running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['RandomSurvivalForestModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['RandomSurvivalForestModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['RandomSurvivalForestModel'] = model

    def condForest(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = ConditionalSurvivalForestModel()
        if winner:
            return model
        print('ConditionalSurvivalForestModel running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['ConditionalSurvivalForestModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['ConditionalSurvivalForestModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['ConditionalSurvivalForestModel'] = model

    def exponential(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = ExponentialModel()
        if winner:
            return model
        print('ExponentialModel  running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['ExponentialModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['ExponentialModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['ExponentialModel'] = model

    def logLogistic(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = LogLogisticModel()
        if winner:
            return model
        print('LogLogisticModel running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['LogLogisticModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['LogLogisticModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['LogLogisticModel'] = model

    def weibull(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = WeibullModel()
        if winner:
            return model
        print('WeibullModel running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['WeibullModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['WeibullModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['WeibullModel'] = model

    def logNormal(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        model = LogNormalModel(penalizer=0.01)
        if winner:
            return model
        print('LogNormalModel running..')
        model.fit(X_t, T_train, E_train)
        self.resultsDict['LogNormalModel'] = pysurv_eval(model, X_tt, T_test, E_test)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        self.predictionsDict['LogNormalModel'] = [survival, risk, hazard, density, cdf]
        self.modelsDict['LogNormalModel'] = model

    def modeller(self):
        current = time.time()
        try:
            self.logNormal()
        except:
            pass
        finally:
            try:
                self.logLogistic()
            except:
                pass
            finally:
                try:
                    self.weibull()
                except:
                    pass
                finally:
                    try:
                        self.condForest()
                    except:
                        pass
                    finally:
                        try:
                            self.randomForest()
                        except:
                            pass
                        finally:
                            try:
                                self.exponential()
                                print(f'Total Modelling Time Taken : {time.time() - current}')
                            except:
                                pass

    def getWinnerModel(self, winnerName):
        switcher = {
            'LogNormalModel': self.logNormal(winner=True),
            'LogLogisticModel': self.logLogistic(winner=True),
            'WeibullModel': self.weibull(winner=True),
            'ConditionalSurvivalForestModel': self.condForest(winner=True),
            'RandomSurvivalForestModel': self.randomForest(winner=True),
            'ExponentialModel': self.exponential(winner=True)
        }
        return switcher[winnerName]

    def scoring(self, model):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        T_train = X_train[T_col]
        E_train = X_train[E_col]
        T_test = X_test[T_col]
        E_test = X_test[E_col]
        X_t = X_train.drop([T_col, E_col], axis=1)
        X_tt = X_test.drop([T_col, E_col], axis=1)
        print('...Running Scoring...')
        model.fit(X_t, T_train, E_train)
        survival = model.predict_survival(X_tt)
        risk = model.predict_risk(X_tt)
        hazard = model.predict_hazard(X_tt)
        density = model.predict_density(X_tt)
        cdf = model.predict_cdf(X_tt)
        predictions = [survival, risk, hazard, density, cdf]
        return predictions

