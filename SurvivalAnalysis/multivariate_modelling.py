from lifelines import *
from pysurvival.utils.display import *
from pysurvival.models.parametric import *
from pysurvival.models.multi_task import *
from pysurvival.models.semi_parametric import *
from pysurvival.models.non_parametric import *
from Metrics import evaluate_2
import time
import warnings

warnings.filterwarnings('ignore')

seed = 42


class Multi_Modelling:
    def __init__(self, X_train, X_test, T_train, T_test, E_train, E_test,
                 resultsDict, predictionsDict, T_col, E_col):
        self.T_train = T_train
        self.X_train = X_train
        self.X_test = X_test
        self.T_test = T_test
        self.E_train = E_train
        self.E_test = E_test
        self.T_col = T_col
        self.E_col = E_col
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def logLogisticAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = LogLogisticAFTFitter()
        if winner:
            return model
        print('LogLogisticFitter running..')
        model.fit(X_train, T_col, E_col)
        self.resultsDict['LogLogisticAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                              T_test, E_test)
        survival = model.predict_survival_function(X_test, T_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogLogisticAFTFitter'] = [survival, cum_hazard, hazard]

    def logNormalAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = LogNormalAFTFitter()
        if winner:
            return model
        print('LogNormalAFTFitter running..')
        model.fit(X_train, T_col, E_col)
        self.resultsDict['LogNormalAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1), T_test,
                                                            E_test)
        survival = model.predict_survival_function(X_test, T_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogNormalAFTFitter'] = [survival, cum_hazard, hazard]

    def generalizedGammaAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = GeneralizedGammaRegressionFitter()
        if winner:
            return model
        print('GeneralizedGammaRegressionFitter running..')
        model.fit(X_train, T_col, E_col)
        self.resultsDict['GeneralizedGammaRegressionFitter'] = evaluate_2(model,
                                                                          X_test.drop(['T_col', 'E_col'], axis=1),
                                                                          T_test, E_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test, T_test)
        self.predictionsDict['GeneralizedGammaRegressionFitter'] = [survival, cum_hazard]

    def weibullAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = WeibullAFTFitter()
        if winner:
            return model
        print('WeibullAFTFitter running..')
        model.fit(X_train, T_col, E_col)
        self.resultsDict['WeibullAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                          T_test, E_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        survival = model.predict_survival_function(X_test, T_test)
        self.predictionsDict['WeibullAFTFitter'] = [survival, cum_hazard, hazard]

    def coxPH(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        E_test = self.E_test.copy()
        T_test = self.T_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = CoxPHFitter(penalizer=0.01)
        if winner:
            return model
        print('CoxPHFitter running..')
        model.fit(X_train, T_col, E_col)
        self.resultsDict['CoxPHFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1), T_test, E_test)
        log_hazard = model.predict_log_partial_hazard(X_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        partial_hazard = model.predict_partial_hazard(X_test)
        survival = model.predict_survival_function(X_test, T_test)
        self.predictionsDict['CoxPHFitter'] = [survival, cum_hazard, partial_hazard, log_hazard]

    def linearMTLR(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col.copy()
        E_col = self.E_col.copy()
        model = LinearMultiTaskModel()
        if winner:
            return model
        print('Multi-Task Logistic Regression running..')
        model.fit(X_train.drop([T_col, E_col], axis=1), T_train, E_train)
        self.resultsDict['LinearMultiTaskModel'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                              T_test, E_test)
        survival = model.predict_survival(X_test.drop([T_col, E_col], axis=1), T_test)
        hazard = model.predict_hazard(X_test.drop([T_col, E_col], axis=1), T_test)
        self.predictionsDict['LinearMultiTaskModel'] = [survival, hazard]

    def exponential(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col.copy()
        E_col = self.E_col.copy()
        model = ExponentialModel()
        if winner:
            return model
        print('Exponential Parametric model running..')
        model.fit(X_train.drop([T_col, E_col], axis=1), T_train, E_train)
        self.resultsDict['ExponentialModel'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                          T_test, E_test)
        survival = model.predict_survival(X_test.drop([T_col, E_col], axis=1), T_test)
        hazard = model.predict_hazard(X_test.drop([T_col, E_col], axis=1), T_test)
        self.predictionsDict['ExponentialModel'] = [survival, hazard]

    def modeller(self):
        current = time.time()
        self.exponential()
        self.coxPH()
        self.weibullAFT()
        self.logLogisticAFT()
        self.logNormalAFT()
        self.linearMTLR()
        self.generalizedGammaAFT()
        print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        switcher = {
            'ExponentialModel': self.exponential(winner=True),
            'CoxPHFitter': self.coxPH(winner=True),
            'WeibullAFTFitter': self.weibullAFT(winner=True),
            'LogLogisticAFTFitter': self.logLogisticAFT(winner=True),
            'LinearMultiTaskModel': self.linearMTLR(winner=True),
            'LogNormalAFTFitter': self.logNormalAFT(winner=True),
            'GeneralizedGammaRegressionFitter': self.generalizedGammaAFT(winner=True),
        }
        return switcher[winnerName]


class Multi_Modelling_left:
    def __init__(self, X_train, X_test, T_train, T_test, E_train, E_test,
                 resultsDict, predictionsDict, T_col, E_col):
        self.T_train = T_train
        self.X_train = X_train
        self.X_test = X_test
        self.T_test = T_test
        self.E_train = E_train
        self.E_test = E_test
        self.T_col = T_col
        self.E_col = E_col
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def logLogisticAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        E_test = self.E_test.copy()
        model = LogLogisticAFTFitter()
        if winner:
            return model
        print('LogLogisticFitter running..')
        model.fit_left_censoring(X_train, T_col, E_col)
        self.resultsDict['LogLogisticAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                              T_test, E_test)
        survival = model.predict_survival_function(X_test, T_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogLogisticAFTFitter'] = [survival, cum_hazard, hazard]

    def logNormalAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = LogNormalAFTFitter()
        if winner:
            return model
        print('LogNormalAFTFitter running..')
        model.fit_left_censoring(X_train, T_col, E_col)
        self.resultsDict['LogNormalAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                            T_test, E_test)
        survival = model.predict_survival_function(X_test, T_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogNormalAFTFitter'] = [survival, cum_hazard, hazard]

    def generalizedGammaAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = GeneralizedGammaRegressionFitter()
        if winner:
            return model
        print('GeneralizedGammaRegressionFitter running..')
        model.fit_left_censoring(X_train, T_col, E_col)
        self.resultsDict['GeneralizedGammaRegressionFitter'] = evaluate_2(model,
                                                                          X_test.drop(['T_col', 'E_col'], axis=1),
                                                                          T_test, E_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test, T_test)
        self.predictionsDict['GeneralizedGammaRegressionFitter'] = [survival, cum_hazard]

    def weibullAFT(self, winner=False):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        E_test = self.E_test.copy()
        T_test = self.T_test.copy()
        T_col = self.T_col
        E_col = self.E_col
        model = WeibullAFTFitter()
        if winner:
            return model
        print('WeibullAFTFitter running..')
        model.fit_left_censoring(X_train, T_col, E_col)
        self.resultsDict['WeibullAFTFitter'] = evaluate_2(model, X_test.drop(['T_col', 'E_col'], axis=1),
                                                          T_test, E_test)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        hazard = model.predict_hazard(X_test)
        survival = model.predict_survival_function(X_test, T_test)
        self.predictionsDict['WeibullAFTFitter'] = [survival, cum_hazard, hazard]

    def modeller(self):
        current = time.time()
        self.weibullAFT()
        self.logLogisticAFT()
        self.logNormalAFT()
        self.generalizedGammaAFT()
        print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        switcher = {
            'WeibullAFTFitter': self.weibullAFT(winner=True),
            'LogLogisticAFTFitter': self.logLogisticAFT(winner=True),
            'LogNormalAFTFitter': self.logNormalAFT(winner=True),
            'GeneralizedGammaRegressionFitter': self.generalizedGammaAFT(winner=True),
        }
        return switcher[winnerName]
