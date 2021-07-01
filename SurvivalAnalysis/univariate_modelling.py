import time
from lifelines import *
from pysurvival.utils.display import *
from pysurvival.models.parametric import *
from pysurvival.models.multi_task import *
from pysurvival.models.semi_parametric import *
from pysurvival.models.non_parametric import *
import warnings
from Metrics import evaluate_1

warnings.filterwarnings('ignore')

seed = 42


class Uni_Modelling:
    def __init__(self, T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict):
        self.T_train = T_train
        self.T_test = T_test
        self.E_train = E_train
        self.E_test = E_test
        self.resultsDict = resultsDict
        self.modelsDict = modelsDict
        self.predictionsDict = predictionsDict

    def get_score(self, model, T_train, T_test, E_train, E_test):
        model.fit(T_train, E_train)
        yhat_tt = model.predict(T_test)
        return yhat_tt, model

    def kaplanMeier(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = KaplanMeierFitter()
        if winner:
            return model
        print('KaplanMeierFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['KaplanMeierFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['KaplanMeierFitter'] = yhat_tt
        self.modelsDict['KaplanMeierFitter'] = model

    def exponential(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = ExponentialFitter()
        if winner:
            return model
        print('ExponentialFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['ExponentialFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['ExponentialFitter'] = yhat_tt
        self.modelsDict['ExponentialFitter'] = model

    def logLogistic(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = LogLogisticFitter()
        if winner:
            return model
        print('LogLogisticFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['LogLogisticFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['LogLogisticFitter'] = yhat_tt
        self.modelsDict['LogLogisticFitter'] = model

    def logNormal(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = LogNormalFitter()
        if winner:
            return model
        print('LogNormalFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['LogNormalFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['LogNormalFitter'] = yhat_tt
        self.modelsDict['LogNormalFitter'] = model

    def generalizedGamma(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = GeneralizedGammaFitter()
        if winner:
            return model
        print('GeneralizedGammaFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['GeneralizedGammaFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['GeneralizedGammaFitter'] = yhat_tt
        self.modelsDict['GeneralizedGammaFitter'] = model

    def weibull(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = WeibullFitter()
        if winner:
            return model
        print('WeibullFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['WeibullFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['WeibullFitter'] = yhat_tt
        self.modelsDict['WeibullFitter'] = model

    def nelsonAalen(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = NelsonAalenFitter()
        if winner:
            return model
        print('NelsonAalenFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['NelsonAalenFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['NelsonAalenFitter'] = yhat_tt
        # self.modelsDict['NelsonAalenFitter'] = model

    def modeller(self):
        current = time.time()
        self.kaplanMeier()
        self.nelsonAalen()
        self.generalizedGamma()
        self.weibull()
        self.exponential()
        self.logLogistic()
        self.logNormal()
        print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        switcher = {
            'KaplanMeierFitter': self.kaplanMeier(winner=True),
            'NelsonAalenFitter': self.nelsonAalen(winner=True),
            'GeneralizedGammaFitter': self.generalizedGamma(winner=True),
            'WeibullFitter': self.weibull(winner=True),
            'ExponentialFitter': self.exponential(winner=True),
            'LogNormalFitter': self.logNormal(winner=True),
            'LogLogisticFitter': self.logLogistic(winner=True),
        }
        return switcher[winnerName]


class Uni_Modelling_left:
    def __init__(self, T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict):
        self.T_train = T_train
        self.T_test = T_test
        self.E_train = E_train
        self.E_test = E_test
        self.resultsDict = resultsDict
        self.modelsDict = modelsDict
        self.predictionsDict = predictionsDict

    def get_score(self, model, T_train, T_test, E_train, E_test):
        model.fit_left_censoring(T_train, E_train)
        yhat_tt = model.predict(T_test)
        return yhat_tt, model

    def kaplanMeier(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = KaplanMeierFitter()
        if winner:
            return model
        print('KaplanMeierFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['KaplanMeierFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['KaplanMeierFitter'] = yhat_tt
        self.modelsDict['KaplanMeierFitter'] = model

    def exponential(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = ExponentialFitter()
        if winner:
            return model
        print('ExponentialFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['ExponentialFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['ExponentialFitter'] = yhat_tt
        self.modelsDict['ExponentialFitter'] = model

    def logLogistic(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = LogLogisticFitter()
        if winner:
            return model
        print('LogLogisticFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['LogLogisticFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['LogLogisticFitter'] = yhat_tt
        self.modelsDict['LogLogisticFitter'] = model

    def logNormal(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = LogNormalFitter()
        if winner:
            return model
        print('LogNormalFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['LogNormalFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['LogNormalFitter'] = yhat_tt
        self.modelsDict['LogNormalFitter'] = model

    def generalizedGamma(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = GeneralizedGammaFitter()
        if winner:
            return model
        print('GeneralizedGammaFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['GeneralizedGammaFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['GeneralizedGammaFitter'] = yhat_tt
        self.modelsDict['GeneralizedGammaFitter'] = model

    def weibull(self, winner=False):
        T_train = self.T_train.copy()
        E_train = self.E_train.copy()
        T_test = self.T_test.copy()
        E_test = self.E_test.copy()
        model = WeibullFitter()
        if winner:
            return model
        print('WeibullFitter running..')
        yhat_tt, model = self.get_score(model, T_train, T_test, E_train, E_test)
        self.resultsDict['WeibullFitter'] = evaluate_1(T_test, yhat_tt, E_test)
        self.predictionsDict['WeibullFitter'] = yhat_tt
        self.modelsDict['WeibullFitter'] = model

    def modeller(self):
        current = time.time()
        self.kaplanMeier()
        self.generalizedGamma()
        self.weibull()
        self.exponential()
        self.logLogistic()
        self.logNormal()
        print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        switcher = {
            'KaplanMeierFitter': self.kaplanMeier(winner=True),
            'GeneralizedGammaFitter': self.generalizedGamma(winner=True),
            'WeibullFiiter': self.weibull(winner=True),
            'ExponentialFitter': self.exponential(winner=True),
            'LogNormalFitter': self.logNormal(winner=True),
            'LogLogisticFitter': self.logLogistic(winner=True),
        }
        return switcher[winnerName]
