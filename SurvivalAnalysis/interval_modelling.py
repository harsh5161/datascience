from lifelines import *
from Metrics import evaluate_3
import time
import warnings

warnings.filterwarnings('ignore')

seed = 42


class Uni_interval:
    def __init__(self, lower_bound_t, upper_bound_t, lower_bound_tt,
                 upper_bound_tt, E_train, E_test, resultsDict, predictionsDict, modelsDict):
        self.lower_bound_t = lower_bound_t
        self.upper_bound_t = upper_bound_t
        self.lower_bound_tt = lower_bound_tt
        self.upper_bound_tt = upper_bound_tt
        self.E_train = E_train
        self.E_test = E_test
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict
        self.modelsDict = modelsDict

    def get_pred(self, model, lower_bound_t, upper_bound_t, lower_bound_tt,
                 upper_bound_tt, E_train):
        model.fit_interval_censoring(lower_bound_t, upper_bound_t, E_train)
        upper_pred = model.predict(upper_bound_tt)
        lower_pred = model.predict(lower_bound_tt)
        return lower_pred, upper_pred, model

    # def kaplanMeier(self, winner=False):
    #     lower_bound_t = self.lower_bound_t.copy()
    #     E_train = self.E_train.copy()
    #     upper_bound_t = self.upper_bound_t.copy()
    #     lower_bound_tt = self.lower_bound_tt.copy()
    #     upper_bound_tt = self.upper_bound_tt.copy()
    #     E_test = self.E_test.copy()
    #     model = KaplanMeierFitter()
    #     if winner:
    #         return model
    #     print('KaplanMeierFitter running..')
    #     lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
    #                                                   lower_bound_tt, upper_bound_tt, E_train)
    #     self.predictionsDict['KaplanMeierFitter'] = [lower_pred, upper_pred]
    #     self.resultsDict['KaplanMeierFitter'] = evaluate_1(upper_bound_tt, lower_pred.iloc[:, 0], E_test)
    #     self.modelsDict['KaplanMeierFitter'] = model

    def exponential(self, winner=False):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        upper_bound_tt = self.upper_bound_tt.copy()
        E_test = self.E_test.copy()
        model = ExponentialFitter()
        if winner:
            return model
        print('ExponentialFitter running..')
        lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                      lower_bound_tt, upper_bound_tt, E_train)
        self.predictionsDict['ExponentialFitter'] = [lower_pred, upper_pred]
        self.resultsDict['ExponentialFitter'] = evaluate_3(model)
        self.modelsDict['ExponentialFitter'] = model

    def logLogistic(self, winner=False):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        E_test = self.E_test.copy()
        upper_bound_tt = self.upper_bound_tt.copy()
        model = LogLogisticFitter()
        if winner:
            return model
        print('LogLogisticFitter running..')
        lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                      lower_bound_tt, upper_bound_tt, E_train)
        self.predictionsDict['LogLogisticFitter'] = [lower_pred, upper_pred]
        self.resultsDict['LogLogisticFitter'] = evaluate_3(model)
        self.modelsDict['LogLogisticFitter'] = model

    def logNormal(self, winner=False):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        upper_bound_tt = self.upper_bound_tt.copy()
        E_test = self.E_test.copy()
        model = LogNormalFitter()
        if winner:
            return model
        print('LogNormalFitter running..')
        lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                      lower_bound_tt, upper_bound_tt, E_train)
        self.predictionsDict['LogNormalFitter'] = [lower_pred, upper_pred]
        self.resultsDict['LogNormalFitter'] = evaluate_3(model)
        self.modelsDict['LogNormalFitter'] = model

    def generalizedGamma(self, winner=False):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        upper_bound_tt = self.upper_bound_tt.copy()
        E_test = self.E_test.copy()
        model = GeneralizedGammaFitter()
        if winner:
            return model
        print('GeneralizedGammaFitter running..')
        lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                      lower_bound_tt, upper_bound_tt, E_train)
        self.predictionsDict['GeneralizedGammaFitter'] = [lower_pred, upper_pred]
        self.resultsDict['GeneralizedGammaFitter'] = evaluate_3(model)
        self.modelsDict['GeneralizedGammaFitter'] = model

    def weibull(self, winner=False):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        E_test = self.E_test.copy()
        upper_bound_tt = self.upper_bound_tt.copy()
        model = WeibullFitter()
        if winner:
            return model
        print('WeibullFitter running..')
        lower_pred, upper_pred, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                      lower_bound_tt, upper_bound_tt, E_train)
        self.predictionsDict['WeibullFitter'] = [lower_pred, upper_pred]
        self.resultsDict['WeibullFitter'] = evaluate_3(model)
        self.modelsDict['WeibullFitter'] = model

    def modeller(self):
        current = time.time()
        # self.kaplanMeier()
        self.generalizedGamma()
        self.weibull()
        self.exponential()
        self.logLogistic()
        self.logNormal()
        print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        switcher = {
            # 'KaplanMeierFitter': self.kaplanMeier(winner=True),
            'GeneralizedGammaFitter': self.generalizedGamma(winner=True),
            'WeibullFitter': self.weibull(winner=True),
            'ExponentialFitter': self.exponential(winner=True),
            'LogNormalFitter': self.logNormal(winner=True),
            'LogLogisticFitter': self.logLogistic(winner=True),
        }
        return switcher[winnerName]

    def scoring(self, model):
        lower_bound_t = self.lower_bound_t.copy()
        E_train = self.E_train.copy()
        upper_bound_t = self.upper_bound_t.copy()
        lower_bound_tt = self.lower_bound_tt.copy()
        upper_bound_tt = self.upper_bound_tt.copy()

        print('...Running Scoring...')
        lower_predictions, upper_predictions, model = self.get_pred(model, lower_bound_t, upper_bound_t,
                                                                    lower_bound_tt, upper_bound_tt, E_train)
        return [lower_predictions, upper_predictions]


class Multi_interval:
    def __init__(self, X_train, X_test, lower_bound_t, upper_bound_t, lower_bound_tt,
                 upper_bound_tt, E_train, E_test, resultsDict, predictionsDict):
        self.X_train = X_train
        self.X_test = X_test
        self.lower_bound_t = lower_bound_t
        self.upper_bound_t = upper_bound_t
        self.lower_bound_tt = lower_bound_tt
        self.upper_bound_tt = upper_bound_tt
        self.E_train = E_train
        self.E_test = E_test
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def weibullAFT(self, winner=False):
        lower_bound_t = self.lower_bound_t
        E_train = self.E_train
        upper_bound_t = self.upper_bound_t
        lower_bound_tt = self.lower_bound_tt
        upper_bound_tt = self.upper_bound_tt
        E_test = self.E_test
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        model = WeibullAFTFitter()
        if winner:
            return model
        print('WeibullAFTFitter running..')
        model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['WeibullAFTFitter'] = [survival, cum_hazard, hazard]
        self.resultsDict['WeibullAFTFitter'] = evaluate_3(model=model)

    def logLogisticAFT(self, winner=False):
        lower_bound_t = self.lower_bound_t
        E_train = self.E_train
        upper_bound_t = self.upper_bound_t
        lower_bound_tt = self.lower_bound_tt
        upper_bound_tt = self.upper_bound_tt
        E_test = self.E_test
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        model = LogLogisticAFTFitter()
        if winner:
            return model
        print('LogLogisticAFTFitter running..')
        model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogLogisticAFTFitter'] = [survival, cum_hazard, hazard]
        self.resultsDict['LogLogisticAFTFitter'] = evaluate_3(model=model)

    def logNormalAFT(self, winner=False):
        lower_bound_t = self.lower_bound_t
        E_train = self.E_train
        upper_bound_t = self.upper_bound_t
        lower_bound_tt = self.lower_bound_tt
        upper_bound_tt = self.upper_bound_tt
        E_test = self.E_test
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        model = LogNormalAFTFitter()
        if winner:
            return model
        print('LogNormalAFTFitter running..')
        model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test)
        hazard = model.predict_hazard(X_test)
        self.predictionsDict['LogNormalAFTFitter'] = [survival, cum_hazard, hazard]
        self.resultsDict['LogNormalAFTFitter'] = evaluate_3(model=model)

    def generalizedGammaAFT(self, winner=False):
        lower_bound_t = self.lower_bound_t
        E_train = self.E_train
        upper_bound_t = self.upper_bound_t
        lower_bound_tt = self.lower_bound_tt
        upper_bound_tt = self.upper_bound_tt
        E_test = self.E_test
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        model = GeneralizedGammaRegressionFitter(penalizer=0.01)
        if winner:
            return model
        print('GeneralizedGammaRegressionFitter running..')
        model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
        cum_hazard = model.predict_cumulative_hazard(X_test)
        survival = model.predict_survival_function(X_test)
        self.predictionsDict['GeneralizedGammaRegressionFitter'] = [survival, cum_hazard]
        self.resultsDict['GeneralizedGammaRegressionFitter'] = evaluate_3(model=model)

    def modeller(self):
        current = time.time()
        try:
            self.generalizedGammaAFT()
        except:
            pass
        finally:
            try:
                self.weibullAFT()
            except:
                pass
            finally:
                try:
                    self.logLogisticAFT()
                except:
                    pass
                finally:
                    try:
                        self.logNormalAFT()
                        print(f'Total Modelling Time Taken : {time.time() - current}')
                    except:
                        pass

    def getWinnerModel(self, winnerName):
        switcher = {
            'GeneralizedGammaRegressionFitter': self.generalizedGammaAFT(winner=True),
            'WeibullAFTFitter': self.weibullAFT(winner=True),
            'LogNormalAFTFitter': self.logNormalAFT(winner=True),
            'LogLogisticAFTFitter': self.logLogisticAFT(winner=True),
        }
        return switcher[winnerName]

    def scoring(self, model, model_ins):
        lower_bound_t = self.lower_bound_t
        E_train = self.E_train
        upper_bound_t = self.upper_bound_t
        lower_bound_tt = self.lower_bound_tt
        upper_bound_tt = self.upper_bound_tt
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        print('...Running Scoring...')
        if model in ['WeibullAFTFitter', 'LogLogisticAFTFitter', 'LogNormalAFTFitter']:
            model = model_ins
            model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
            cum_hazard = model.predict_cumulative_hazard(X_test)
            survival = model.predict_survival_function(X_test)
            hazard = model.predict_hazard(X_test)
            predictions = [survival, cum_hazard, hazard]
            return predictions
        elif model == 'GeneralizedGammaRegressionFitter':
            model = model_ins
            model.fit_interval_censoring(X_train, lower_bound_t, upper_bound_t, E_train)
            cum_hazard = model.predict_cumulative_hazard(X_test)
            survival = model.predict_survival_function(X_test)
            predictions = [survival, cum_hazard]
            return predictions
