from pysurvival.models.parametric import *
from pysurvival.models.survival_forest import *
from pysurvival.utils.display import integrated_brier_score
from metrics_pysurv import pysurv_eval
import warnings
import time
import joblib
from dask.distributed import Client, progress

warnings.filterwarnings('ignore')

seed = 42

client = Client()


# This will work only for Uncensored and Right censored data
# PySurvival doesn't has censorship attribute


class pysurv_Modelling:
    def __init__(self, X_t, X_tt, resultsDict, predictionsDict, T_col, E_col, modelsDict,
                 E_test, T_test, T_train, E_train):
        self.X_t = X_t
        self.X_tt = X_tt
        self.T_col = T_col
        self.E_col = E_col
        self.E_test = E_test
        self.E_train = E_train
        self.T_test = T_test
        self.T_train = T_train
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict
        self.modelsDict = modelsDict

    def randomForest(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        model = RandomSurvivalForestModel()
        if winner:
            return model
        print('RandomSurvivalForestModel running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train)
            print('Trained!')
            self.resultsDict['RandomSurvivalForestModel'] = pysurv_eval(model)
            print('Model has been evaluated')
            survival = model.predict_survival(X_tt)
            print('Predicted Survival Function')
            risk = model.predict_risk(X_tt)
            print('Predicted Risk')
            hazard = model.predict_hazard(X_tt)
            print('Predicted Hazard Function')
            self.predictionsDict['RandomSurvivalForestModel'] = [survival, risk, hazard]
            self.modelsDict['RandomSurvivalForestModel'] = model

    def condForest(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        # with joblib.parallel_backend('dask'):
        model = ConditionalSurvivalForestModel()
        if winner:
            return model
        print('ConditionalSurvivalForestModel running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train)
            print('Trained!')
            self.resultsDict['ConditionalSurvivalForestModel'] = pysurv_eval(model)
            print('Model has been evaluated')
            survival = model.predict_survival(X_tt)
            print('Predicted Survival Function')
            risk = model.predict_risk(X_tt)
            print('Predicted Risk')
            hazard = model.predict_hazard(X_tt)
            print('Predicted Hazard Function')
            self.predictionsDict['ConditionalSurvivalForestModel'] = [survival, risk, hazard]
            self.modelsDict['ConditionalSurvivalForestModel'] = model

    def pysurvExp(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        # with joblib.parallel_backend('dask'):
        model = ExponentialModel()
        if winner:
            return model
        print('ExponentialModel  running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train)
            self.resultsDict['ExponentialModel'] = pysurv_eval(model)
            survival = model.predict_survival(X_tt)
            risk = model.predict_risk(X_tt)
            hazard = model.predict_hazard(X_tt)
            self.predictionsDict['ExponentialModel'] = [survival, risk, hazard]
            self.modelsDict['ExponentialModel'] = model

    def pysurvLogistic(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        # with joblib.parallel_backend('dask'):
        model = LogLogisticModel()  # lr = 1e-4 by default; exception handling in modeller
        if winner:
            return model
        print('LogLogisticModel running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train)
            self.resultsDict['LogLogisticModel'] = pysurv_eval(model)
            survival = model.predict_survival(X_tt)
            risk = model.predict_risk(X_tt)
            hazard = model.predict_hazard(X_tt)
            self.predictionsDict['LogLogisticModel'] = [survival, risk, hazard]
            self.modelsDict['LogLogisticModel'] = model

    def pysurvWeibull(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        model = WeibullModel()
        if winner:
            return model
        print('WeibullModel running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train)
            self.resultsDict['WeibullModel'] = pysurv_eval(model)
            survival = model.predict_survival(X_tt)
            risk = model.predict_risk(X_tt)
            hazard = model.predict_hazard(X_tt)
            self.predictionsDict['WeibullModel'] = [survival, risk, hazard]
            self.modelsDict['WeibullModel'] = model

    def pysurvNormal(self, winner=False):
        T_train = self.T_train
        E_train = self.E_train
        X_t = self.X_t
        X_tt = self.X_tt
        # with joblib.parallel_backend('dask'):
        model = LogNormalModel()
        if winner:
            return model
        print('LogNormalModel running..')
        with joblib.parallel_backend('dask'):
            model.fit(X_t, T_train, E_train, lr=1e-6)
            self.resultsDict['LogNormalModel'] = pysurv_eval(model)
            survival = model.predict_survival(X_tt)
            risk = model.predict_risk(X_tt)
            hazard = model.predict_hazard(X_tt)
            self.predictionsDict['LogNormalModel'] = [survival, risk, hazard]
            self.modelsDict['LogNormalModel'] = model

    def modeller(self):
        with joblib.parallel_backend('dask'):
            current = time.time()
            try:
                self.pysurvNormal()
            except:
                pass
            finally:
                try:
                    self.pysurvLogistic()
                except:
                    pass
                finally:
                    try:
                        self.pysurvWeibull()
                    except:
                        pass
                    finally:
                        # try:
                        #     self.condForest()
                        # except:
                        #     pass
                        # finally:
                        #     try:
                        #         self.randomForest()
                        #     except:
                        #         pass
                        #     finally:
                        try:
                            self.pysurvExp()
                        except:
                            pass
                        finally:
                            print(f'Total Modelling Time Taken : {time.time() - current}')

    # def modeller(self):
    #     current = time.time()
    #     self.pysurvNormal()
    #     self.pysurvLogistic()
    #     self.pysurvWeibull()
    #     # self.condForest()
    #     # self.randomForest()
    #     self.pysurvExp()
    #     print(f'Total Modelling Time Taken : {time.time() - current}')

    def getWinnerModel(self, winnerName):
        with joblib.parallel_backend('dask'):
            switcher = {
                'LogNormalModel': self.pysurvNormal(winner=True),
                'LogLogisticModel': self.pysurvLogistic(winner=True),
                'WeibullModel': self.pysurvWeibull(winner=True),
                # 'ConditionalSurvivalForestModel': self.condForest(winner=True),
                # 'RandomSurvivalForestModel': self.randomForest(winner=True),
                'ExponentialModel': self.pysurvExp(winner=True)
            }
            return switcher[winnerName]

    def lead(self, model_instance_):
        X_tt = self.X_tt
        survival = model_instance_.predict_survival(X_tt)
        risk = model_instance_.predict_risk(X_tt)
        hazard = model_instance_.predict_hazard(X_tt)
        # density = model_instance_.predict_density(X_tt)
        # cdf = model_instance_.predict_cdf(X_tt)
        predictions = [survival, risk, hazard]
        return predictions

    def scoring_lead(self, model_instance_, X_tt):
        survival = model_instance_.predict_survival(X_tt)
        risk = model_instance_.predict_risk(X_tt)
        hazard = model_instance_.predict_hazard(X_tt)
        # density = model_instance_.predict_density(X_tt)
        # cdf = model_instance_.predict_cdf(X_tt)
        predictions = [survival, risk, hazard]
        return predictions

    def scoring(self, model_ins):
        T_test = self.T_test
        E_test = self.E_test
        X_tt = self.X_tt
        X_t = self.X_t
        T_train = self.T_train
        E_train = self.E_train
        tt_mean = T_test.mean()
        t_mean = T_train.mean()
        print('Undergoing training for highest scoring model')
        with joblib.parallel_backend('dask'):
            ibs = integrated_brier_score(model_ins, X_t, T_train, E_train, t_max=t_mean, figure_size=(20, 6.5))
            print('IBS for the training data is: {:.2f}'.format(ibs))
            ibs1 = integrated_brier_score(model_ins, X_tt, T_test, E_test, t_max=tt_mean, figure_size=(20, 6.5))
            print('IBS for the test data is: {:.2f}'.format(ibs1))
            # survival = model_ins.predict_survival(X_tt)
            # risk = model_ins.predict_risk(X_tt)
            # hazard = model_ins.predict_hazard(X_tt)
            # density = model.predict_density(X_tt)
            # cdf = model.predict_cdf(X_tt)
            # predictions = [survival, risk, hazard]
            # return predictions
