import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from plots import plot_aic_, pysurv_winner, plot_rmst, plot_cdf, winnerPlots
from random import randint
from IPython.display import display_html, display
from itertools import chain, cycle
from pysurv import pysurv_Modelling
from sklearn.feature_selection import f_classif, SelectKBest
import joblib
from dask.distributed import progress, Client
from univariate_modelling import Uni_Modelling

client = Client()

seed = 42

np.random.seed(seed)
np.finfo(np.float64)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


def targetEngineering(df, target):
    """ Creates target feature from dataframe """
    y = df[target]
    return df, y


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def findTrainSize(df):
    df = df.sample(frac=1)
    if len(df) > 100000:
        return 100000, 40000
    else:
        return int(len(df) * 0.75), int(len(df) * 0.25)


# For univariate modelling with Left/Right/Uncensored data
def testPlot_uni(predictionsDict={}):
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        yhat = value
        plt.plot(yhat, color=cmap(randint(0, 50)), label=f'{key}')
        plt.legend()
    plt.show()


def createframe_right(resultsDict, predictionsDict,
                      resultsDict_, predictionsDict_):
    # Pysurvival
    result_df = pd.DataFrame(resultsDict_, index=['AIC'])
    result_df = result_df.loc['AIC'].sort_values(ascending=True)

    # Univariate
    result_df_ = pd.DataFrame(resultsDict, index=['AIC'])
    result_df_ = result_df_.loc['AIC'].sort_values(ascending=True)

    result_Df = pd.concat([result_df, result_df_], axis=0)
    # Merged
    result_Df = result_Df.to_frame('AIC')
    result_Df = result_Df.sort_values(['AIC'], ascending=True)
    # print(type(result_Df))
    # print(result_Df.index)
    print(result_Df)
    result = result_Df.index[0]
    print(f"Winner model is {result}")

    if result in ['LogLogisticModel', 'LogNormalModel', 'ExponentialModel', 'WeibullModel']:
        pysurv_winner(predictionsDict_, result)
    else:
        testPlot_uni({result: predictionsDict[result][0]})
    return result


def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h5>{title}</h5>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


# Feature Selection
def featureSelection(df, target, dur, censoring_type):
    if censoring_type != 'INTERVAL' and df.shape[1] > 25:
        X = df.drop([target, dur['indicator']], axis=1)
        time_events = df[dur['indicator']]
        y = df[target]
        fs = SelectKBest(score_func=f_classif, k=15)
        X_selected = fs.fit_transform(X, y)
        X_selected = pd.DataFrame(X_selected)
        df_ = pd.concat([X_selected, y], axis=1)
        _df = pd.concat([df_, time_events], axis=1)
        return _df
    elif censoring_type == 'INTERVAL' and df.shape[1] > 25:
        X = df.drop([target, dur['indicator']['lower'], dur['indicator']['upper']], axis=1)
        y = df[target]
        time_events = df[[dur['indicator']['lower'], dur['indicator']['upper']]]
        fs = SelectKBest(score_func=f_classif, k=15)
        X_selected = fs.fit_transform(X, y)
        X_selected = pd.DataFrame(X_selected)
        df_ = pd.concat([X_selected, y], axis=1)
        _df = pd.concat([df_, time_events], axis=1)
        return _df


def findTrainSize_uni(df):
    df = df.sample(frac=1)
    if len(df) >= 300000:
        return 300000, 50000
    else:
        return int(len(df) * 0.75), int(len(df) * 0.25)


def right_cens(df, target, resultsDict, predictionsDict, dur, modelsDict, resultsDict_, predictionsDict_,
               modelsDict_):
    print("Length of data", len(df))
    train_size, test_size = findTrainSize_uni(df)
    df_training = df[:train_size]
    df_test = df[-1 * test_size:]

    df_training, E_train = targetEngineering(df_training, target=target)
    T_train = df_training[dur['indicator']]
    print(df_training[[dur['indicator'], target]].head(5), '\n')
    df_test, E_test = targetEngineering(df_test, target=target)
    T_test = df_test[dur['indicator']]
    t_mean = T_train.mean()
    # Univariate Analysis training
    print('\nModelling starts!')
    modelling_obj = Uni_Modelling(T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict)
    modelling_obj.modeller()

    # Multivariate Analysis model training
    train_size, test_size = findTrainSize(df)
    df_training = df[:train_size]
    df_test = df[-1 * test_size:]
    E_train, E_test = df_training[target], df_test[target]
    T_train, T_test = df_training[dur['indicator']], df_test[dur['indicator']]
    df_training = df_training.drop([target, dur['indicator']], axis=1)
    df_test = df_test.drop([target, dur['indicator']], axis=1)
    modelling_obj_ = pysurv_Modelling(df_training, df_test, resultsDict_, predictionsDict_, dur['indicator'],
                                      target, modelsDict_, E_test, T_test, E_train, T_train)
    modelling_obj_.modeller()

    # Model training outcomes
    plot_aic_(resultsDict, resultsDict_)
    plot_rmst(modelsDict, t_mean)
    plot_cdf(modelsDict)

    print('\n')
    winnerModelName = createframe_right(resultsDict, predictionsDict, resultsDict_, predictionsDict_)
    del df, df_training, df_test, T_test, E_test, T_train, E_train
    if winnerModelName in ['LogLogisticModel', 'LogNormalModel', 'ExponentialModel', 'WeibullModel']:
        return winnerModelName, modelsDict_[winnerModelName]
    else:
        return winnerModelName, modelsDict[winnerModelName]


def winnerCens(df, target, dur, winnerModelName, model_instance_):
    if winnerModelName not in ['LogLogisticModel', 'LogNormalModel', 'ExponentialModel', 'WeibullModel']:
        df, E = targetEngineering(df, target)
        # print(X.head(5))
        T = df[dur['indicator']]
        train_size, test_size = findTrainSize_uni(df)
        df_test = df[-1 * test_size:]
        T_test = df_test[dur['indicator']]
        # winner_obj = Uni_Modelling(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        # winnerPlots(winnerModelName, winnerModel)
        scoring_obj = Uni_Modelling(T, T_test, E, pd.DataFrame(), None, None, None)
        winnerPredictionsList = scoring_obj.get_lead(model_instance_)

        if winnerModelName not in ['NelsonAalenFitter', 'KaplanMeierFitter']:
            print(model_instance_.print_summary(decimals=4), '\n')
        winnerPlots(winnerModelName, model_instance_)
        for predictions in winnerPredictionsList:
            try:
                predictions = predictions.to_frame()
            except: pass
            predictions.index.name = 'Point of Time'
            predictions.columns = [f'{winnerModelName} Estimate']
        winnerPredictionsList[0].to_csv('Probability_prediction.csv', index=False)
        winnerPredictionsList[1].to_csv('Survival_prediction.csv', index=False)
        winnerPredictionsList[2].to_csv('Hazard_prediction.csv', index=False)
        return

    else:
        train_size, test_size = findTrainSize(df)
        df_training = df[:train_size]
        df_test = df[-1 * test_size:]
        E_train, E_test = df_training[target], df_test[target]
        T_train, T_test = df_training[dur['indicator']], df_test[dur['indicator']]
        df_training = df_training.drop([target, dur['indicator']], axis=1)
        df_test = df_test.drop([target, dur['indicator']], axis=1)
        # winner_obj = pysurv_Modelling(df_training, df_test, None, None, dur['indicator'], target, None,
        #                               E_test, T_test, E_train, T_train)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        # winnerModel is the untrained model instance
        print("Multivariate Model Scoring Running...")
        scoring_obj = pysurv_Modelling(df_training, df_test, None, None, dur['indicator'], target, None,
                                       E_test, T_test, E_train, T_train)
        winnerPredictionsList = scoring_obj.lead(model_instance_)
        Tt = pd.DataFrame(T_test)
        for i in range(len(winnerPredictionsList)):
            winnerPredictionsList[i] = pd.DataFrame(winnerPredictionsList[i])
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], Tt], axis=1)
            # predictions = predictions.to_frame()
        winnerPredictionsList[1].to_csv('Risk_prediction.csv', index=False)
        winnerPredictionsList[0].to_csv('Survival_prediction.csv', index=False)
        winnerPredictionsList[2].to_csv('Hazard_prediction.csv', index=False)
        return
