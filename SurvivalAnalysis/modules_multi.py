from lifelines.plotting import plot_interval_censored_lifetimes, plot_lifetimes
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from tabulate import tabulate
from plots import plot_simple_cindex, plot_aic, winner_multi, plot_pysurv_cindex, pysurv_winner
from multivariate_modelling import Multi_Modelling, Multi_Modelling_left
from interval_modelling import Multi_interval
from random import randint
from IPython.display import display_html, display
from itertools import chain, cycle
from pysurv import pysurv_Modelling
from pysurvival.utils.display import integrated_brier_score


seed = 42
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


def lifetimes(time_events, actual, censor,
              lower_bound_t, upper_bound_t):
    if censor != 'Interval':
        fig = plt.figure(figsize=(15, 10))
        plt.title('Lifetime variation from the beginning')
        fig.add_subplot(131)
        ax = plot_lifetimes(time_events[:int(0.01 * (len(time_events)))], actual[:int(0.01 * (len(time_events)))])
        plt.title('Lifetime variation from the middle')
        fig.add_subplot(132)
        ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):], actual[:int(0.51 * (len(time_events)))])
        plt.title('Lifetime variation from the end')
        fig.add_subplot(133)
        ax = plot_lifetimes(time_events[int(0.99 * (len(time_events))):], actual[:int(0.99 * (len(time_events))):])
        plt.show()
    else:
        fig = plt.figure(figsize=(15, 10))
        plt.title('Lifetime variation from the beginning')
        fig.add_subplot(131)
        ax = plot_lifetimes(lower_bound_t[:int(0.01 * (len(lower_bound_t)))],
                            upper_bound_t[:int(0.01 * (len(upper_bound_t)))])
        plt.title('Lifetime variation from the middle')
        fig.add_subplot(132)
        ax = plot_lifetimes(lower_bound_t[int(0.50 * (len(lower_bound_t))):],
                            upper_bound_t[:int(0.51 * (len(upper_bound_t)))])
        plt.title('Lifetime variation from the end')
        fig.add_subplot(133)
        ax = plot_lifetimes(lower_bound_t[int(0.99 * (len(lower_bound_t))):],
                            upper_bound_t[:int(0.99 * (len(upper_bound_t))):])
        plt.show()


def normalityPlots(series):
    fig = plt.figure(figsize=(10, 10))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    series.hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")

    plt.show()


def findTrainSize(df):
    return int(len(df) * 0.90)


def targetEngineering(df, target):
    """ Creates target feature from dataframe """
    y = df[target]
    return df, y


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


# For univariate modelling with Left/Right/Uncensored data
def testPlot_uni(predictionsDict={}):
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        yhat = value
        plt.plot(yhat, color=cmap(randint(0, 50)), label=f'{key}')
        plt.legend()
    plt.show()


def testPlot_uni_interval(predictionsDict={}):
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        yhat = value[0]
        plt.plot(yhat, color=cmap(randint(0, 50)), label=f'{key}')
        plt.legend()
    plt.show()


# Used for Right/Uncensored data
def createResultFrame_multi(resultsDict, predictionsDict):
    result_df = pd.DataFrame(columns=['Model', 'simple_cindex'])
    for model, values in resultsDict.items():
        temp = [model]
        temp.extend(values.values())
        result_df.loc[len(result_df)] = temp
        result_df.sort_values(by=['simple_cindex'], inplace=True, ignore_index=True, ascending=False)
    print("\nModel Information Table [sorted by Concordance Index score]")
    print(tabulate(result_df, headers=[
        'Model', 'simple_cindex'], tablefmt="fancy_grid"))
    result = result_df.iloc[0, 0]
    print(f"Winner model is {result}")
    winner_multi(predictionsDict, result)
    return result


def createResultFrame_pysurv(resultsDict, predictionsDict):
    result_df = pd.DataFrame(columns=['Model', 'concordance_index'])
    for model, values in resultsDict.items():
        temp = [model]
        temp.extend(values.values())
        result_df.loc[len(result_df)] = temp
        result_df.sort_values(by=['concordance_index'], inplace=True, ignore_index=True, ascending=False)
    print("\nModel Information Table [sorted by Concordance Index score]")
    print(tabulate(result_df, headers=[
        'Model', 'concordance_index'], tablefmt="fancy_grid"))
    result = result_df.iloc[0, 0]
    print(f"Winner model is {result}")
    pysurv_winner(predictionsDict, result)
    return result


def createResultFrame_multi_int(resultsDict, predictionsDict):
    result_df = pd.DataFrame(columns=['Model', 'aic'])
    for model, values in resultsDict.items():
        temp = [model]
        temp.extend(values.values())
        result_df.loc[len(result_df)] = temp
        result_df.sort_values(by=['aic'], inplace=True, ignore_index=True, ascending=False)
    print("\nModel Information Table [sorted by AIC score]")
    print(tabulate(result_df, headers=[
        'Model', 'aic'], tablefmt="fancy_grid"))
    result = result_df.iloc[0, 0]
    print(f"Winner model is {result}")
    winner_multi(predictionsDict, result)
    return result


def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h5>{title}</h5>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict, censor, dur):
    """
    :param df: The complete dataframe
    :param target: events column name
    :param resultsDict: Dictionary of _concordance_index
    :param predictionsDict: Dictionary of yhat
    :param modelsDict: Dictionary of model instances
    :param censor: Censoring type - (Right/Uncensored/Left)
    :param dur: Dictionary containing column names of duration columns
    :return: Winner model
    """
    # X = df.values
    print("Length before preprocessing", len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    print("Index is split at", split_date)
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))
    print(
        f"We have {len(df_training)} rows of training data and {len(df_test)} rows of validation data ")

    df_training, E_train = targetEngineering(df_training, target=target)
    T_train = df_training[dur['indicator']]
    print(df_training[[dur['indicator'], target]].head(5), '\n')
    df_test, E_test = targetEngineering(df_test, target=target)
    T_test = df_test[dur['indicator']]

    if censor in ['Right', 'Uncensored']:
        modelling_obj = Multi_Modelling(df_training, df_test, T_train, T_test, E_train, E_test,
                                        resultsDict, predictionsDict, dur['indicator'], target, modelsDict)
        modelling_obj.modeller()
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_simple_cindex(resultsDict)
        winnerModelName = createResultFrame_multi(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName

    elif censor == 'Left':
        modelling_obj = Multi_Modelling_left(df_training, df_test, T_train, T_test, E_train, E_test,
                                             resultsDict, predictionsDict, dur['indicator'], target, modelsDict)
        modelling_obj.modeller()
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_simple_cindex(resultsDict)
        winnerModelName = createResultFrame_multi(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName


def modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur):
    print("Length before preprocessing", len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    print("Index is split at", split_date)
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))
    print(
        f"We have {len(df_training)} rows of training data and {len(df_test)} rows of validation data ")

    df_training, E_train = targetEngineering(df_training, target=target)
    upper_t = dur['indicator']['upper']
    lower_t = dur['indicator']['lower']
    df_test, E_test = targetEngineering(df_test, target=target)
    upper_tt = upper_t
    lower_tt = lower_t

    modelling_obj = Multi_interval(df_training, df_test, lower_t, upper_t, lower_tt,
                                   upper_tt, target, target, resultsDict, predictionsDict)
    modelling_obj.modeller()
    plot_aic(resultsDict)
    winnerModelName = createResultFrame_multi_int(resultsDict, predictionsDict)
    del df, df_training, df_test, E_train, E_test, lower_t, lower_tt, upper_t, upper_tt
    return winnerModelName


def modelling_pysurvival(df, target, resultsDict, predictionsDict, dur, modelsDict):
    print("Length before preprocessing", len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    print("Index is split at", split_date)
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))
    print(
        f"We have {len(df_training)} rows of training data and {len(df_test)} rows of validation data ")

    modelling_obj = pysurv_Modelling(df_training, df_test, resultsDict, predictionsDict, dur['indicator'],
                                     target, modelsDict)
    modelling_obj.modeller()

    plot_pysurv_cindex(resultsDict)
    winnerModelName = createResultFrame_pysurv(resultsDict, predictionsDict)
    del df, df_training, df_test
    return winnerModelName


def winnerModeltrainer_pysurv(df, target, dur, winnerModelName):
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    df_t = df_training.drop([target, dur['indicator']], axis=1)
    E_train = df_training[target]
    T_train = df_training[dur['indicator']]
    t_mean = T_train.mean()
    df_tt = df_test.drop([target, dur['indicator']], axis=1)
    E_test = df_test[target]
    T_test = df_test[dur['indicator']]
    tt_mean = T_test.mean()
    winner_obj = pysurv_Modelling(df_training, pd.DataFrame(), None, None, dur['indicator'], target, None)
    winnerModel = winner_obj.getWinnerModel(winnerModelName)
    ibs = integrated_brier_score(winnerModel, df_t, T_train, E_train, t_max=t_mean, figure_size=(20, 6.5))
    print('IBS for the training data is: {:.2f}'.format(ibs))
    ibs1 = integrated_brier_score(winnerModel, df_tt, T_test, E_test, t_max=tt_mean, figure_size=(20, 6.5))
    print('IBS for the test data is: {:.2f}'.format(ibs1))
    print("Multivariate Model Scoring Running...")
    scoring_obj = pysurv_Modelling(df_training, df_test, None, None, dur['indicator'], target, None)
    winnerPredictionsList = scoring_obj.scoring(winnerModel)
    for i in range(len(winnerPredictionsList)):
        winnerPredictionsList[i] = pd.DataFrame(winnerPredictionsList[i])
    winnerPredictionsList[0].to_csv('survival.csv')
    winnerPredictionsList[1].to_csv('risk.csv')
    winnerPredictionsList[2].to_csv('hazard.csv')
    winnerPredictionsList[3].to_csv('density.csv')
    winnerPredictionsList[3].to_csv('cum_density.csv')


def winnerModelTrainer_multi(df, target, winnerModelName, dur, censor):
    # print("Inside Winner Trainer",len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]

    df_training, E_train = targetEngineering(df_training, target=target)
    T_train = df_training[dur['indicator']]
    df_test, E_test = targetEngineering(df_test, target=target)
    T_test = df_test[dur['indicator']]

    if censor == 'Left':
        # Getting winner model object
        winner_obj = Multi_Modelling_left(df_training, pd.DataFrame(), T_train, pd.DataFrame(), E_train, pd.DataFrame(),
                                          None, None, dur['indicator'], target, None)
        winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Multivariate Model Scoring Running...")
        scoring_obj = Multi_Modelling_left(df_training, df_test, T_train, T_test, E_train, pd.DataFrame(),
                                           None, None, dur['indicator'], target, None)
        winnerPredictionsList = scoring_obj.scoring(winnerModelName, winnerModel)
        print("Length of winner predictions - ", len(winnerPredictionsList))

    elif censor in ['Right', 'Uncensored']:
        # Getting winner model object
        winner_obj = Multi_Modelling(df_training, pd.DataFrame(), T_train, pd.DataFrame(), E_train, pd.DataFrame(),
                                     None, None, dur['indicator'], target, None)
        winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Multivariate Model Scoring Running...")
        # winnerPlots(winnerModelName, winnerModel)
        scoring_obj = Multi_Modelling(df_training, df_test, T_train, T_test, E_train, pd.DataFrame(),
                                      None, None, dur['indicator'], target, None)
        winnerPredictionsList = scoring_obj.scoring(winnerModelName, winnerModel)
        print("Length of winner predictions - ", len(winnerPredictionsList), '\n')

    print("View the Model's Summary")
    print(winnerModel.print_summary(decimals=4), '\n')

    if winnerModelName == 'GeneralizedGammaRegressionFitter':
        winner_multi(winnerPredictionsList, winnerModelName)
        display_side_by_side(winnerPredictionsList[0], winnerPredictionsList[1],
                             titles=['Survival', 'Cumulative Hazard'])
        winnerPredictionsList[0].to_csv('survival.csv')
        winnerPredictionsList[1].to_csv('cum_hazard.csv')

    if len(winnerPredictionsList) == 3:
        winnerPredictionsList[0].to_csv('survival.csv')
        winnerPredictionsList[1].to_csv('cum_hazard.csv')
        winnerPredictionsList[2].to_csv('hazard.csv')

    elif winnerModelName == 'CoxPHFitter':
        winnerPredictionsList[0].to_csv('survival.csv')
        winnerPredictionsList[1].to_csv('cum_hazard.csv')
        winnerPredictionsList[2].to_csv('partial_hazard.csv')
        winnerPredictionsList[3].to_csv('log_hazard.csv')

    return


def winnerModelTrainer_multi_interval(df, target, winnerModelName, dur):
    print("Length before generating lags", len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    print("Index is split at", split_date)
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))
    print(
        f"We have {len(df_training)} rows of training data and {len(df_test)} rows of validation data ")

    df_training, E_train = targetEngineering(df_training, target=target)

    upper_t = dur['indicator']['upper']
    lower_t = dur['indicator']['lower']
    df_test, E_test = targetEngineering(df_test, target=target)
    upper_tt = upper_t
    lower_tt = lower_t

    winner_obj = Multi_interval(df_training, pd.DataFrame(), lower_t, upper_t, None,
                                None, target, None, None, None)
    winnerModel = winner_obj.getWinnerModel(winnerModelName)
    print("Multivariate Model Scoring Running...")
    scoring_obj = Multi_interval(df_training, df_test, lower_t, upper_t, lower_tt, upper_tt,
                                 target, None, None, None)
    winnerPredictionsList = scoring_obj.scoring(winnerModelName, winnerModel)
    print("Length of winner predictions - ", len(winnerPredictionsList))

    print("View the Model's Summary")
    print(winnerModel.print_summary(decimals=4), '\n')
    if len(winnerPredictionsList) == 3:
        winnerPredictionsList[0].to_csv('survival.csv')
        winnerPredictionsList[1].to_csv('cum_hazard.csv')
        winnerPredictionsList[2].to_csv('hazard.csv')

    elif winnerModelName == 'GeneralizedGammaRegressionFitter':
        winnerPredictionsList[0].to_csv('survival.csv')
        winnerPredictionsList[1].to_csv('cum_hazard.csv')

    return
