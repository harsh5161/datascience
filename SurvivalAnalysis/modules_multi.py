from lifelines.plotting import plot_interval_censored_lifetimes, plot_lifetimes
from userInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from tabulate import tabulate
from plots import plot_simple_cindex, plot_aic, winner_multi
from multivariate_modelling import Multi_Modelling, Multi_Modelling_left
from random import randint
from IPython.display import display_html, display
from itertools import chain, cycle

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
        plt.title('Lifetime variation')
        ax = plot_lifetimes(time_events[:int(0.60*(len(time_events)))], actual[:int(0.60*(len(time_events)))])
        plt.show()
    else:
        plt.title('Lifetime Variation')
        ax = plot_interval_censored_lifetimes(lower_bound_t, upper_bound_t)
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
    # Use if-else here
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


def winnerModelTrainer_multi(df, target, winnerModelName, dur, censor):
    # print("Inside Winner Trainer",len(df))
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    print("Index is split at", split_date)
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))

    df_training, E_train = targetEngineering(df_training, target=target)
    T_train = df_training[dur['indicator']]
    print(df_training[[dur['indicator'], target]].head(5), '\n')
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
        winnerPredictionsList = scoring_obj.scoring(winnerModel)
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

    # if winnerModelName == 'GeneralizedGammaRegressionFitter':
    #     winner_multi(winnerPredictionsList, winnerModelName)
    #     display_side_by_side(winnerPredictionsList[0], winnerPredictionsList[1],
    #                          titles=['Survival', 'Cumulative Hazard'])
    #     winnerPredictionsList[0].to_csv('survival.csv')
    #     winnerPredictionsList[1].to_csv('cum_hazard.csv')

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
