from userInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
from plots import plot_cindex_, plot_cdf, plot_rmst, winnerPlots_interval, plot_rmst_interval, winnerPlots, plot_aic
from univariate_modelling import Uni_Modelling, Uni_Modelling_left
from interval_modelling import Uni_interval
from random import randint
from scipy import stats
from IPython.display import display
import seaborn as sns

seed = 42

np.finfo(np.float64)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


# You want to call this script separately so that we take in a large dataset input by the user and push
# it out of memory after extracting the two columns that we need


# Exploratory Data Analysis
def dataExploration(df, dict_, censor):
    # with joblib.parallel_backend('dask'):
    if len(df) < 2000:
        if censor != 'INTERVAL':
            values = df[dict_['indicator']]
            plt.figure(figsize=(15, 5))
            plt.plot(values)
            plt.title('Duration column')
            plt.show()
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(df[dict_['indicator']['lower']])
            plt.plot(df[dict_['indicator']['upper']])
            plt.title('Duration column')
            plt.show()
    elif 2000 <= len(df) < 10000:
        dis = int(0.25 * len(df))
        _df = df.head(dis)
        df_ = df.tail(dis)
        if censor != 'INTERVAL':
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(211)
            values = _df[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the starting')
            fig.add_subplot(212)
            values = df_[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the end')
            plt.show()
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(_df[dict_['indicator']['lower']])
            plt.plot(_df[dict_['indicator']['upper']])
            plt.title('Duration column')
            plt.show()
    elif 10000 <= len(df) < 50000:
        dis = int(0.002 * len(df))
        _df = df.head(dis)
        df_ = df.tail(dis)
        DF = df.iloc[int(0.499 * len(df)):int(0.501 * len(df)), :]
        if censor != 'INTERVAL':
            fig = plt.figure(figsize=(15, 15))
            fig.add_subplot(311)
            values = _df[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the starting')
            fig.add_subplot(312)
            values = DF[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the middle')
            fig.add_subplot(313)
            values = df_[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the end')
            plt.show()
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(_df[dict_['indicator']['lower']])
            plt.plot(_df[dict_['indicator']['upper']])
            plt.title('Duration column')
            plt.show()
    elif len(df) >= 50000:
        dis = int(0.0004 * len(df))
        _df = df.head(dis)
        df_ = df.tail(dis)
        DF = df.iloc[int(0.4998 * len(df)):int(0.5002 * len(df)), :]
        if censor != 'INTERVAL':
            fig = plt.figure(figsize=(15, 15))
            fig.add_subplot(311)
            values = _df[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the starting')
            fig.add_subplot(312)
            values = DF[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the middle')
            fig.add_subplot(313)
            values = df_[dict_['indicator']]
            plt.plot(values)
            plt.title('Duration column from the end')
            plt.show()
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(_df[dict_['indicator']['lower']])
            plt.plot(_df[dict_['indicator']['upper']])
            plt.title('Duration column')
            plt.show()


def remove_outliers(df):
    z = np.abs(stats.zscore(df))
    new_df = df[(z < 3).all(axis=1)]
    return new_df


# Is original series normally distributed?
# Is the std deviation normally distributed?
def normalityPlots(series):
    fig = plt.figure(figsize=(8, 8))
    sns.distplot(series, hist=True, kde=True)


def targetDistribution(df, target):
    fig = plt.figure(figsize=(8, 8))
    sns.countplot(df[target])
    plt.xlabel('Event target data points')
    plt.ylabel('Number of occurences')
    plt.title('Distribution of event target column values')
    plt.show()


def findTrainSize(df):
    df = df.sample(frac=1)
    return int(len(df) * 0.75), int(len(df) * 0.25)


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
    fig = plt.figure(figsize=(8, 8))
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        plt.plot(value[0], color=cmap(randint(0, 50)), label=f'{key}')
        plt.legend()
    plt.show()


def testPlot_uni_interval(predictionsDict={}):
    fig = plt.figure(figsize=(8, 8))
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        plt.plot(value[0], color=cmap(randint(0, 50)), label=f'{key}_lower')
        plt.plot(value[1], color=cmap(randint(0, 50)), label=f'{key}_upper')
        plt.legend()
    plt.show()


def createResultFrame_uni(resultsDict, predictionsDict):
    result_df = pd.DataFrame(resultsDict, index=['AIC'])
    result_df = result_df.loc['AIC'].sort_values(ascending=True)
    print("\nModel Information Table [sorted by AIC score]\n")
    display(result_df)
    result = result_df.index[0]
    print(f"Winner model is {result}")
    testPlot_uni({result: predictionsDict[result]})
    return result


def createResultFrame_uni_interval(resultsDict, predictionsDict):
    result_df = pd.DataFrame(resultsDict, index=['AIC'])
    result_df = result_df.loc['AIC'].sort_values(ascending=True)
    print("\nModel Information Table [sorted by AIC score]\n")
    display(result_df)
    result = result_df.index[0]
    print(f"Winner model is {result}")
    testPlot_uni_interval({result: predictionsDict[result]})
    return result


def modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censor, dur):
    """
    :param df: The complete dataframe
    :param target: events column name
    :param resultsDict: Dictionary of _concordance_index
    :param predictionsDict: Dictionary of yhat
    :param modelsDict: Dictionary of model instances
    :param censor: Censoring type - (Right/Uncensored/Left)
    :param dur: Dictionary containing column names of duation columns
    :return: Winner model
    """
    # X = df.values
    print("Length of data", len(df))
    train_size, test_size = findTrainSize(df)
    df_training = df[:train_size]
    df_test = df[-1 * test_size:]

    df_training, E_train = targetEngineering(df_training, target=target)
    T_train = df_training[dur['indicator']]
    print(df_training[[dur['indicator'], target]].head(5), '\n')
    df_test, E_test = targetEngineering(df_test, target=target)
    T_test = df_test[dur['indicator']]
    t_mean = T_train.mean()
    print('\nModelling starts!')

    if censor in ['RIGHT', 'UNCENSORED']:
        modelling_obj = Uni_Modelling(T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict)
        modelling_obj.modeller()
        plot_rmst(modelsDict, t_mean)
        plot_cdf(modelsDict)
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_aic(resultsDict)
        winnerModelName = createResultFrame_uni(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName, modelsDict[winnerModelName]
    elif censor == 'LEFT':
        modelling_obj = Uni_Modelling_left(T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict)
        modelling_obj.modeller()
        plot_rmst(modelsDict, t_mean)
        plot_cdf(modelsDict)
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_aic(resultsDict)
        print('\n')
        winnerModelName = createResultFrame_uni(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName, modelsDict[winnerModelName]


def modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur):
    """
    :param df: The complete dataframe
    :param target: events column name
    :param resultsDict: Dictionary of _concordance_index
    :param predictionsDict: Dictionary of yhat
    :param modelsDict: Dictionary of model instances
    :param dur: Dictionary containing column names of duation columns
    :return: Winner model
    """
    # X = df.values
    print("Length of data", len(df))
    train_size, test_size = findTrainSize(df)
    df_training = df[:train_size]
    df_test = df[-1 * test_size:]
    print("Train Size", len(df_training))
    print("Test Size", len(df_test))
    print(
        f"We have {len(df_training)} rows of training data and {len(df_test)} rows of validation data ")

    df_training, E_train = targetEngineering(df_training, target=target)
    upper_t = df_training[dur['indicator']['upper']]
    lower_t = df_training[dur['indicator']['lower']]

    # T_train = df_training[dur['indicator']]
    # print(df_training[[lower_t, upper_t, target]].head(5), '\n')
    df_test, E_test = targetEngineering(df_test, target=target)
    upper_tt = df_test[dur['indicator']['upper']]
    lower_tt = df_test[dur['indicator']['lower']]
    ut_mean = upper_t.mean()
    lt_mean = lower_t.mean()

    # try:
    modelling_obj = Uni_interval(lower_t, upper_t, lower_tt,
                                 upper_tt, E_train, E_test, resultsDict, predictionsDict, modelsDict)
    modelling_obj.modeller()
    plot_rmst_interval(modelsDict, lt_mean, ut_mean)
    # plot_cdf(modelsDict)
    # print('\nPrediction estimate for all models')
    # testPlot_uni(predictionsDict)
    plot_aic(resultsDict)
    print('\n')
    winnerModelName = createResultFrame_uni_interval(resultsDict, predictionsDict)
    del df, df_training, df_test, E_train, E_test, lower_t, lower_tt, upper_t, upper_tt
    return winnerModelName, modelsDict[winnerModelName]
    # except:
    #     print('The data submitted by the user is not Interval Censored. Please Check Again.\n')
    #     return None


def winnerModelTrainer_uni(df, target, winnerModelName, dur, censor, winnerModel_ins, orig, mode, target_availability):
    # print("Inside Winner Trainer",len(df))
    if target_availability is None and mode != 'default':
        df[target] = 0
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    T = df[dur['indicator']]
    train_size, test_size = findTrainSize(df)
    df_test = df[-1 * test_size:]
    T_test = df_test[dur['indicator']]
    T_s = df[dur['indicator']]
    print('Shape of input for scoring - ', len(T_s))
    if censor == 'LEFT':
        # Getting winner model object
        # winner_obj = Uni_Modelling_left(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        scoring_obj = Uni_Modelling_left(T, T_test, E, pd.DataFrame(), None, None, None)
        if mode == 'default':
            winnerPredictionsList = scoring_obj.get_lead(winnerModel_ins)
        elif mode == 'scoring':
            winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, T_s)
        print("Length of winner predictions", len(winnerPredictionsList))

    elif censor in ['RIGHT', 'UNCENSORED']:
        # Getting winner model object
        # winner_obj = Uni_Modelling(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        # winnerPlots(winnerModelName, winnerModel)
        scoring_obj = Uni_Modelling(T, T_test, E, pd.DataFrame(), None, None, None)
        if mode == 'default':
            winnerPredictionsList = scoring_obj.get_lead(winnerModel_ins)
        elif mode == 'scoring':
            winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, T_s)
        print("Length of winner predictions", len(winnerPredictionsList))

    if winnerModelName not in ['NelsonAalenFitter', 'KaplanMeierFitter']:
        print(winnerModel_ins.print_summary(decimals=4), '\n')
    if mode == 'default':
        winnerPlots(winnerModelName, winnerModel_ins)

    for predictions in winnerPredictionsList:
        try:
            predictions = predictions.to_frame()
        except:
            pass
    for i in range(3):
        winnerPredictionsList[i] = winnerPredictionsList[i].reset_index(drop=False)
        winnerPredictionsList[i].rename(columns={'index': 'Point of Time'}, inplace=True)
    # print(winnerPredictionsList[1], '\n\n', winnerPredictionsList[2])
    if mode != 'default':
        for i in range(3):
            winnerPredictionsList[i] = winnerPredictionsList[i].iloc[:, :2]
    if target_availability is not None and mode == 'scoring':
        for i in range(3):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], df[target]], axis=1)
    # print(winnerPredictionsList[1], '\n\n', winnerPredictionsList[2])
    if mode=='default':
        winnerPredictionsList[0].columns = ['Point of Time', f'{winnerModelName} Estimate']
        winnerPredictionsList[1].columns = ['Point of Time', f'{winnerModelName} Estimate']
        winnerPredictionsList[2].columns = ['Point of Time', f'{winnerModelName} Estimate']
        for i in range(3):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
            # winnerPredictionsList[i].drop([dur['indicator']], axis=1)
            winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
        winnerPredictionsList[0].to_csv('default/Probability_prediction.csv', index=False)
        winnerPredictionsList[1].to_csv('default/Survival_prediction.csv', index=False)
        winnerPredictionsList[2].to_csv('default/Hazard_prediction.csv', index=False)
        return
    elif mode=='scoring' and target_availability is not None:
        for i in range(3):
            print(winnerPredictionsList[i].shape, '\n')
        winnerPredictionsList[0].columns = ['Point of Time', f'{winnerModelName} Estimate', 'target']
        winnerPredictionsList[1].columns = ['Point of Time', f'{winnerModelName} Estimate', 'target']
        winnerPredictionsList[2].columns = ['Point of Time', f'{winnerModelName} Estimate', 'target']
        # df = df.rename(columns={dur['indicator']:'Point of Time'})
        for i in range(3):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
            # winnerPredictionsList[i].drop([dur['indicator']], axis=1)
            winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    if mode == 'scoring':
        winnerPredictionsList[0].to_csv('unknown/Probability_prediction.csv', index=False)
        winnerPredictionsList[1].to_csv('unknown/Survival_prediction.csv', index=False)
        winnerPredictionsList[2].to_csv('unknown/Hazard_prediction.csv', index=False)
        return


def winnerModelTrainer_uni_interval(df, target, winnerModelName, winnerModel_ins, dur,orig, mode, target_availability):
    # print("Inside Winner Trainer",len(df))
    if target_availability is None and mode != 'default':
        df[target] = 0
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    upper = df[dur['indicator']['upper']]
    lower = df[dur['indicator']['lower']]
    train_size, test_size = findTrainSize(df)
    df_test = df[-1 * test_size:]
    upper_tt = df_test[dur['indicator']['upper']]
    lower_tt = df_test[dur['indicator']['lower']]
    up = df[dur['indicator']['upper']]
    lo = df[dur['indicator']['lower']]
    # Getting winner model object
    # winner_obj = Uni_interval(lower, upper, pd.DataFrame(),
    #                           pd.DataFrame(), E, pd.DataFrame(), None, None, None)
    # winnerModel = winner_obj.getWinnerModel(winnerModelName)
    print("Univariate Model Scoring Running...")
    scoring_obj = Uni_interval(lower, upper, lower_tt, upper_tt, E, pd.DataFrame(), None, None, None)
    if mode == 'default':
        winnerPredictionsList = scoring_obj.get_lead(winnerModel_ins)
    elif mode == 'scoring':
        winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, up, lo)
    print("Length of winner predictions", len(winnerPredictionsList))
    print(winnerModel_ins.print_summary(decimals=4), '\n')
    df = df.sort_values(by=[dur['indicator']['lower']], ascending=True)
    ttt = df[dur['indicator']['upper']]
    # print("Length of winner predictions",len(winnerPredictionsList))
    if mode == 'default':
        winnerPlots_interval(winnerModelName, winnerModel_ins)

    for predictions in winnerPredictionsList:
        try:
            predictions = predictions.to_frame()
        except:
            pass

    for i in range(4):
        # winnerPredictionsList[i] = winnerPredictionsList[i].to_frame()
        winnerPredictionsList[i] = winnerPredictionsList[i].reset_index(drop=False)
        winnerPredictionsList[i].rename(columns={'index': 'Point of Time'}, inplace=True)
    for i in range(2, 4):
        winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], ttt], axis=1)
        winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    if mode == 'scoring':
        for i in range(2, 4):
            winnerPredictionsList[i] = winnerPredictionsList[i].drop(['prob'], axis=1)
    for i in range(4):
        winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    if mode == 'scoring' and target_availability is not None:
        for i in range(3):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], df[target]], axis=1)
        # winnerPredictionsList[i] = winnerPredictionsList[i].drop(['prob'], axis=1)
    # print(winnerPredictionsList[1], '\n', winnerPredictionsList[2], '\n', winnerPredictionsList[3])
    if mode == 'default':
        winnerPredictionsList[0].columns = ['Point of Time', 'Probability Estimate in lower bound']
        winnerPredictionsList[1].columns = ['Point of Time', 'Probability Estimate in upper bound']
        winnerPredictionsList[2].columns = ['Lower Bound', 'Survival Estimate', 'Upper Bound']
        winnerPredictionsList[3].columns = ['Lower Bound', 'Hazard Estimate', 'Upper Bound']
        for i in range(4):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
            winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
        winnerPredictionsList[0].to_csv('default/Probability_future_lower.csv', index=False)
        winnerPredictionsList[1].to_csv('default/Probability_future_upper.csv', index=False)
        winnerPredictionsList[2].to_csv('default/Survival_future.csv', index=False)
        winnerPredictionsList[3].to_csv('default/Hazard_future.csv', index=False)
        return
    elif mode == 'scoring' and target_availability is not None:
        winnerPredictionsList[0].columns = ['Point of Time', 'Probability Estimate in lower bound', 'target']
        winnerPredictionsList[1].columns = ['Point of Time', 'Probability Estimate in upper bound', 'target']
        winnerPredictionsList[2].columns = ['Lower Bound', 'Survival Estimate', 'Upper Bound', 'target']
        winnerPredictionsList[3].columns = ['Lower Bound', 'Hazard Estimate', 'Upper Bound', 'target']
        for i in range(4):
            winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
            winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    if mode == 'scoring':
        winnerPredictionsList[0].to_csv('unknown/Probability_future_lower.csv', index=False)
        winnerPredictionsList[1].to_csv('unknown/Probability_future_upper.csv', index=False)
        winnerPredictionsList[2].to_csv('unknown/Survival_future.csv', index=False)
        winnerPredictionsList[3].to_csv('unknown/Hazard_future.csv', index=False)
        return


def scoring_prediction(df, target, winnerModelName, dur, censor, winnerModel_ins, orig):
    # print("Inside Winner Trainer",len(df))
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    T = df[dur['indicator']]
    train_size, test_size = findTrainSize(df)
    df_test = df[-1 * test_size:]
    T_test = df_test[dur['indicator']]
    T_s = df[dur['indicator']]
    if censor == 'LEFT':
        # Getting winner model object
        # winner_obj = Uni_Modelling_left(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        scoring_obj = Uni_Modelling_left(T, T_test, E, pd.DataFrame(), None, None, None)
        winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, T_s)
        # print("Length of winner predictions", len(winnerPredictionsList))

    elif censor in ['RIGHT', 'UNCENSORED']:
        # Getting winner model object
        # winner_obj = Uni_Modelling(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        # winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        # winnerPlots(winnerModelName, winnerModel)
        scoring_obj = Uni_Modelling(T, T_test, E, pd.DataFrame(), None, None, None)
        winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, T_s)
        # print("Length of winner predictions", len(winnerPredictionsList))

    for predictions in winnerPredictionsList:
        try:
            predictions = predictions.to_frame()
        except:
            pass
    for i in range(3):
        # winnerPredictionsList[i] = winnerPredictionsList[i].to_frame()
        winnerPredictionsList[i] = winnerPredictionsList[i].reset_index(drop=False)
        winnerPredictionsList[i].rename(columns={'index': 'Point of Time'}, inplace=True)
        # winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], ttt], axis=1)
    winnerPredictionsList[0].columns = ['Point of Time', 'Probability Estimate']
    winnerPredictionsList[1].columns = ['Point of Time', 'Survival Estimate']
    winnerPredictionsList[2].columns = ['Point of Time', 'Hazard Estimate']
    winnerPredictionsList[0] = pd.concat([winnerPredictionsList[0], orig], axis=1)
    winnerPredictionsList[1] = pd.concat([winnerPredictionsList[1], orig], axis=1)
    winnerPredictionsList[2] = pd.concat([winnerPredictionsList[2], orig], axis=1)
    for i in range(3):
        # winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
        # winnerPredictionsList[i].drop([dur['indicator']], axis=1)
        winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    winnerPredictionsList[0].to_csv('scoring/Probability_future.csv', index=False)
    winnerPredictionsList[1].to_csv('scoring/Survival_future.csv', index=False)
    winnerPredictionsList[2].to_csv('scoring/Hazard_future.csv', index=False)
    return


def scoring_pred_interval(df, target, winnerModelName, winnerModel_ins, dur, orig):
    # print("Inside Winner Trainer",len(df))
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    upper = df[dur['indicator']['upper']]
    lower = df[dur['indicator']['lower']]
    train_size, test_size = findTrainSize(df)
    df_test = df[-1 * test_size:]
    upper_tt = df_test[dur['indicator']['upper']]
    lower_tt = df_test[dur['indicator']['lower']]
    up = df[dur['indicator']['upper']]
    lo = df[dur['indicator']['lower']]
    # Getting winner model object
    # winner_obj = Uni_interval(lower, upper, pd.DataFrame(),
    #                           pd.DataFrame(), E, pd.DataFrame(), None, None, None)
    # winnerModel = winner_obj.getWinnerModel(winnerModelName)
    print("Univariate Model Scoring Running...")
    scoring_obj = Uni_interval(lower, upper, lower_tt, upper_tt, E, pd.DataFrame(), None, None, None)
    winnerPredictionsList = scoring_obj.get_lead_sc(winnerModel_ins, up, lo)

    df = df.sort_values(by=[dur['indicator']['lower']], ascending=True)
    ttt = df[dur['indicator']['upper']]
    for predictions in winnerPredictionsList:
        try:
            predictions = predictions.to_frame()
        except:
            pass
        predictions.index.name = 'Point of Time'
        predictions.columns = [f'{winnerModelName} Estimate']
    for i in range(4):
        # winnerPredictionsList[i] = winnerPredictionsList[i].to_frame()
        winnerPredictionsList[i] = winnerPredictionsList[i].reset_index(drop=False)
        winnerPredictionsList[i].rename(columns={'index': 'Point of Time'}, inplace=True)
    for i in range(2, 4):
        winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], ttt.reindex(winnerPredictionsList[i].index)],
                                             axis=1)
        winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    # print(winnerPredictionsList[1], '\n', winnerPredictionsList[2], '\n', winnerPredictionsList[3])
    winnerPredictionsList[0].columns = ['Point of Time', 'Probability Estimate in lower bound']
    winnerPredictionsList[1].columns = ['Point of Time', 'Probability Estimate in upper bound']
    winnerPredictionsList[2].columns = ['Lower Bound', 'Survival Estimate', 'Upper Bound']
    winnerPredictionsList[3].columns = ['Lower Bound', 'Hazard Estimate', 'Upper Bound']
    for i in range(4):
        winnerPredictionsList[i] = pd.concat([winnerPredictionsList[i], orig], axis=1)
        winnerPredictionsList[i] = winnerPredictionsList[i].dropna()
    winnerPredictionsList[0].to_csv('scoring/Probability_future_lower.csv', index=False)
    winnerPredictionsList[1].to_csv('scoring/Probability_future_upper.csv', index=False)
    winnerPredictionsList[2].to_csv('scoring/Survival_future.csv', index=False)
    winnerPredictionsList[3].to_csv('scoring/Hazard_future.csv', index=False)
    return
