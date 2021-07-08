from userInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib as mpl
from tabulate import tabulate
from plots import plot_cindex_, plot_cdf, plot_rmst, winnerPlots_interval, plot_rmst_interval, winnerPlots
from univariate_modelling import Uni_Modelling, Uni_Modelling_left
from interval_modelling import Uni_interval
from lifelines.utils import datetimes_to_durations
from lifelines.plotting import *
from random import randint
from scipy import stats
from IPython.display import display

seed = 42
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


# You want to call this script separately so that we take in a large dataset input by the user and push
# it out of memory after extracting the two columns that we need


def userInputs():
    path = input("Enter the path of the dataset :")
    try:
        df, _ = importFile(path)
        display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None

    key = input('\nEnter optional ID column or enter "None" if no id column is present: ')
    if id in df.columns:
        df = df.drop(key, axis=1)

    dur = {}

    num_cat = input('Are categorical columns present in the dataset? (Y/N): ')
    if num_cat == 'Y':
        print('Please give ONE space after every comma!')
        categorical = [item for item in input("Enter the name of categorical columns separated by comma"
                                              " or None in case there are no categorical columns : ").split(', ')]
        for col in categorical:
            if col in df.columns:
                pass
            else:
                print('Invalid Categorical column name')

        lb = LabelEncoder()
        for col in categorical:
            df[col] = lb.fit_transform(df[col])
    elif num_cat == 'N':
        pass

    # Definition of different kind of censoring types that may be present in the data
    print('\nExplanation of different types of censoring:')
    print('1. Right censoring is when the event of interest happens after the survey timeframe.')
    print('2. Left censoring is when the event of interest has happened before the start of the survey timeframe.')
    print('3. Uncensored is when the event of interest happens within the survey time frame.')
    print('4. Interval censoring is when there is a mix of Right and Left censoring.\n')

    censoring_type = input("\nEnter the type of censoring present in the dataset (Right/Uncensored/Left/Interval) :")
    if censoring_type not in ['Right', 'Uncensored', 'Left', 'Interval']:
        print('Invalid censoring type')

    # Censoring except Interval censoring
    elif censoring_type in ['Uncensored', 'Right', 'Left']:
        cen_input = int(input(
            "\nNumber of duration columns? (1/2) :"))  # Duration can be expressed as a single column as well as 2 columns

        if cen_input == 1:
            indicator = input("Enter the single duration column name: ")
            if indicator in df.columns:
                print(f'Duration indicator column is {indicator}\n')
                dur['indicator'] = indicator
                df[indicator] = df[indicator] + 1e-10
            else:
                print('Invalid duration column indicator\n')
                return None

        elif cen_input == 2:
            type_in = input('Are the indicators duration or dates? (Duration/Dates):')
            if type_in == 'Duration':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df['duration_new'] = df['duration_new'] + 1e-10
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')

            elif type_in == 'Dates':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")

                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                    for ind in [indicator_1, indicator_2]:
                        df[ind] = pd.to_datetime(df[ind])
                    df.loc[df[indicator_1].isnull(), indicator_1] = df[indicator_2]
                    df.loc[df[indicator_2].isnull(), indicator_2] = df[indicator_1]

                    print('\nDatetime frequency explanation:')
                    print('M - Monthly, when the lower and upper bounds differ by atleast a month or 30 days')
                    print('Y - Monthly, when the lower and upper bounds differ by atleast a year or 365 days')
                    print('D - Monthly, when the lower and upper bounds differ by atleast a day or 24 hours')
                    print('H - Monthly, when the differnece between lower and upper bound is less than 24 hours.\n')

                    freq = input(
                        'Enter the frequency of the data (M for Monthly, Y for Yearly, H for Hourly, D for Day) :')
                    freq_dict = {'M': 30, 'Y': 365, 'D': 1, 'H': 24}

                    # Checking the frequency of the duration data
                    if freq in ['M', 'Y', 'D', 'H']:
                        df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days/freq_dict[freq]
                        print(f'Datetimes have been converted to durations of {freq}.\n')
                        df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                        df['duration_new'] = df['duration_new'] + 1e-10
                        dur['indicator'] = 'duration_new'
                    else:
                        print('Default frequency - Day')
                        df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days/freq_dict[freq]
                        print('Datetimes have been converted to durations of Days.\n')
                        df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                        df['duration_new'] = df['duration_new'] + 1e-10
                        dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')
                    return None
        else:
            print('Invalid number of duration column\n')
            return None

    # Interval Censoring
    # for interval censoring 'dur' will be a dictionary and not a dictionary
    else:
        type_in = input('Are the indicators duration or dates? (Duration/Dates):')
        if type_in == 'Duration':
            indicator_1 = input("Enter the lower bound duration column name: ")
            indicator_2 = input("Enter the upper bound duration column name: ")
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
            else:
                print('Invalid duration column indicators\n')
                return None

        elif type_in == 'Dates':
            indicator_1 = input("Enter the lower bound duration column name: ")
            indicator_2 = input("Enter the upper bound duration column name: ")
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                for ind in [indicator_1, indicator_2]:
                    df[ind] = pd.to_datetime(df[ind])
                    # Will check whether interval censoring can work directly on datetime columns or not
            else:
                print('Invalid duration column indicators\n')
                return None

        else:
            print('Invalid duration indicator\n')
            return None

    target = input("Enter the Target event Column :")
    if target in df.columns:
        print(f'Target column is {target}\n')
        if df[target].dtype in ['object', 'str']:
            df[target] = df[target].replace({'False': 0, 'True': 1, 'false': 0, 'true': 1, 'f': 0, 't': 1})
        elif df[target].dtype == bool:
            df[target] = df[target].replace({False: 0, True: 1})
    else:
        print("Target event entered does not exist in DataFrame: Please check spelling ")
        return None

    print('\nPlease drop complete text columns except categorical columns and give ONE space after every comma.')
    cols_to_drop = [item for item in input("Enter the name of columns separated by comma that you want to drop"
                                           " or None in case there are no columns to drop: ").split(', ')]

    if cols_to_drop[0] != "None":
        for col in cols_to_drop:
            if col not in df.columns:
                print('Please check column name!')
                return None
        df.drop(cols_to_drop, axis=1, inplace=True)
        print('Unwanted columns have been dropped\n')
    else:
        print('There are NO unwanted columns')
    df = df.dropna()
    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur['indicator'])
    # df = remove_outliers(df)
    # print('Outliers have been removed.\n')

    print(f'Final shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns.\n')
    # col_to_scale = [cols for cols in df.columns if cols not in [dur['indicator'], target]]
    # scale = MinMaxScaler()
    # df[col_to_scale] = scale.fit_transform(df[col_to_scale])

    print("Visualising the final DataFrame")
    display(df.head(10))
    # target is the target column name
    return df, target, dur, censoring_type


# Exploratory Data Analysis
def dataExploration(df, target_col):
    values = df[target_col]
    plt.figure(figsize=(15, 5))
    plt.plot(values)
    plt.title('Duration column')
    plt.show()


def remove_outliers(df):
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]
    return df


def lifetimes(time_events, actual, censor,
              lower_bound_t, upper_bound_t):
    if censor != 'Interval':
        plt.title('Lifetime variation')
        ax = plot_lifetimes(time_events, actual)
        plt.show()
    else:
        plt.title('Lifetime Variation')
        ax = plot_interval_censored_lifetimes(lower_bound_t, upper_bound_t)
        plt.show()


# Is original series normally distributed?
# Is the std deviation normally distributed?
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


def createResultFrame_uni(resultsDict, predictionsDict):
    result_df = pd.DataFrame(columns=['Model', '_concordance_index'])
    for model, values in resultsDict.items():
        temp = [model]
        temp.extend(values.values())
        result_df.loc[len(result_df)] = temp
        result_df.sort_values(by=['_concordance_index'], inplace=True, ignore_index=True, ascending=False)
    print("\nModel Information Table [sorted by Concordance Index score]")
    print(tabulate(result_df, headers=[
        'Model', '_concordance_index'], tablefmt="fancy_grid"))
    result = result_df.iloc[0, 0]
    print(f"Winner model is {result}")
    testPlot_uni({result: predictionsDict[result]})
    return result


def createResultFrame_uni_interval(resultsDict, predictionsDict):
    result_df = pd.DataFrame(columns=['Model', '_concordance_index'])
    for model, values in resultsDict.items():
        temp = [model]
        temp.extend(values.values())
        result_df.loc[len(result_df)] = temp
        result_df.sort_values(by=['_concordance_index'], inplace=True, ignore_index=True, ascending=False)
    print("\nModel Information Table [sorted by Concordance Index score]")
    print(tabulate(result_df, headers=[
        'Model', '_concordance_index'], tablefmt="fancy_grid"))
    result = result_df.iloc[0, 0]
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
    t_mean = T_train.mean()

    if censor in ['Right', 'Uncensored']:
        modelling_obj = Uni_Modelling(T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict)
        modelling_obj.modeller()
        plot_rmst(modelsDict, t_mean)
        plot_cdf(modelsDict)
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_cindex_(resultsDict)
        winnerModelName = createResultFrame_uni(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName
    elif censor == 'Left':
        modelling_obj = Uni_Modelling_left(T_train, T_test, E_train, E_test, resultsDict, predictionsDict, modelsDict)
        modelling_obj.modeller()
        plot_rmst(modelsDict, t_mean)
        plot_cdf(modelsDict)
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_cindex_(resultsDict)
        winnerModelName = createResultFrame_uni(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, T_train, T_test
        return winnerModelName


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
    upper_t = df_training[dur['indicator']['upper']]
    lower_t = df_training[dur['indicator']['lower']]

    # T_train = df_training[dur['indicator']]
    print(df_training[[lower_t, upper_t, target]].head(5), '\n')
    df_test, E_test = targetEngineering(df_test, target=target)
    upper_tt = df_training[dur['indicator']['upper']]
    lower_tt = df_training[dur['indicator']['lower']]
    ut_mean = upper_t.mean()
    lt_mean = lower_t.mean()

    try:
        modelling_obj = Uni_interval(lower_t, upper_t, lower_tt,
                                     upper_tt, E_train, E_test, resultsDict, predictionsDict, modelsDict)
        modelling_obj.modeller()
        plot_rmst_interval(modelsDict, lt_mean, ut_mean)
        # plot_cdf(modelsDict)
        # print('\nPrediction estimate for all models')
        # testPlot_uni(predictionsDict)
        plot_cindex_(resultsDict)
        winnerModelName = createResultFrame_uni_interval(resultsDict, predictionsDict)
        del df, df_training, df_test, E_train, E_test, lower_t, lower_tt, upper_t, upper_tt
        return winnerModelName
    except:
        print('The data submitted by the user is not Interval Censored. Please Check Again.\n')
        return None


def winnerModelTrainer_uni(df, target, winnerModelName, dur, censor):
    # print("Inside Winner Trainer",len(df))
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    T = df[dur['indicator']]
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    df_test = df.loc[df.index > split_date]
    T_test = df_test[dur['indicator']]

    if censor == 'Left':
        # Getting winner model object
        winner_obj = Uni_Modelling_left(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        scoring_obj = Uni_Modelling_left(T, T_test, E, pd.DataFrame(), None, None, None)
        winnerPredictionsList = scoring_obj.scoring(winnerModel)
        # print("Length of winner predictions",len(winnerPredictionsList))

    elif censor in ['Right', 'Uncensored']:
        # Getting winner model object
        winner_obj = Uni_Modelling(T, pd.DataFrame(), E, pd.DataFrame(), None, None, None)
        winnerModel = winner_obj.getWinnerModel(winnerModelName)
        print("Univariate Model Scoring Running...")
        # winnerPlots(winnerModelName, winnerModel)
        scoring_obj = Uni_Modelling(T, T_test, E, pd.DataFrame(), None, None, None)
        winnerPredictionsList = scoring_obj.scoring(winnerModel)

        # print("Length of winner predictions",len(winnerPredictionsList))
    if winnerModelName not in ['NelsonAalenFitter', 'KaplanMeierFitter']:
        print(winnerModel.print_summary(decimals=4), '\n')
    winnerPlots(winnerModelName, winnerModel, T, E)
    winnerPredictions = winnerPredictionsList.to_frame()
    winnerPredictions.index.name = 'Point of Time'
    winnerPredictions.columns = ['Winner Model Estimate']
    print(len(winnerPredictions))
    display(winnerPredictions.head(10))
    testPlot_uni({'Winner Model Estimate': winnerPredictions['Winner Model Estimate']})

    winnerPredictions.to_csv("scoring.csv")
    return


def winnerModelTrainer_uni_interval(df, target, winnerModelName, dur):
    # print("Inside Winner Trainer",len(df))
    df, E = targetEngineering(df, target)
    # print(X.head(5))
    upper = df[dur['indicator']['upper']]
    lower = df[dur['indicator']['lower']]
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    df_test = df.loc[df.index > split_date]
    upper_tt = df_test[dur['indicator']['upper']]
    lower_tt = df_test[dur['indicator']['lower']]

    # Getting winner model object
    winner_obj = Uni_interval(lower, upper, pd.DataFrame(),
                              pd.DataFrame(), E, pd.DataFrame(), None, None, None)
    winnerModel = winner_obj.getWinnerModel(winnerModelName)
    print("Univariate Model Scoring Running...")
    scoring_obj = Uni_interval(lower, upper, lower_tt, upper_tt, E, pd.DataFrame(), None, None, None)
    winnerPredictionsList = scoring_obj.scoring(winnerModel)
    # print("Length of winner predictions",len(winnerPredictionsList))
    print(winnerModel.print_summary(decimals=4), '\n')
    # print("Length of winner predictions",len(winnerPredictionsList))
    winnerPlots_interval(winnerModelName, winnerModel, lower, upper, E)
    winnerPredictions = winnerPredictionsList[0].to_frame()
    winnerPredictions.index.name = 'Point of Time'
    winnerPredictions.columns = ['Winner Model Estimate']
    print(len(winnerPredictions))
    print(winnerPredictions.head(10))
    testPlot_uni_interval({'Winner Model Estimate': winnerPredictions['Winner Model Estimate']})

    winnerPredictions.to_csv("scoring.csv")
    return
