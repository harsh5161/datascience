from userInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib as mpl
from tabulate import tabulate
from plots import plot_cindex_, plot_cdf, plot_rmst, winnerPlots_interval, plot_rmst_interval, winnerPlots, plot_aic
from univariate_modelling import Uni_Modelling, Uni_Modelling_left
from interval_modelling import Uni_interval
from lifelines.plotting import *
from random import randint
from scipy import stats
from IPython.display import display
import seaborn as sns

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
    # Correct path of the dataset
    path = input("Enter the path of the dataset :")
    try:
        df, _ = importFile(path)
        display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None

    # ID column is kept optional
    key = input('\nEnter optional ID column or enter "None" if no id column is present: ')
    if id in df.columns:
        df = df.drop(key, axis=1)

    # Dictionary to store the column names for time events
    dur = {}

    # Feature encoding for categorical features
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

        lb = LabelEncoder()  # Using LabelEncoder as semantics of the column is unknown
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

    censoring_type = input(
        "\nEnter the type of censoring present in the dataset (Right/Uncensored/Left/Interval) :").upper()
    if censoring_type not in ['RIGHT', 'UNCENSORED', 'LEFT', 'INTERVAL']:
        print('Invalid censoring type')

    # Censoring except Interval censoring
    elif censoring_type in ['UNCENSORED', 'RIGHT', 'LEFT']:
        cen_input = int(input(
            "\nNumber of duration columns? (1/2) :"))  # Duration can be expressed as a single column as well as 2 columns

        # A single time event column can be present indicating the duration of event
        if cen_input == 1:
            indicator = input("Enter the single duration column name: ")
            if indicator in df.columns:
                print(f'Duration indicator column is {indicator}\n')
                dur['indicator'] = indicator
                df[df[indicator] == 0] = 1e-5
                df[indicator] = abs(df[indicator])
            else:
                print('Invalid duration column indicator\n')
                return None

        # 2 time events column can be present to indicate the start and end of events
        elif cen_input == 2:

            # Indicators can be both duration and dates
            # When indicators are durations, we can calculate the difference to get the duration
            type_in = input('Are the indicators duration or dates? (Duration/Dates):')
            if type_in == 'Duration':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5
                    df['duration_new'] = abs(df['duration_new'])
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')

            elif type_in == 'Dates':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")

                # Datetime differences are converted to days which are normalized to frequency of datetime
                # datetime_to_duration method has a glitch, therefore manual conversion
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                    for ind in [indicator_1, indicator_2]:
                        df[ind] = pd.to_datetime(df[ind])
                    df.loc[df[indicator_1].isnull(), indicator_1] = df[indicator_2]
                    df.loc[df[indicator_2].isnull(), indicator_2] = df[indicator_1]

                    # Explanation of different datetime frequencies
                    print('\nDatetime frequency explanation:')
                    print('M - Monthly, when the lower and upper bounds differ by atleast a month or 30 days')
                    print('Y - Monthly, when the lower and upper bounds differ by atleast a year or 365 days')
                    print('D - Monthly, when the lower and upper bounds differ by atleast a day or 24 hours')
                    print('H - Monthly, when the differnece between lower and upper bound is less than 24 hours.\n')

                    freq = input(
                        'Enter the frequency of the data (M for Monthly, Y for Yearly, H for Hourly, D for Day) :')\
                        .upper()
                    freq_dict = {'M': 30, 'Y': 365, 'D': 1, 'H': 24}

                    # Checking the frequency of the duration data
                    if freq in ['M', 'Y', 'D', 'H']:
                        df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days / freq_dict[freq]
                        print(f'Datetimes have been converted to durations of {freq}.\n')
                        df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                        df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                        df['duration_new'] = abs(df['duration_new'])
                        dur['indicator'] = 'duration_new'
                    else:
                        print('Default frequency - Day')
                        df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days / freq_dict[freq]
                        print('Datetimes have been converted to durations of Days.\n')
                        df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                        df[df['duration_new'] == 0] = 1e-5
                        df['duration_new'] = abs(df['duration_new'])
                        dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')
                    return None
        else:
            print('Invalid number of duration column\n')
            return None

    # Interval Censoring
    # for interval censoring 'dur' will be a dictionary containing lists
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

        # Check logic
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

    # Target Encoding
    # Negative - 0, Positive - 1
    if censoring_type != 'Interval':
        target = input("Enter the Target event Column :")
        if target in df.columns:
            print(f'Target column is {target}\n')

            # Event target can have different datatypes as well as different display conventions
            if df[target].dtype in ['object', 'str']:
                df[target] = df[target].str.lower()
                df[target] = df[target].replace({'false': 0, 'true': 1, 'f': 0, 't': 1, 'yes': 1, 'no': 0, 'y': 1,
                                                 'n': 0})
            elif df[target].dtype == bool:
                df[target] = df[target].replace({False: 0, True: 1})
        else:
            print("Target event entered does not exist in DataFrame: Please check spelling ")
            return None

    # Target encoding for Interval Censoring
    else:
        target_opt = input('Is a target event column present? (Y/N): ')
        if target_opt == 'Y':
            target = input("Enter the Target event Column :")
            if target in df.columns:
                print(f'Target column is {target}\n')

                # Event target can have different datatypes as well as different display conventions
                if df[target].dtype in ['object', 'str']:
                    df[target] = df[target].str.lower()
                    df[target] = df[target].replace({'false': 0, 'true': 1, 'f': 0, 't': 1, 'yes': 1, 'no': 0, 'y': 1,
                                                     'n': 0})
                elif df[target].dtype == bool:
                    df[target] = df[target].replace({False: 0, True: 1})
            else:
                print("Target event entered does not exist in DataFrame: Please check spelling ")
                return None

        # Interval censoring need not necessarily have an event target column
        elif target_opt == 'N':

            # Generation of event target column by Interval Censoring rule
            # When lower bound is equal to upper bound, event target is equal to 1
            print('Generating target column for Interval censored data\n')
            df.loc[df[dur['indicator_1']].eq(df[dur['indicator_2']]), 'target'] = 1

    # Optional columns to drop
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
    print('Null values have been dropped\n')

    # Check code for removing outliers, doesn't work
    # Important for solving ConvergenceError in case of Multivariate Analysis
    # df = remove_outliers(df)
    # print('Outliers have been removed.\n')

    print(f'Final shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns.\n')

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censoring_type)

    print("Visualising the final DataFrame")
    display(df.head(10))

    # If quicker results, only Univariate analysis will work and otherwise
    inp = input('Do you want quick results or slower results? If quick, enter "y": ').lower()
    rate = True if inp == 'y' else False

    # target is the target column name
    return df, target, dur, censoring_type, rate


# Exploratory Data Analysis
def dataExploration(df, dict_, censor):
    if censor != 'Interval':
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


def remove_outliers(df):
    z = np.abs(stats.zscore(df))
    new_df = df[(z < 3).all(axis=1)]
    return new_df


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
    if len(df) < 2000:
        return int(len(df) * 0.90)
    else:
        return int(len(df) * 0.75)


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
        plt.plot(value[0], color=cmap(randint(0, 50)), label=f'{key}')
        plt.legend()
    plt.show()


def createResultFrame_uni(resultsDict, predictionsDict):
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
    testPlot_uni({result: predictionsDict[result]})
    return result


def createResultFrame_uni_interval(resultsDict, predictionsDict):
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
        plot_aic(resultsDict)
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
        plot_aic(resultsDict)
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
    winnerModelName = createResultFrame_uni_interval(resultsDict, predictionsDict)
    del df, df_training, df_test, E_train, E_test, lower_t, lower_tt, upper_t, upper_tt
    return winnerModelName
    # except:
    #     print('The data submitted by the user is not Interval Censored. Please Check Again.\n')
    #     return None


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
    try:
        testPlot_uni_interval({'Winner Model Estimate': winnerPredictions['Winner Model Estimate']})
    except:
        pass
    winnerPredictions.to_csv("scoring.csv")
    return
