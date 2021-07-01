from UserInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib as mpl
from tabulate import tabulate
from plots import bar_metrics, plot_cindex_, plot_simple_cindex, plot_aic, plot_cdf, plot_rmst
from univariate_modelling import Uni_Modelling, Uni_Modelling_left
from multivariate_modelling import Multi_Modelling, Multi_Modelling_left
from interval_modelling import Uni_interval, Multi_interval
from lifelines.utils import datetimes_to_durations
from lifelines.plotting import *
from random import randint

seed = 42
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


# You want to call this script separately so that we take in a large dataset input by the user and push
# it out of memory after extracting the two columnss that we need


def userInputs():
    path = input("Enter the path of the dataset :")
    try:
        df, _ = importFile(path)
        print(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None

    key = input('Enter optional ID column: ')
    if id in df.columns:
        df = df.drop(key, axis=1)

    dur = {}

    categorical = [item for item in input("Enter the name of categorical columns : ").split(', ')]
    for col in categorical:
        if col in df.columns:
            pass
        else:
            print('Invalid Categorical column name')

    lb = LabelEncoder()
    for col in categorical:
        df[col] = lb.fit_transform(df[col])

    censoring_type = input("Enter the type of censoring present in the dataset (Right/Uncensored/Left/Interval) :")
    if censoring_type not in ['Right', 'Uncensored', 'Left', 'Interval']:
        print('Invalid censoring type')

    # Censoring except Interval censoring
    elif censoring_type in ['Uncensored', 'Right', 'Left']:
        cen_input = input(
            "Number of duration columns? (1/2) :")  # Duration can be expressed as a single column as well as 2 columns

        if cen_input == 1:
            indicator = input("Enter the single duration column name: ")
            if indicator in df.columns:
                print(f'Duration indicator column is {indicator}')
                dur['indicator'] = indicator
            else:
                print('Invalid duration column indicator')
                return None

        elif cen_input == 2:
            type_in = input('Are the indicators duration or dates? (Duration/Dates):')
            if type_in == 'Duration':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators')

            elif type_in == 'Dates':
                indicator_1 = input("Enter the lower bound duration column name: ")
                indicator_2 = input("Enter the upper bound duration column name: ")

                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    for ind in [indicator_1, indicator_2]:
                        df[ind] = pd.to_datetime(df[ind])
                    freq = input(
                        'Enter the frequency of the data ( M for Monthly, Y for Yearly, H for Hourly, D for Day) :')

                    # Checking the frequency of the duration data
                    if freq in ['M', 'Y', 'D', 'H']:
                        T, E = datetimes_to_durations(indicator_1, indicator_2, freq=freq)
                        df['duration_new'] = T
                        # df['events_new'] = E
                        # df['events_new'].replace({True: 1, False: 0}, inplace=True)
                        dur['indicator'] = 'duration_new'
                    else:
                        print('Default frequency - Day')
                        T, E = datetimes_to_durations(indicator_1, indicator_2)
                        df['duration_new'] = T
                        # df['events_new'] = E
                        # df['events_new'].replace({True: 1, False: 0}, inplace=True)
                        dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators')
                    return None

        else:
            print('Invalid number of duration columns')
            return None

    # Interval Censoring
    else:
        type_in = input('Are the indicators duration or dates? (Duration/Dates):')
        if type_in == 'Duration':
            indicator_1 = input("Enter the lower bound duration column name: ")
            indicator_2 = input("Enter the upper bound duration column name: ")
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                dur['indicator'] = {'lower':indicator_1, 'upper':indicator_2}
            else:
                print('Invalid duration column indicators')
                return None

        elif type_in == 'Dates':
            indicator_1 = input("Enter the lower bound duration column name: ")
            indicator_2 = input("Enter the upper bound duration column name: ")
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                for ind in [indicator_1, indicator_2]:
                    df[ind] = pd.to_datetime(df[ind])
                    # Will check whether interval censoring can work directly on datetime columns or not
            else:
                print('Invalid duration column indicators')
                return None

        else:
            print('Invalid duration indicator')
            return None

    print("Exploring the different series' present in the DataFrame")
    dataExploration(df)

    try:
        target = input("Enter the Target event Column :")
        plt.figure(num=None, figsize=(40, 15), dpi=80,
                   facecolor='w')
        plt.title(f'{target}', fontsize=30)

        plt.plot(df[target])
        plt.show()
        # plt.savefig(f'{target}.png')
    except:
        print("Target event entered does not exist in DataFrame or couldn't be plotted : Please check spelling ")
        return None

    print("Visualising the final DataFrame")
    print(df.head(10))
    return df, target, dur


# Exploratory Data Analysis
def dataExploration(df):
    values = df.values
    groups = [i for i in range(len(df.columns.to_list()) - 1)]
    print(f'Groups are : {groups}')
    i = 1
    plt.figure(figsize=(10, 10))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


def lifetimes(time_events, actual, censor,
              lower_bound_t, upper_bound_t):
    if censor != 'Interval':
        ax = plot_lifetimes(time_events, actual)
        plt.show()
    else:
        ax = plot_interval_censored_lifetimes(lower_bound_t, upper_bound_t)
        plt.show()


# Is original series normally distributed?
# Is the std deviation normally distributed?
def normalityPlots(series):
    fig = plt.figure(figsize=(12, 7))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    hist_std_ax = plt.subplot2grid(layout, (0, 1))
    series.hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")
    mm = series.std()
    mm.hist(ax=hist_std_ax)
    hist_std_ax.set_title("Standard deviation histogram")

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