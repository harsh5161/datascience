import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from userInputs import importFile
from IPython.display import display
from Modules import dataExploration
from engineering import *
from sklearn_pandas import CategoricalImputer
import category_encoders as ce

pd.set_option('display.max_columns', None)


def userInputs_():
    # with joblib.parallel_backend('dask'):
    # Correct path of the dataset
    path = input("Enter the path of the dataset: ")
    try:
        df, _ = importFile(path)
        # display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None

    # print(df.head())
    ###############################
    ##### Numeric Engineering #####
    ###############################


    # ID column is kept optional
    key = input('\nEnter optional ID column or enter "None" if no id column is present: ')
    print(df.head())
    if key in df.columns:
        df = df.drop([key], axis=1)
        orig = df.copy()
        # df = numeric_engineering(df)
        print(df.dtypes)
    else:
        orig = df.copy()
        # df = numeric_engineering(df)
    ###############################
    ##### Numeric Engineering #####
    ###############################

    ###############################
    ########## Censoring ##########
    ###############################
    # Definition of different kind of censoring types that may be present in the data
    print('\nExplanation of different types of censoring:')
    print('1. Right censoring is when the event of interest happens after the survey timeframe.')
    print('2. Left censoring is when the event of interest has happened before the start of the survey timeframe.')
    print('3. Uncensored is when the event of interest happens within the survey time frame.')
    print('4. Interval censoring is when there is a mix of Right and Left censoring.\n')

    dur = {}

    censoring_type = input(
        "\nEnter the type of censoring present in the dataset (Right/Uncensored/Left/Interval) :").upper()
    if censoring_type == '':
        censoring_type = 'UNCENSORED'
    print('Censoring type is chosen as ', censoring_type)

    # Censoring except Interval censoring
    if censoring_type in ['UNCENSORED', 'RIGHT', 'LEFT']:
        dur_cols = [item for item in input("Enter the starting date(optional) and the end date"
                                           " separated by a comma: ").split(', ')]

        # A single time event column can be present indicating the duration of event
        if len(dur_cols) == 1:
            if dur_cols[0] in df.columns:
                print(f'Duration indicator column is {dur_cols[0]}\n')
                dur['indicator'] = dur_cols[0]
                df[df[dur_cols[0]] == 0] = 1e-5
                df[dur_cols[0]] = abs(df[dur_cols[0]])
            else:
                print('Invalid duration column indicator\n')
                return None

        # 2 time events column can be present to indicate the start and end of events
        elif len(dur_cols) == 2:
            # Indicators can be both duration and dates
            # When indicators are durations, we can calculate the difference to get the duration
            if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print('Working with numerical duration column')
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5
                    df['duration_new'] = abs(df['duration_new'])
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')

            else:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                # Datetime differences are converted to days which are normalized to frequency of datetime
                # datetime_to_duration method has a glitch, therefore manual conversion
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print('Working with non-numerical duration column')
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                    for ind in [indicator_1, indicator_2]:
                        try:
                            df[ind] = pd.to_datetime(df[ind])
                        except:
                            try:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                            except:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    print('Converted to datetime...')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    print('Removed ambiguous duration points')
                    df.loc[df[indicator_1].isnull(), indicator_1] = df[indicator_2]
                    df.loc[df[indicator_2].isnull(), indicator_2] = df[indicator_1]
                    print('Imputed null values')

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]') / (24 * 60)
                    print(f'Datetimes have been converted to durations of days.\n')
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                    # scaler = MinMaxScaler(feature_range=(1, 100))
                    df['duration_new'] = abs(df['duration_new'])
                    # df['duration_new'] = scaler.fit_transform(df['duration_new'])
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
        dur_cols = [item for item in input("Enter the starting date(optional) and the end date"
                                           " separated by a comma: ").split(', ')]
        if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                print(dur)
                # print('Duration columns have been stored...')
            else:
                print('Invalid duration column indicators\n')
                return None

        # Check logic
        else:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                for ind in [indicator_1, indicator_2]:
                    try:
                        df[ind] = pd.to_datetime(df[ind])
                    except:
                        try:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                        except:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    # Will check whether interval censoring can work directly on datetime columns or not
            else:
                print('Invalid duration column indicators\n')
                return None
    ###############################
    ########## Censoring ##########
    ###############################
    if censoring_type != 'INTERVAL':
        print(df[dur['indicator']].min(), df[dur['indicator']].max())
    else:
        print(df[dur['indicator']['lower']].min(), df[dur['indicator']['upper']].max())
    #########################
    #### Target Encoding ####
    #########################
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
            elif df[target].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])
        else:
            print("Target event entered does not exist in DataFrame: Please check spelling ")
            return None

    # Target encoding for Interval Censoring
    else:
        target_opt = input('Is a target event column present? (Y/N): ').lower()
        if target_opt == 'y':
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
                elif df[target].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                    le = LabelEncoder()
                    df[target] = le.fit_transform(df[target])
            else:
                print("Target event entered does not exist in DataFrame: Please check spelling ")
                return None

        # Interval censoring need not necessarily have an event target column
        elif target_opt == 'n':
            # Generation of event target column by Interval Censoring rule
            # When lower bound is equal to upper bound, event target is equal to 1
            print('Generating target column for Interval censored data\n')
            df.loc[df[dur['indicator_1']].eq(df[dur['indicator_2']]), 'target'] = 1
    #########################
    #### Target Encoding ####
    #########################

    print('\nNumber of null values in each column:-')
    print(df.isnull().sum())

    print(f'\nFinal shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns.\n')

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censoring_type)

    print("\nVisualising a part of the final DataFrame")
    display(df.iloc[:5, :])

    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

    if censoring_type != 'INTERVAL':
        df = df[[dur['indicator'], target]]
    else:
        df = df[[dur['indicator']['lower'], dur['indicator']['upper'], target]]

    return df, target, dur, censoring_type, orig, dur_cols


def lead_input_(orig, censor, dur, target, dur_cols):
    # df = df.drop([target], axis=1)
    path = input("Enter the path of the dataset: ")
    try:
        df, _ = importFile(path)
        # display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None
    df = df[(df.columns) & (orig.columns)]
    df[target] = 0
    # df = numeric_engineering(df)

    # Censoring except Interval censoring
    if censor in ['UNCENSORED', 'RIGHT', 'LEFT']:

        # A single time event column can be present indicating the duration of event
        if len(dur_cols) == 1:
            if dur_cols[0] in df.columns:
                print(f'Duration indicator column is {dur_cols[0]}\n')
                dur['indicator'] = dur_cols[0]
                df[df[dur_cols[0]] == 0] = 1e-5
                df[dur_cols[0]] = abs(df[dur_cols[0]])
            else:
                print('Invalid duration column indicator\n')
                return None

        # 2 time events column can be present to indicate the start and end of events
        elif len(dur_cols) == 2:
            # Indicators can be both duration and dates
            # When indicators are durations, we can calculate the difference to get the duration
            if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5
                    df['duration_new'] = abs(df['duration_new'])
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')

            else:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                # Datetime differences are converted to days which are normalized to frequency of datetime
                # datetime_to_duration method has a glitch, therefore manual conversion
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                    for ind in [indicator_1, indicator_2]:
                        try:
                            df[ind] = pd.to_datetime(df[ind])
                        except:
                            try:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                            except:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    df.loc[df[indicator_1].isnull(), indicator_1] = df[indicator_2]
                    df.loc[df[indicator_2].isnull(), indicator_2] = df[indicator_1]

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]') / (24 * 60)
                    print(f'Datetimes have been converted to durations of days.\n')
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                    # scaler = MinMaxScaler(feature_range=(1, 100))
                    df['duration_new'] = abs(df['duration_new'])
                    # df['duration_new'] = scaler.fit_transform(df['duration_new'])
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
        if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
            else:
                print('Invalid duration column indicators\n')
                return None

        # Check logic
        else:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                for ind in [indicator_1, indicator_2]:
                    try:
                        df[ind] = pd.to_datetime(df[ind])
                    except:
                        try:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                        except:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    # Will check whether interval censoring can work directly on datetime columns or not
            else:
                print('Invalid duration column indicators\n')
                return None
        ###############################
        ########## Censoring ##########
        ###############################

    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

    if censor != 'INTERVAL':
        df = df[[dur['indicator']]]
    else:
        df = df[[dur['indicator']['lower'], dur['indicator']['upper']]]

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censor)
    # df = df.drop([target], axis=1)
    return df, dur, censor, target


def lead_input_target_(orig, censor, dur, target, dur_cols):
    # df = df.drop([target], axis=1)
    path = input("Enter the path of the dataset: ")
    try:
        df, _ = importFile(path)
        # display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None
    df = df[(df.columns) & (orig.columns)]
    # df[target] = 0
    # df = numeric_engineering(df)

    # Censoring except Interval censoring
    if censor in ['UNCENSORED', 'RIGHT', 'LEFT']:

        # A single time event column can be present indicating the duration of event
        if len(dur_cols) == 1:
            if dur_cols[0] in df.columns:
                print(f'Duration indicator column is {dur_cols[0]}\n')
                dur['indicator'] = dur_cols[0]
                df[df[dur_cols[0]] == 0] = 1e-5
                df[dur_cols[0]] = abs(df[dur_cols[0]])
            else:
                print('Invalid duration column indicator\n')
                return None

        # 2 time events column can be present to indicate the start and end of events
        elif len(dur_cols) == 2:
            # Indicators can be both duration and dates
            # When indicators are durations, we can calculate the difference to get the duration
            if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    df['duration_new'] = df[indicator_2] - df[indicator_1]
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5
                    df['duration_new'] = abs(df['duration_new'])
                    dur['indicator'] = 'duration_new'
                else:
                    print('Invalid duration column indicators\n')

            else:
                indicator_1 = dur_cols[0]
                indicator_2 = dur_cols[1]
                # Datetime differences are converted to days which are normalized to frequency of datetime
                # datetime_to_duration method has a glitch, therefore manual conversion
                if indicator_2 in df.columns and indicator_1 in df.columns:
                    print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                    for ind in [indicator_1, indicator_2]:
                        try:
                            df[ind] = pd.to_datetime(df[ind])
                        except:
                            try:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                            except:
                                df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    df = df[df[indicator_1] <= df[indicator_2]]
                    df.loc[df[indicator_1].isnull(), indicator_1] = df[indicator_2]
                    df.loc[df[indicator_2].isnull(), indicator_2] = df[indicator_1]

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]') / (24 * 60)
                    print(f'Datetimes have been converted to durations of days.\n')
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                    # scaler = MinMaxScaler(feature_range=(1, 100))
                    df['duration_new'] = abs(df['duration_new'])
                    # df['duration_new'] = scaler.fit_transform(df['duration_new'])
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
        if df[dur_cols[0]].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
            else:
                print('Invalid duration column indicators\n')
                return None

        # Check logic
        else:
            indicator_1 = dur_cols[0]
            indicator_2 = dur_cols[1]
            df = df[df[indicator_1] <= df[indicator_2]]
            if indicator_2 in df.columns and indicator_1 in df.columns:
                print(f'Duration indicator columns is {indicator_1} and {indicator_2}\n')
                dur['indicator'] = {'lower': indicator_1, 'upper': indicator_2}
                for ind in [indicator_1, indicator_2]:
                    try:
                        df[ind] = pd.to_datetime(df[ind])
                    except:
                        try:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M', errors='coerce')
                        except:
                            df[ind] = pd.to_datetime(df[ind], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    # Will check whether interval censoring can work directly on datetime columns or not
            else:
                print('Invalid duration column indicators\n')
                return None
        ###############################
        ########## Censoring ##########
        ###############################

    if censor != 'Interval':
        # target = input("Enter the Target event Column :")
        if target in df.columns:
            print(f'Target column is {target}\n')

            # Event target can have different datatypes as well as different display conventions
            if df[target].dtype in ['object', 'str']:
                df[target] = df[target].str.lower()
                df[target] = df[target].replace({'false': 0, 'true': 1, 'f': 0, 't': 1, 'yes': 1, 'no': 0, 'y': 1,
                                                 'n': 0})
            elif df[target].dtype == bool:
                df[target] = df[target].replace({False: 0, True: 1})
            elif df[target].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])
        else:
            print("Target event entered does not exist in DataFrame: Please check spelling ")
            return None

    # Target encoding for Interval Censoring
    else:
        target_opt = input('Is a target event column present? (Y/N): ').lower()
        if target_opt == 'y':
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
                elif df[target].dtype in ['float', 'int', np.float32, np.float64, np.int32, np.int64, int, float]:
                    le = LabelEncoder()
                    df[target] = le.fit_transform(df[target])
            else:
                print("Target event entered does not exist in DataFrame: Please check spelling ")
                return None

        # Interval censoring need not necessarily have an event target column
        elif target_opt == 'n':
            # Generation of event target column by Interval Censoring rule
            # When lower bound is equal to upper bound, event target is equal to 1
            print('Generating target column for Interval censored data\n')
            df.loc[df[dur['indicator_1']].eq(df[dur['indicator_2']]), 'target'] = 1

    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

    if censor != 'INTERVAL':
        df = df[[dur['indicator'], target]]
    else:
        df = df[[dur['indicator']['lower'], dur['indicator']['upper'], target]]

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censor)
    # df = df.drop([target], axis=1)
    return df, dur, censor, target
