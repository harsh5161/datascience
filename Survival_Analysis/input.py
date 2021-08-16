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


def userInputs():
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
    df = df.sample(n=500000) if len(df) > 500000 else df
    df = df.reset_index(drop=True)

    # ID column is kept optional
    key = input('\nEnter optional ID column or enter "None" if no id column is present: ')
    print(df.head())
    if key in df.columns:
        df = df.drop([key], axis=1)
        orig = df.copy()
        df = numeric_engineering(df)
        print(df.dtypes)
    else:
        orig = df.copy()
        df = numeric_engineering(df)
    ###############################
    ##### Numeric Engineering #####
    ###############################

    ##############################
    #### LAT-LONG Engineering ####
    ##############################
    lat, lon, lat_lon_cols = findLatLong(df)
    print("Latitude Columns Impacted are: ", lat)
    print("Longitude Columns Impacted are:", lon)
    print("Latitude-Longitude Columns Impacted are:", lat_lon_cols)

    if (lat and lon) or lat_lon_cols:
        try:
            LAT_LONG_DF = latlongEngineering(df, lat, lon, lat_lon_cols)
            # print(LAT_LONG_DF)
            # print(LAT_LONG_DF.shape)
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
        except Exception as exceptionMessage:
            print('#### LAT-LONG ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
            LAT_LONG_DF = pd.DataFrame(None)
            lat = []
            lon = []
            lat_lon_cols = []
    else:
        print("No Latitude or Longitude Columns found")
        LAT_LONG_DF = pd.DataFrame(None)
        lat = []
        lon = []
        lat_lon_cols = []
    if len(LAT_LONG_DF):
        df = pd.concat([df, LAT_LONG_DF], axis=1)

    print(df.columns)
    ##############################
    #### LAT-LONG Engineering ####
    ##############################

    ##################################
    ### URL and E-Mail Engineering ###
    #################################
    obj_df = df.select_dtypes('object')  # Sending in only object dtype columns
    short_obj_df = obj_df.astype(str).sample(3000, random_state=42).dropna(how='all') if len(
        obj_df) > 3000 else obj_df.astype(str).dropna(how='all')
    email_cols = identifyEmailColumns(short_obj_df)

    short_obj_df = obj_df.astype(str).sample(500, random_state=42).dropna(how='all') if len(
        obj_df) > 500 else obj_df.astype(str).dropna(how='all')
    url_cols = findURLS(short_obj_df)

    df = df.drop(email_cols, axis=1)
    df = df.drop(url_cols, axis=1)

    ##################################
    ### URL and E-Mail Engineering ###
    ##################################

    # Feature encoding for categorical features
    # num_cat = input('Are categorical columns present in the dataset? (Y/N): ').upper()
    # if num_cat == 'Y':
    #     print('Please give ONE space after every comma!')
    #     categorical = [item for item in input("Enter the name of categorical columns separated by comma"
    #                                           " or None in case there are no categorical columns : ").split(', ')]
    #     for col in categorical:
    #         if col in df.columns:
    #             pass
    #         else:
    #             print('Invalid Categorical column name')
    #
    #     lb = LabelEncoder()  # Using LabelEncoder as semantics of the column is unknown
    #     for col in categorical:
    #         df[col] = lb.fit_transform(df[col])
    # elif num_cat == 'N':
    #     pass


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

                    # # Explanation of different datetime frequencies
                    # print('\nDatetime frequency explanation:')
                    # print('M - Monthly, when the lower and upper bounds differ by atleast a month or 30 days')
                    # print('Y - Monthly, when the lower and upper bounds differ by atleast a year or 365 days')
                    # print('D - Monthly, when the lower and upper bounds differ by atleast a day or 24 hours')
                    # print('H - Monthly, when the difference between lower and upper bound is less than 24 hours.\n')

                    # freq = input(
                    #     'Enter the frequency of the data (M for Monthly, Y for Yearly, H for Hourly, D for Day) :') \
                    #     .upper()
                    # freq_dict = {'M': 30, 'Y': 365, 'D': 1, 'H': 24}

                    # Checking the frequency of the duration data
                    # if freq in ['M', 'Y', 'D', 'H']:
                    #     df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days / freq_dict[freq]
                    #     print(f'Datetimes have been converted to durations of {freq}.\n')
                    #     df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    #     df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                    #     df['duration_new'] = abs(df['duration_new'])
                    #     dur['indicator'] = 'duration_new'

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]')/(24*60)
                    print(f'Datetimes have been converted to durations of days.\n')
                    df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    df[df['duration_new'] == 0] = 1e-5  # Prohibition of time event value to be equal to 0
                    # scaler = MinMaxScaler(feature_range=(1, 100))
                    df['duration_new'] = abs(df['duration_new'])
                    # df['duration_new'] = scaler.fit_transform(df['duration_new'])
                    dur['indicator'] = 'duration_new'
                    # else:
                    #     print('Default frequency - Day')
                    #     df['duration_new'] = (df[indicator_2] - df[indicator_1]).dt.days / freq_dict[freq]
                    #     print('Datetimes have been converted to durations of Days.\n')
                    #     df.drop([indicator_2, indicator_1], axis=1, inplace=True)
                    #     df[df['duration_new'] == 0] = 1e-5
                    #     df['duration_new'] = abs(df['duration_new'])
                    #     dur['indicator'] = 'duration_new'
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
            else:
                print('Invalid duration column indicators\n')
                return None

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
    ########################
    ### Date Engineering ###
    ########################
    if censoring_type != "Interval":
        df_ = df.drop([dur['indicator']], axis=1)
    else:
        df_ = df.drop([dur['indicator']['lower'], dur['indicator']['upper']], axis=1)
    date_cols, possible_datecols = getDateColumns(df_.sample(1500, random_state=42) if len(df_) > 1500 else df_)
    if possible_datecols:
        date_cols = date_cols + possible_datecols
    print('Date Columns Impacted are:  {}'.format(date_cols))
    del df_
    if date_cols:
        try:
            DATE_DF, dropped_cols, possibleDateTimeCols = date_engineering(df[date_cols], possible_datecols)
            if dropped_cols:
                for val in dropped_cols:
                    if val in date_cols[:]:
                        date_cols.remove(val)
                    if val in possible_datecols[:]:
                        possible_datecols.remove(val)
            # print(DATE_DF.shape)
            DATE_DF.index = df.index
            df.drop(date_cols, axis=1, inplace=True)
            # df = pd.concat([df, DATE_DF], axis=1)
        except Exception as exceptionMessage:
            print('#### DATE ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            df.drop(date_cols, axis=1, inplace=True)
            DATE_DF = pd.DataFrame(None)
            date_cols = []
            possibleDateTimeCols = []
    else:
        DATE_DF = pd.DataFrame(None)
        date_cols = []
        possibleDateTimeCols = []
    ########################
    ### Date Engineering ###
    ########################
    print(df.columns)
    #########################
    #### Target Encoding ####
    #########################
    # Negative - 0, Positive - 1
    if censoring_type != 'INTERVAL':
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
            df['target'] = df['target'].fillna(0)
            target = 'target'
    #########################
    #### Target Encoding ####
    #########################

    ##########################
    #### Text Engineering ####
    ##########################
    useless_cols, disc_df, cat_num = Segregation(df.drop(target, axis=1), df[target])
    if not disc_df.empty:
        # print('after segregation:\n', disc_df.head(3))
        # disc_df = disc_df.astype('category')
        disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(disc_df.columns, axis=1)
        print('Encoding\n')
        for cols in disc_df.columns:
            imputer = CategoricalImputer()
            disc_df[cols] = imputer.fit_transform(disc_df[cols])
        print(disc_df.isnull().sum())
        # try:
        # print('We will be using Label Encoding for the task')
        # for cols in disc_df.columns:
        #     le = LabelEncoder()
        #     disc_df[cols] = le.fit_transform(disc_df[cols])
        # print('Encoded!...')

        print(disc_df.head(3))
        print('\nConcatenating-')
        df = pd.concat([df, disc_df], axis=1)
        # print(type(disc_df))
        # print(disc_df.head(3))
    else:
        # print('Else condition disc_df after segregation:')
        disc_df = pd.DataFrame()
        disc_cat = {}
    if not cat_num.empty:
        # print('after segregation:\n', cat_num.head(3))
        # disc_df = disc_df.astype('category')
        # disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(cat_num.columns, axis=1)
        print('\nConcatenating-')
        df = pd.concat([df, cat_num], axis=1)
        print('Done')
    else:
        # print('Else condition disc_df after segregation:')
        cat_num = pd.DataFrame()
        disc_cat_num = {}
    print('Printing df for debugging')
    df.head(3)
    print('Columns with no information are: ', useless_cols)
    # df.drop([disc_df.columns, cat_num.columns], axis=1, inplace=True)
    # df = pd.concat([df, disc_df], axis=1)
    # df = pd.concat([df, cat_num], axis=1)
    some_list, remove_list = findReviewColumns(df[useless_cols])
    # print("Extracting Review Columns time", end-start)
    if some_list is None:
        TEXT_DF = pd.DataFrame(None)
        lda_models = pd.DataFrame(None)
        print("No review/comment columns found")
    else:
        try:
            # print(
            #     'Respective columns will undergo text engineering and will be imputed in the function itself')
            # print('\n#### TEXT ENGINEERING RUNNING WAIT ####')
            print("Text Columns impacted are as follows : ", some_list)
            start = time.time()
            sentiment_frame = sentiment_analysis(df[some_list])
            sentiment_frame.fillna(value=0.0, inplace=True)
            # print(sentiment_frame)
            # TEXT_DF = pd.concat([df, sentiment_frame], axis=1, sort=False)
            TEXT_DF = sentiment_frame.copy()
            TEXT_DF.reset_index(drop=True, inplace=True)
            end = time.time()
            print("Sentiment time", end-start)
            start = time.time()
            new_frame = df[some_list].copy()
            new_frame.fillna(value="None", inplace=True)
            lda_models = pd.DataFrame(index=range(5), columns=['Model'])
            ind = 0
            LE = LabelEncoder()
            df_ = df.copy()
            df_[target] = LE.fit_transform(df[target])
            del df_
            # text_analytics(sentiment_frame, new_frame, class_or_Reg, y, LE)
            for col in new_frame.columns:
                topic_frame, lda_model = topicExtraction(new_frame[[col]])
                topic_frame.rename(
                    columns={0: str(col) + "_Topic"}, inplace=True)
                # print(topic_frame)
                text_analytics(new_frame, col,
                               'Classification', df[target], LE, topic_frame)
                topic_frame.reset_index(drop=True, inplace=True)
                TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
                lda_models['Model'][ind] = lda_model
                ind = ind + 1
            end = time.time()
            print("Topic time", end-start)
            df.drop(some_list, axis=1, inplace=True)
            df.drop(remove_list, axis=1, inplace=True)
            lda_models.dropna(axis=0, inplace=True)
        except Exception as e:
            print('#### TEXT ENGINEERING HAD ERRORS ####', e)
            df.drop(some_list, axis=1, inplace=True)
            if remove_list:
                df.drop(remove_list, axis=1, inplace=True)
            remove_list = []
            some_list = []
            TEXT_DF = pd.DataFrame(None)
            lda_models = pd.DataFrame(None)

        TEXT_DF.reset_index(drop=True, inplace=True)
        if not TEXT_DF.empty:
            for col in TEXT_DF.columns:
                if col.find("_Topic") != -1:
                    disc_df = pd.concat(
                        [disc_df, pd.DataFrame(TEXT_DF[col])], axis=1)
                else:
                    pass

        df = df.drop(TEXT_DF.columns, axis=1)
        df = pd.concat([df, TEXT_DF], axis=1)
    ##########################
    #### Text Engineering ####
    ##########################

    # print('Null values have been dropped\n')
    print('\nNumber of null values in each column:-')
    print(df.isnull().sum())

    print(f'\nFinal shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns.\n')

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censoring_type)

    enc_df = df.select_dtypes(include=['object'])
    df = df.drop(enc_df.columns, axis=1)
    print('Object engineered columns have been segregated!')
    display(enc_df.head(3))
    try:
        encoder = LabelEncoder()
        enc_df = enc_df.apply(encoder.fit_transform)
        print('Encoded!..')
        df = pd.concat([df, enc_df], axis=1)
        print('Merged...')
    except:
        print('Trying Target Encoding')
        for col in enc_df:
            encoder = ce.TargetEncoder(cols=col)
            enc_df[col] = encoder.fit_transform(enc_df[col], df[target])
        df = pd.concat([df, enc_df], axis=1)
        print('Merged!')

    print("\nVisualising a part of the final DataFrame")
    display(df.iloc[:5, :5])

    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

# target is the target column name
    return df, target, dur, censoring_type, orig, dur_cols, encoder


def lead_input(orig, censor, dur, target, dur_cols, encoder):
    path = input("Enter the path of the dataset: ")
    try:
        df, _ = importFile(path)
        # display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None
    df = df[(df.columns) & (orig.columns)]
    df[target] = 0
    df = numeric_engineering(df)

    lat, lon, lat_lon_cols = findLatLong(df)
    print("Latitude Columns Impacted are: ", lat)
    print("Longitude Columns Impacted are:", lon)
    print("Latitude-Longitude Columns Impacted are:", lat_lon_cols)

    if (lat and lon) or lat_lon_cols:
        try:
            LAT_LONG_DF = latlongEngineering(df, lat, lon, lat_lon_cols)
            # print(LAT_LONG_DF)
            # print(LAT_LONG_DF.shape)
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
        except Exception as exceptionMessage:
            print('#### LAT-LONG ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
            LAT_LONG_DF = pd.DataFrame(None)
            lat = []
            lon = []
            lat_lon_cols = []
    else:
        print("No Latitude or Longitude Columns found")
        LAT_LONG_DF = pd.DataFrame(None)
        lat = []
        lon = []
        lat_lon_cols = []
    if len(LAT_LONG_DF):
        df = pd.concat([df, LAT_LONG_DF], axis=1)

    print(df.columns)
    ##############################
    #### LAT-LONG Engineering ####
    ##############################

    ##################################
    ### URL and E-Mail Engineering ###
    #################################
    obj_df = df.select_dtypes('object')  # Sending in only object dtype columns
    short_obj_df = obj_df.astype(str).sample(3000, random_state=42).dropna(how='all') if len(
        obj_df) > 3000 else obj_df.astype(str).dropna(how='all')
    email_cols = identifyEmailColumns(short_obj_df)

    short_obj_df = obj_df.astype(str).sample(500, random_state=42).dropna(how='all') if len(
        obj_df) > 500 else obj_df.astype(str).dropna(how='all')
    url_cols = findURLS(short_obj_df)

    df = df.drop(email_cols, axis=1)
    df = df.drop(url_cols, axis=1)

    print(df.columns)
    ##################################
    ### URL and E-Mail Engineering ###
    ##################################

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

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]')/(24*60)
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
    print(df.columns)
    ########################
    ### Date Engineering ###
    ########################
    if censor != "INTERVAL":
        df_ = df.drop([dur['indicator']], axis=1)
    else:
        df_ = df.drop([dur['indicator']['lower'], dur['indicator']['upper']], axis=1)
    date_cols, possible_datecols = getDateColumns(df_.sample(1500, random_state=42) if len(df_) > 1500 else df_)
    if possible_datecols:
        date_cols = date_cols + possible_datecols
    print('Date Columns Impacted are:  {}'.format(date_cols))
    del df_
    if date_cols:
        try:
            DATE_DF, dropped_cols, possibleDateTimeCols = date_engineering(df[date_cols], possible_datecols)
            if dropped_cols:
                for val in dropped_cols:
                    if val in date_cols[:]:
                        date_cols.remove(val)
                    if val in possible_datecols[:]:
                        possible_datecols.remove(val)
            # print(DATE_DF.shape)
            DATE_DF.index = df.index
            df.drop(date_cols, axis=1, inplace=True)
            # df = pd.concat([df, DATE_DF], axis=1)
        except Exception as exceptionMessage:
            print('#### DATE ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            df.drop(date_cols, axis=1, inplace=True)
            DATE_DF = pd.DataFrame(None)
            date_cols = []
            possibleDateTimeCols = []
    else:
        DATE_DF = pd.DataFrame(None)
        date_cols = []
        possibleDateTimeCols = []
    ########################
    ### Date Engineering ###
    ########################
    ###############################
    ########## Censoring ##########
    ###############################


    ##########################
    #### Text Engineering ####
    ##########################
    useless_cols, disc_df, cat_num = Segregation(df.drop(target, axis=1), df[target])
    if not disc_df.empty:
        # print('after segregation:\n', disc_df.head(3))
        # disc_df = disc_df.astype('category')
        disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(disc_df.columns, axis=1)
        print('Encoding\n')
        for cols in disc_df.columns:
            imputer = CategoricalImputer()
            disc_df[cols] = imputer.fit_transform(disc_df[cols])
        print(disc_df.isnull().sum())


        print(disc_df.head(3))
        print('\nConcatenating-')
        df = pd.concat([df, disc_df], axis=1)
        # print(type(disc_df))
        # print(disc_df.head(3))
    else:
        # print('Else condition disc_df after segregation:')
        disc_df = pd.DataFrame()
        disc_cat = {}
    if not cat_num.empty:
        # print('after segregation:\n', cat_num.head(3))
        # disc_df = disc_df.astype('category')
        # disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(cat_num.columns, axis=1)
        print('\nConcatenating-')
        df = pd.concat([df, cat_num], axis=1)
        print('Done')
    else:
        # print('Else condition disc_df after segregation:')
        cat_num = pd.DataFrame()
        disc_cat_num = {}
    print('Printing df for debugging')
    df.head(3)
    print('Columns with no information are: ', useless_cols)

    some_list, remove_list = findReviewColumns(df[useless_cols])
    # print("Extracting Review Columns time", end-start)
    if some_list is None:
        TEXT_DF = pd.DataFrame(None)
        lda_models = pd.DataFrame(None)
        print("No review/comment columns found")
    else:
        try:
            # print(
            #     'Respective columns will undergo text engineering and will be imputed in the function itself')
            # print('\n#### TEXT ENGINEERING RUNNING WAIT ####')
            print("Text Columns impacted are as follows : ", some_list)
            start = time.time()
            sentiment_frame = sentiment_analysis(df[some_list])
            sentiment_frame.fillna(value=0.0, inplace=True)
            # print(sentiment_frame)
            # TEXT_DF = pd.concat([df, sentiment_frame], axis=1, sort=False)
            TEXT_DF = sentiment_frame.copy()
            TEXT_DF.reset_index(drop=True, inplace=True)
            end = time.time()
            print("Sentiment time", end - start)
            start = time.time()
            new_frame = df[some_list].copy()
            new_frame.fillna(value="None", inplace=True)
            lda_models = pd.DataFrame(index=range(5), columns=['Model'])
            ind = 0
            LE = LabelEncoder()
            df_ = df.copy()
            df_[target] = LE.fit_transform(df[target])
            del df_
            # text_analytics(sentiment_frame, new_frame, class_or_Reg, y, LE)
            for col in new_frame.columns:
                topic_frame, lda_model = topicExtraction(new_frame[[col]])
                topic_frame.rename(
                    columns={0: str(col) + "_Topic"}, inplace=True)
                # print(topic_frame)
                text_analytics(new_frame, col,
                               'Classification', df[target], LE, topic_frame)
                topic_frame.reset_index(drop=True, inplace=True)
                TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
                lda_models['Model'][ind] = lda_model
                ind = ind + 1
            end = time.time()
            print("Topic time", end - start)
            df.drop(some_list, axis=1, inplace=True)
            df.drop(remove_list, axis=1, inplace=True)
            lda_models.dropna(axis=0, inplace=True)
        except Exception as e:
            print('#### TEXT ENGINEERING HAD ERRORS ####', e)
            df.drop(some_list, axis=1, inplace=True)
            if remove_list:
                df.drop(remove_list, axis=1, inplace=True)
            remove_list = []
            some_list = []
            TEXT_DF = pd.DataFrame(None)
            lda_models = pd.DataFrame(None)

        TEXT_DF.reset_index(drop=True, inplace=True)
        if not TEXT_DF.empty:
            for col in TEXT_DF.columns:
                if col.find("_Topic") != -1:
                    disc_df = pd.concat(
                        [disc_df, pd.DataFrame(TEXT_DF[col])], axis=1)
                else:
                    pass

        df = df.drop(TEXT_DF.columns, axis=1)
        df = pd.concat([df, TEXT_DF], axis=1)
    ##########################
    #### Text Engineering ####
    ##########################

    enc_df = df.select_dtypes(include=['object'])
    df = df.drop(enc_df.columns, axis=1)
    print('Object engineered columns have been segregated!')
    display(enc_df.head(3))
    try:
        # le = LabelEncoder()
        enc_df = enc_df.apply(encoder.transform)
        print('Encoded!..')
        df = pd.concat([df, enc_df], axis=1)
        print('Merged...')
    except:
        print('Trying Target Encoding')
        df['new_target'] = np.random.choice([0,1], len(df))
        for col in enc_df:
            encoder = ce.TargetEncoder(cols=col)
            enc_df[col] = encoder.fit_transform(enc_df[col], df['new_target'])
        df = pd.concat([df, enc_df], axis=1)
        print('Merged!')
    df = df.drop(['new_target'], axis=1)
    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

    print("\nVisualising a part of the final DataFrame")
    display(df.iloc[:5, :5])

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censor)
    # df = df.drop([target], axis=1)
    return df, dur, censor, target


def lead_input_target(orig, censor, dur, target, dur_cols, encoder):
    path = input("Enter the path of the dataset: ")
    try:
        df, _ = importFile(path)
        # display(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None
    df = df[(df.columns) & (orig.columns)]
    # df[target] = 0
    df = numeric_engineering(df)

    lat, lon, lat_lon_cols = findLatLong(df)
    print("Latitude Columns Impacted are: ", lat)
    print("Longitude Columns Impacted are:", lon)
    print("Latitude-Longitude Columns Impacted are:", lat_lon_cols)

    if (lat and lon) or lat_lon_cols:
        try:
            LAT_LONG_DF = latlongEngineering(df, lat, lon, lat_lon_cols)
            # print(LAT_LONG_DF)
            # print(LAT_LONG_DF.shape)
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
        except Exception as exceptionMessage:
            print('#### LAT-LONG ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            if lat:
                df.drop(lat, axis=1, inplace=True)
            if lon:
                df.drop(lon, axis=1, inplace=True)
            if lat_lon_cols:
                df.drop(lat_lon_cols, axis=1, inplace=True)
            LAT_LONG_DF = pd.DataFrame(None)
            lat = []
            lon = []
            lat_lon_cols = []
    else:
        print("No Latitude or Longitude Columns found")
        LAT_LONG_DF = pd.DataFrame(None)
        lat = []
        lon = []
        lat_lon_cols = []
    if len(LAT_LONG_DF):
        df = pd.concat([df, LAT_LONG_DF], axis=1)

    print(df.columns)
    ##############################
    #### LAT-LONG Engineering ####
    ##############################

    ##################################
    ### URL and E-Mail Engineering ###
    #################################
    obj_df = df.select_dtypes('object')  # Sending in only object dtype columns
    short_obj_df = obj_df.astype(str).sample(3000, random_state=42).dropna(how='all') if len(
        obj_df) > 3000 else obj_df.astype(str).dropna(how='all')
    email_cols = identifyEmailColumns(short_obj_df)

    short_obj_df = obj_df.astype(str).sample(500, random_state=42).dropna(how='all') if len(
        obj_df) > 500 else obj_df.astype(str).dropna(how='all')
    url_cols = findURLS(short_obj_df)

    df = df.drop(email_cols, axis=1)
    df = df.drop(url_cols, axis=1)

    print(df.columns)
    ##################################
    ### URL and E-Mail Engineering ###
    ##################################

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

                    df['duration_new'] = (df[indicator_2] - df[indicator_1]).astype('timedelta64[m]')/(24*60)
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
    print(df.columns)
    ########################
    ### Date Engineering ###
    ########################
    if censor != "INTERVAL":
        df_ = df.drop([dur['indicator']], axis=1)
    else:
        df_ = df.drop([dur['indicator']['lower'], dur['indicator']['upper']], axis=1)
    date_cols, possible_datecols = getDateColumns(df_.sample(1500, random_state=42) if len(df_) > 1500 else df_)
    if possible_datecols:
        date_cols = date_cols + possible_datecols
    print('Date Columns Impacted are:  {}'.format(date_cols))
    del df_
    if date_cols:
        try:
            DATE_DF, dropped_cols, possibleDateTimeCols = date_engineering(df[date_cols], possible_datecols)
            if dropped_cols:
                for val in dropped_cols:
                    if val in date_cols[:]:
                        date_cols.remove(val)
                    if val in possible_datecols[:]:
                        possible_datecols.remove(val)
            # print(DATE_DF.shape)
            DATE_DF.index = df.index
            df.drop(date_cols, axis=1, inplace=True)
            # df = pd.concat([df, DATE_DF], axis=1)
        except Exception as exceptionMessage:
            print('#### DATE ENGINEERING HAD ERRORS ####')
            print('Exception message is {}'.format(exceptionMessage))
            df.drop(date_cols, axis=1, inplace=True)
            DATE_DF = pd.DataFrame(None)
            date_cols = []
            possibleDateTimeCols = []
    else:
        DATE_DF = pd.DataFrame(None)
        date_cols = []
        possibleDateTimeCols = []
    ########################
    ### Date Engineering ###
    ########################
    ###############################
    ########## Censoring ##########
    ###############################
    if censor != 'INTERVAL':
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
            df['target'] = df['target'].fillna(0)
            target = 'target'

    ##########################
    #### Text Engineering ####
    ##########################
    useless_cols, disc_df, cat_num = Segregation(df.drop(target, axis=1), df[target])
    if not disc_df.empty:
        # print('after segregation:\n', disc_df.head(3))
        # disc_df = disc_df.astype('category')
        disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(disc_df.columns, axis=1)
        print('Encoding\n')
        for cols in disc_df.columns:
            imputer = CategoricalImputer()
            disc_df[cols] = imputer.fit_transform(disc_df[cols])
        print(disc_df.isnull().sum())
        # try:
        # print('We will be using Label Encoding for the task')
        # for cols in disc_df.columns:
        #     le = LabelEncoder()
        #     disc_df[cols] = le.fit_transform(disc_df[cols])
        # print('Encoded!...')

        print(disc_df.head(3))
        print('\nConcatenating-')
        df = pd.concat([df, disc_df], axis=1)
        # print(type(disc_df))
        # print(disc_df.head(3))
    else:
        # print('Else condition disc_df after segregation:')
        disc_df = pd.DataFrame()
        disc_cat = {}
    if not cat_num.empty:
        # print('after segregation:\n', cat_num.head(3))
        # disc_df = disc_df.astype('category')
        # disc_cat = {}
        # for column in disc_df:
        #     disc_cat[column] = disc_df[column].cat.categories
        print('Dropping-')
        df = df.drop(cat_num.columns, axis=1)
        print('\nConcatenating-')
        df = pd.concat([df, cat_num], axis=1)
        print('Done')
    else:
        # print('Else condition disc_df after segregation:')
        cat_num = pd.DataFrame()
        disc_cat_num = {}
    print('Printing df for debugging')
    df.head(3)
    print('Columns with no information are: ', useless_cols)

    some_list, remove_list = findReviewColumns(df[useless_cols])
    # print("Extracting Review Columns time", end-start)
    if some_list is None:
        TEXT_DF = pd.DataFrame(None)
        lda_models = pd.DataFrame(None)
        print("No review/comment columns found")
    else:
        try:
            # print(
            #     'Respective columns will undergo text engineering and will be imputed in the function itself')
            # print('\n#### TEXT ENGINEERING RUNNING WAIT ####')
            print("Text Columns impacted are as follows : ", some_list)
            start = time.time()
            sentiment_frame = sentiment_analysis(df[some_list])
            sentiment_frame.fillna(value=0.0, inplace=True)
            # print(sentiment_frame)
            # TEXT_DF = pd.concat([df, sentiment_frame], axis=1, sort=False)
            TEXT_DF = sentiment_frame.copy()
            TEXT_DF.reset_index(drop=True, inplace=True)
            end = time.time()
            print("Sentiment time", end - start)
            start = time.time()
            new_frame = df[some_list].copy()
            new_frame.fillna(value="None", inplace=True)
            lda_models = pd.DataFrame(index=range(5), columns=['Model'])
            ind = 0
            LE = LabelEncoder()
            df_ = df.copy()
            df_[target] = LE.fit_transform(df[target])
            del df_
            # text_analytics(sentiment_frame, new_frame, class_or_Reg, y, LE)
            for col in new_frame.columns:
                topic_frame, lda_model = topicExtraction(new_frame[[col]])
                topic_frame.rename(
                    columns={0: str(col) + "_Topic"}, inplace=True)
                # print(topic_frame)
                text_analytics(new_frame, col,
                               'Classification', df[target], LE, topic_frame)
                topic_frame.reset_index(drop=True, inplace=True)
                TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
                lda_models['Model'][ind] = lda_model
                ind = ind + 1
            end = time.time()
            print("Topic time", end - start)
            df.drop(some_list, axis=1, inplace=True)
            df.drop(remove_list, axis=1, inplace=True)
            lda_models.dropna(axis=0, inplace=True)
        except Exception as e:
            print('#### TEXT ENGINEERING HAD ERRORS ####', e)
            df.drop(some_list, axis=1, inplace=True)
            if remove_list:
                df.drop(remove_list, axis=1, inplace=True)
            remove_list = []
            some_list = []
            TEXT_DF = pd.DataFrame(None)
            lda_models = pd.DataFrame(None)

        TEXT_DF.reset_index(drop=True, inplace=True)
        if not TEXT_DF.empty:
            for col in TEXT_DF.columns:
                if col.find("_Topic") != -1:
                    disc_df = pd.concat(
                        [disc_df, pd.DataFrame(TEXT_DF[col])], axis=1)
                else:
                    pass

        df = df.drop(TEXT_DF.columns, axis=1)
        df = pd.concat([df, TEXT_DF], axis=1)
    ##########################
    #### Text Engineering ####
    ##########################

    enc_df = df.select_dtypes(include=['object'])
    df = df.drop(enc_df.columns, axis=1)
    print('Object engineered columns have been segregated!')
    display(enc_df.head(3))
    try:
        # le = LabelEncoder()
        enc_df = enc_df.apply(encoder.transform)
        print('Encoded!..')
        df = pd.concat([df, enc_df], axis=1)
        print('Merged...')
    except:
        print('Trying Target Encoding')
        df[target] = np.random.choice([0,1], len(df))
        for col in enc_df:
            encoder = ce.TargetEncoder(cols=col)
            enc_df[col] = encoder.fit_transform(enc_df[col], df[target])
        df = pd.concat([df, enc_df], axis=1)
        print('Merged!')
    # df = df.drop(['new_target'], axis=1)
    print('Final - Null values in every column')
    print(df.isnull().sum())
    df = df.fillna(0)

    print("\nVisualising a part of the final DataFrame")
    display(df.iloc[:5, :5])

    print("Exploring the duration series present in the DataFrame-")
    dataExploration(df, dur, censor)
    # df = df.drop([target], axis=1)
    return df, dur, censor, target
