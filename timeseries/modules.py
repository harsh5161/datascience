from modelling import Modelling
from userInputs import importFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.tsa.api as smt
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
from plots import bar_metrics
from random import randint
seed = 42
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
# You want to call this script separately so that we take in a large dataset input by the user and push it out of memory after extracting the two columnss that we need


def userInputs():
    path = input("Enter the path of the dataset :")
    try:
        df, _ = importFile(path)
        print(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None, None
    date_column = input("Enter the Date Column :")
    # format = '%Y-%m-%d %H:%M:%S'
    try:
        df['Datetime'] = pd.to_datetime(df[date_column])  # ,format = format
        df.drop(date_column, axis=1, inplace=True)
        df.set_index(pd.DatetimeIndex(df['Datetime']), inplace=True)
        df.sort_index(inplace=True)
        df.drop('Datetime', axis=1, inplace=True)
        print(df.head())
    except Exception as e:
        print(
            f"Date Column could not be found or Date Column could not be set as Index : {e}")
        return None, None
    print("Exploring the different series' present in the DataFrame")
    dataExploration(df)

    # According to the EDA done above, now we decide which column is going to be our timeseries.
    try:
        target = input("Enter the Target Column :")
        plt.figure(num=None, figsize=(40, 15), dpi=80,
                   facecolor='w', edgecolor='k')
        plt.title(f'{target}', fontsize=30)

        plt.plot(df[target])
        plt.show()
        # plt.savefig(f'{target}.png')
    except:
        print("Target entered does not exist in DataFrame or couldn't be plotted : Please check spelling ")
        return None

    result_df = pd.DataFrame(df[target].copy())
    try:
        predictors = input(
            "Do you want to add any other column as a predictor in the timeseries? [Separate by commas if you want to add multiple predictors || Press Enter to Continue without adding Predictors] ").split(",")
        for col in predictors:
            result_df[col] = df[col]
    except Exception as e:
        print(f"Predictor Could not be added : {e}")

    print("Visualising the final DataFrame")
    print(result_df.head(10))
    return result_df, target
# Exploratory DataAnalysis


def dataExploration(df):
    values = df.values
    groups = [i for i in range(len(df.columns.to_list())-1)]
    print(f'Groups are : {groups}')
    i = 1
    plt.figure(figsize=(10, 10))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


# takes an input of a dataframe with one column and of index DatetimeIndex type
def seriesIdentifier(df):
    formats = []
    ind = df.index
    years = pd.Series(ind.year, name='years', index=df.index)
    # print(years)
    months = pd.Series(ind.month, name='months', index=df.index)
    # print(months)
    days = pd.Series(ind.day, name='days', index=df.index)
    # print(days)
    df['years'] = years
    df['months'] = months
    df['days'] = days
#     print(df.head())

    if years.nunique() > 1:
        formats.append('Yearly')
    months_obj = df.groupby('years')['months'].nunique()
    if months_obj.mean() > 10.0:
        formats.append('Monthly')
    days_obj = df.groupby('years')['days'].count()
    if days_obj.mean() > 250.0:
        formats.append('Daily')

    return formats


def processIdentifier(df):
    process_list = seriesIdentifier(df)
    perform_list = []
    print(
        f"The various analysis' that can performed on the data without any resampling are \n{process_list}")
    print("Hint: Performing Resampling will decrease the size of your dataset, the higher the degree of resampling;\nThe smaller the dataset. For optimal performance only choose to resample when you have sufficient data or choose to perform the forecast in the period that the data was originally collected")
    if 'Yearly' in process_list:
        q = input(
            'Do you want to perform Yearly forecasts[y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('Y')
            return perform_list
    if 'Monthly' in process_list:
        q = input(
            'Do you want to perform Monthly forecasts[y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('M')
        q = input(
            'Do you also want to perform Quarterly forecasts with resampled data [y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('RQ')
        if len(perform_list) > 0:
            return perform_list
    if 'Daily' in process_list:
        q = input(
            'Do you want to perform Daily forecasts[y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('D')
        q = input(
            'Do you want to perform Weekly forecasts with resampled data [y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('RW')
        if 'M' not in perform_list:
            q = input(
                'Do you want to perform Monthly forecasts with resampled data [y if Yes, anything else if no]')
            if 'y' in q:
                perform_list.append('RM')
        if len(perform_list) > 0:
            return perform_list

    print(
        f"Various processes can be applied onto the data : \n {perform_list}")
    return perform_list

# Resampling the dataframes if necessary
# Yearly, Monthly and Daily forecasts if possible will not require any resampling to preserve information.


def dataResampler(df, perform_list):
    resampled_data = {}
    if 'RQ' in perform_list:
        quarter_df = df.resample('Q').sum()
        quarter_df.drop(['years', 'months', 'days'], axis=1, inplace=True)
        resampled_data["Quarter"] = quarter_df
        print(
            f'Quarterly Resampling done, engineered dataframe size {quarter_df.shape}')
    if 'RW' in perform_list:
        weekly_df = df.resample('W').sum()
        weekly_df.drop(['years', 'months', 'days'], axis=1, inplace=True)
        resampled_data["Week"] = weekly_df
        print(
            f'Weekly Resampling done, engineered dataframe size {weekly_df.shape}')
    if 'RM' in perform_list:
        monthly_df = df.resample('M').sum()
        monthly_df.drop(['years', 'months', 'days'], axis=1, inplace=True)
        resampled_data["Month"] = monthly_df
        print(
            f'Monthly Resampling done, engineered dataframe size {monthly_df.shape}')
    return resampled_data


def seasonalDecompose(series, period):
    mpl.rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80,
               facecolor='w', edgecolor='k')
    # Logic to determine freq need to think (?)
    result = seasonal_decompose(
        series, model='multiplicative', period=period)  # freq? period?
    plt.show(result)

# Now we look for white noise by looking at the plots below, if the mean and std histograms are normally distributed then there is white noise present and we cannot predict that part as it is random.
# If our time series has white noise this will mean we can't predict that component of the series (as is random) and we should aim to produce a model with errors close to this white noise.
# things to look for
# Is original series normally distributed?
# Is the std deviation normally distributed?
# Is the mean over time constant?
# Is the autocorrelation plot reaching minima soon and continuing till the end?
# If Yes, then the series is stationary, else it is non stationary


def stationaryNormalityPlots(series, lags, rolling):
    fig = plt.figure(figsize=(12, 7))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    ac_ax = plt.subplot2grid(layout, (1, 0))
    hist_std_ax = plt.subplot2grid(layout, (0, 1))
    mean_ax = plt.subplot2grid(layout, (1, 1))

    series.hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")

    plot_acf(series, lags=lags, ax=ac_ax)
    ac_ax.set_title("Autocorrelation")

    mm = series.rolling(rolling).std()
    mm.hist(ax=hist_std_ax)
    hist_std_ax.set_title("Standard deviation histogram")

    mm = series.rolling(lags).mean()
    mm.plot(ax=mean_ax)
    mean_ax.set_title("Mean over time")
    plt.show()


def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=(15, 10))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        mean_std_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        hypothesis_result = "Stationary" if p_value <= 0.05 else "Non-Stationary"
        ts_ax.set_title(
            'Time Series stationary analysis Plots\n Dickey-Fuller: p={0:.5f} Result: {1}'.format(p_value, hypothesis_result))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

        rolmean = y.rolling(window=12).mean()
        rolstd = y.rolling(window=12).std()

        # Plot rolling statistics:
        orig = plt.plot(y, label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')

    plt.show()


def findTrainSize(df):
    return int(len(df)*0.90)


def featureEngineering(df, target=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['seconds'] = df['date'].dt.second
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    X = df.drop(['date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# Change the logic for this function and also check the logic of resample


def testPlot(y_test, predictionsDict):
    plt.plot(y_test.values, color='black', label='Original')
    cmap = get_cmap(50)
    for key, value in predictionsDict.items():
        yhat = value
        plt.plot(yhat, color=cmap(randint(0, 50)), label=f'{key}')
    plt.legend()
    plt.show()


def modellingInit(df, target, resultsDict, predictionsDict):
    # X = df.values
    train_size = findTrainSize(df)
    split_date = df.index[train_size]
    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print(
        f"We have {len(df_training)} days of training data and {len(df_test)} days of testing data ")
    X_train_df, y_train = featureEngineering(df_training, target=target)
    X_test_df, y_test = featureEngineering(df_test, target=target)
    scaler = StandardScaler()
    scaler.fit(X_train_df)  # No cheating, never scale on the training+test!
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)

    modelling_obj = Modelling(X_train_df, X_test_df,
                              y_train, y_test, resultsDict, predictionsDict)
    modelling_obj.modeller()
    testPlot(y_test, predictionsDict)
    bar_metrics(resultsDict)
