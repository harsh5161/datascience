# You want to call this script separately so that we take in a large dataset input by the user and push it out of memory after extracting the two columnss that we need
def userInputs():
    path = input("Enter the path of the dataset :")
    try:
        df, _ = importFile(path)
        print(df.head(5))
    except:
        print("Import Error: Please try importing an appropriate dataset")
        return None
    date_column = input("Enter the Date Column :")
    # format = '%Y-%m-%d %H:%M:%S'
    try:
        df['Datetime'] = pd.to_datetime(df[date_column])  # ,format = format
        df.drop(date_column, axis=1, inplace=True)
        df.set_index(pd.DatetimeIndex(df['Datetime']), inplace=True)
        df.sort_index(inplace=True)
        df.drop('Datetime', axis=1, inplace=True)
        print(df.head())
    except:
        print("Date Column could not be found or Date Column could not be set as Index")
        return None
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

    df = pd.DataFrame(df[target].copy())
    print("Visualising the final DataFrame")
    print(df.head(10))
    return df
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
    if 'Monthly' in process_list:
        q = input(
            'Do you want to perform Monthly forecasts[y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('M')
        q = input(
            'Do you want to perform Quarterly forecasts with resample [y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('RQ')
    if 'Daily' in process_list:
        q = input(
            'Do you want to perform Daily forecasts[y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('D')
        q = input(
            'Do you want to perform Weekly forecasts with resample [y if Yes, anything else if no]')
        if 'y' in q:
            perform_list.append('RW')
        if 'M' not in perform_list:
            q = input(
                'Do you want to perform Monthly forecasts with resample [y if Yes, anything else if no]')
            if 'y' in q:
                perform_list.append('RM')

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
        resampled_data["Weekly"] = weekly_df
        print(
            f'Weekly Resampling done, engineered dataframe size {weekly_df.shape}')
    if 'RM' in perform_list:
        monthly_df = df.resample('M').sum()
        monthly_df.drop(['years', 'months', 'days'], axis=1, inplace=True)
        resampled_data["Monthly"] = monthly_df
        print(
            f'Monthly Resampling done, engineered dataframe size {monthly_df.shape}')
    return resampled_data
