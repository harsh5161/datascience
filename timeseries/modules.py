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


def dataExploration(df):
    # Exploratory DataAnalysis
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
