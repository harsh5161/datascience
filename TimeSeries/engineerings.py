import pandas as pd
import numpy as np
import time
import holidays

############################################
############## NUMERIC ENGINEERING #########
############################################

def numeric_engineering(df):
    start = time.time()

    def returnMoney(col):
        # Remove Commas from currencies
        try:
            return pd.to_numeric(col.str.replace([',','$','€','£','¥'],''))
        except:
            return col

    obj_columns= list(df.dtypes[df.dtypes == np.object].index)
    # print(f'object type columns are {obj_columns}')
    print('\t\t stripping spaces, symbols, and lower casing all entries')
    df[obj_columns]=df[obj_columns].apply(lambda x: x.astype(str).str.strip(' %$€£¥+').str.lower())
    print('done ...')
    print('\t\t Replacing empty and invalid strings')
    possible_empties = ['-','n/a','na','nan','nil',np.inf,-np.inf,'']
    df[obj_columns]=df[obj_columns].replace(possible_empties,np.nan)
    print('done ...')
    print('\t\t Replacing commas if present in Currencies')
    df[obj_columns]=df[obj_columns].apply(lambda x:returnMoney(x))
    print('done ...')
    obj_columns= list(df.dtypes[df.dtypes == np.object].index)
    df1 = df[obj_columns].copy()
    print('\t\t Finding Numeric Columns')
    df1 = df1.apply(lambda x : pd.to_numeric(x,errors='coerce'))
    df1.dropna(axis=1,thresh = 0.65*len(df),inplace=True)
    new_num_cols = df1.columns
    df[new_num_cols] = df[new_num_cols].apply(lambda x : pd.to_numeric(x,errors='coerce'))
    print('done ...')

    for i in df.columns :
        print(f'\t\t   {i} is of type {df[i].dtypes}')

    # # End of Testing codes
    end = time.time();print('Numeric Engineering time taken:',end - start);print('\n')
    return(df)

############################################
############## DATE ENGINEERING ############
############################################

def getDateColumns(df,withPossibilies=0):
    '''
    This method Identifies all columns with 'DATE' data by maximizing out the possibilities
    '''
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    # First get all non-numerical Columns
    non_numeric_cols = df.select_dtypes('object')
    # This dictionary stores possibilities of a column containing 'DATES' based on the name of the column
    Possibility = {}
    for column in non_numeric_cols:
        if 'date' in column.lower():
            Possibility[column] = int(len(df)*0.1)
        else:
            Possibility[column] = 0
        for entry in df[column]:           # ITERATE THROUGH EVERY ENTRY AND TRY SPLITTING THE VALUE AND INCREMENT/DECREMENT POSSIBILITY
            try:                                                                      # USING EXCEPTION HANDLING
                if len(entry.split('/')) == 3 or len(entry.split('-')) == 3 or len(entry.split(':')) == 3:
                    Possibility[column] += 1
                    for month in months:
                        if month in entry.lower():
                            Possibility[column] += 1
                else:
                    Possibility[column] -= 1
            except:
                Possibility[column] -= 1
      # This contains the final DATE Columns
    DATE_COLUMNS = []
    for key,value in Possibility.items():
        if value > 0.8 * len(df):          # IF THE POSSIBILITY OF THE COLUMN IN GREATER THAN 1, THEN IT IS DEFINITELY A 'DATE COLUMN'
            DATE_COLUMNS.append(key)
            
    # to find missed date columns    
    def finddate(entry):    # function to find the presence of a month in an entry     
        a=0
        for month in months:
            if month in str(entry).lower():
                a=1
        return a
    
    Y=non_numeric_cols
    Y=Y.drop(DATE_COLUMNS, axis =1) 
    Possible_date_col=[]
    for col in Y.columns:
        a=Y[col].apply(finddate) # returns a series where value is one if the entry has a month in it
        if sum(a)>0.8*len(Y[col]):   # if there is a name of a month in atleast 80% entries of the column
            Possible_date_col.append(col)
            
    if not withPossibilies:
        return DATE_COLUMNS,Possible_date_col
    else:
        return DATE_COLUMNS,Possible_date_col,Possibility

############################################
############## TIME ENGINEERING ############
############################################

def time_engineering(props):
    print('\n#### ENTERING TIME ENGINEERING ####')
    print('\nUNPACKING PROPS')
    # Unpacking Props parameter
    df = props['df']
    primaryDate = props['info']['PrimaryDate']
    
    print('\nSorting DataFrame according to Primary Date Column')
    df.sort_values(by=[primaryDate],inplace=True)
    df.reset_index(inplace=True,drop=True)
    
    print('\nDropping rows with empty Date')
    df.dropna(subset=[primaryDate],inplace=True)
    
    print('\nFinding a set of all spaces present in the primary Date Column\n')
    spaces = []
    for i in range(len(df)-1):
        # print('Checking index {}'.format(i))
        spaces.append((df.loc[i+1][primaryDate]-df.loc[i][primaryDate]))
    uniqueSpaces = set(spaces)
    # print(uniqueSpaces)
    if len(uniqueSpaces) == 1:
        print('The Data is equally spaced!')
    else:
        spacesDictionary = {}
        for space in uniqueSpaces:
            spacesDictionary[space] = spaces.count(space)
        print('The spaces found are (space:frequency format): ')
        print(spacesDictionary)
        keys = list(spacesDictionary.keys())
        values = list(spacesDictionary.values())
        bestSpace = keys[values.index(max(values))]
        print('\nThe best space found is : {}'.format(bestSpace))    
        print('\nResetting index!')
        df.reset_index(drop=True,inplace=True)
        
        
    # Exploiting Date Column
    print('\nCreating exogenous Date Variables from Primary Date Column')
    print('Obtaining Year,Quarter,Month,Week,Day of Week,Day,Hour,Minute,Seconds,Weekend or not')
    df['Year'] = df[primaryDate].dt.year
    df['quarter'] = df[primaryDate].dt.quarter
    df['month'] = df[primaryDate].dt.month
    df['Week'] = df[primaryDate].dt.week
    df['day of Week'] = df[primaryDate].dt.dayofweek
    df['Day'] = df[primaryDate].dt.day
    df['Hour'] = df[primaryDate].dt.hour
    df['Minute'] = df[primaryDate].dt.minute
    df['Seconds'] = df[primaryDate].dt.second
    df['Weekend_or_not'] = df['day of Week'].apply(lambda x: 1 if x in [5,6,0,1] else 0)
    df['BusinessHrs_or_not'] = df['Hour'].apply(lambda x: 1 if 9<=x<=17 else 0)
    
    quarters = {1:'Q1',2:'Q2',3:'Q3',4:'Q4'}
    months = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',
              9:'September',10:'October',11:'November',12:'December'}
    days_of_week = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',
                    6:'Sunday'}
    
    print('Creating English levels for Visualization purposes!')
    def getQuarter(entry):
        try:
            return quarters[entry]
        except:
            return np.nan
    
    def getMonth(entry):
        try:
            return months[entry]
        except:
            return np.nan
        
    def getDayOfWeek(entry):
        try:
            return days_of_week[entry]
        except:
            return np.nan

    df['Quarter'] = df['quarter'].apply(getQuarter)
    df['Month'] = df['month'].apply(getMonth)
    df['Day of Week'] = df['day of Week'].apply(getDayOfWeek)
    
    englishLevelsCols = ['Quarter','Month','Day of Week']
    props['EnglishCols'] = englishLevelsCols
    
    print('\nCreating a dictionary of holidays for every year present in the data')
    holidayDictionary = {}
    uniqueYears = df['Year'].unique().tolist()
    for year in uniqueYears:
        holidayDictionary[year] = pd.to_datetime(list(holidays.US(years=year).keys()))
    
    def nearHol(entry):
        hols_list = holidayDictionary[entry.year]
        for holiday in hols_list:
            if abs((entry-holiday).days) <= 5:
                return 1
        else:
            return 0
    
    df['Near Holiday'] = df[primaryDate].apply(nearHol)
        
    print('\nThings yet to be done')
    print('1. Equally Space Data')

    # Removing columns that have absolutely one level
    print('\nRemoving columns with only one level')
    for column in df:
        if df[column].nunique() == 1:
            print('Column \'{}\' will be dropped since it has 1 level'.format(column))
            df.drop(column,axis=1,inplace=True)

    print('\nPrinting DataFrame Head\n')
    print(df.head())

    print('\n#### PACKING/UPDATING and RETURNING PROPS ####\n')
    props['df'] = df
    print('#### TIME ENGINEERING DONE ####')
    return props