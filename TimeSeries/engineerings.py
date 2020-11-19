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
    print(f'\t\t stripping spaces, symbols, and lower casing all entries')
    df[obj_columns]=df[obj_columns].apply(lambda x: x.astype(str).str.strip(' %$€£¥+').str.lower())
    print('done ...')
    print(f'\t\t Replacing empty and invalid strings')
    possible_empties = ['-','n/a','na','nan','nil',np.inf,-np.inf,'']
    df[obj_columns]=df[obj_columns].replace(possible_empties,np.nan)
    print('done ...')
    print(f'\t\t Replacing commas if present in Currencies')
    df[obj_columns]=df[obj_columns].apply(lambda x:returnMoney(x))
    print('done ...')
    obj_columns= list(df.dtypes[df.dtypes == np.object].index)
    df1 = df[obj_columns].copy()
    print(f'\t\t Finding Numeric Columns')
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

    import itertools
    
    def fixdate(entry):    # function to introduce '-' before and after month and and removing timestamp if it is seperated from date by':' 
        months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        for month in months:
            if month in str(entry).lower():
                index1= entry.find(month)
                index2= index1+3
                entry = entry[:index1]+'-'+entry[index1:index2]+'-'+entry[index2:]
                index3=entry.find(':')  #only specific to messy dataset
                entry=entry[:index3]
        return entry  
    
    start = time.time()
    
    if possible_datecols:
        for col in possible_datecols:
            df[col]=df[col].apply(fixdate)
            
    print('\n\t Entering Date Engineering')
    df = df.apply(pd.to_datetime,errors='coerce')
    print('\nPrinting Missing % of date columns')
    MISSING = pd.DataFrame(((df.isnull().sum().sort_values(ascending=False)*100/len(df)).round(2)),columns=['Missing in %'])[:10]
    print(MISSING)
    
    if validation==False:   # dropping columns with missing greater than 35% only in training not scoring
        print('Dropping Columns with missing greater than 35% of total number of entries')
        df.dropna(thresh=len(df)*0.65,axis=1,inplace=True)
    try:     
        for c in df.columns:
            df[c].fillna(df[c].mode()[0],inplace=True)    # replacing null values with mode 

    except:
        for c in df.columns:
            df[c].fillna(df[c].mean(),inplace=True)   # if error in mode then replacing null values with mean      
     
    date_cols = df.columns
    visualize_dict = dict.fromkeys(date_cols, [])

    # creating separate month and year columns, and difference from current date
    for i in date_cols:
        df[str(i)+"_month"] = df[str(i)].dt.month.astype(int)
        df[str(i)+"_year"] = df[str(i)].dt.year.astype(int)
        df[str(i)+"-today"] = (pd.to_datetime('today')-df[str(i)]).dt.days.astype(int)
        visualize_dict[str(i)] =  visualize_dict[str(i)] + [str(i)+"_month"] + [str(i)+"_year"]+[str(i)+"-today"]

    # create difference columns
    diff_days = list()
    if (len(date_cols)>1) :
        for i in itertools.combinations(date_cols,2):
            diff_days = diff_days + [str(i[0])+"-"+str(i[1])]
            df[str(i[0])+"-"+str(i[1])]=(df[i[0]]-df[i[1]]).dt.days.astype(int)


    print('\n\t #### RUNNING WAIT ####')

    # See Near Holiday or not
    def nearHol(currentDate, us_hols, currentYear):
        new_list = []
        append = new_list.append
        for date, _ in us_hols:
            if(date.year == currentYear):
                append(date)
        flag = 1
        for i in new_list:
            a = (currentDate.date()-i).days

            if abs(a)<=5:flag =1;break
            else:flag = 0

        return 0 if flag == 0 else 1

    for col in date_cols:
        # print('LOOP')
        #creating a unique list of all years corresponding to a column to minimise mapping
        us_hols = holidays.US(years=df[str(col)+'_year'].unique(), expand= False)
        #creating a new columns to check if a date falls near a holiday
        df[str(col)+'_Holiday'] = df.apply(lambda x: nearHol(x[col],us_hols.items(),x[str(col)+'_year']),axis=1)
        visualize_dict[str(col)] =  visualize_dict[str(col)] + [str(col)+"_nearestHoliday"]

    print("\nVisualizing Coloumns Generated\n {}" .format(visualize_dict))
    print("\nThe Following columns were generated to get days between dates of two seperate date columns\n {}".format(diff_days))
    end = time.time()
    print('\nDate Engineering Time Taken : {}'.format(end-start))
    print('\n\t #### DONE ####')
    return df.drop(date_cols,axis=1)

def nearHol(currentDate, us_hols, currentYear):
        new_list = []
        append = new_list.append
        for date, _ in us_hols:
            if(date.year == currentYear):
                append(date)
        flag = 1
        for i in new_list:
            a = (currentDate.date()-i).days

            if abs(a)<=5:flag =1;break
            else:flag = 0

        return 0 if flag == 0 else 1

############################################
############## TIME ENGINEERING ############
############################################

def time_engineering(props):
    print('#### ENTERING TIME ENGINEERING ####')
    print('UNPACKING PROPS')
    # Unpacking Props parameter
    df = props['df']
    primaryDate = props['info']['PrimaryDate']

    print('\nSorting DataFrame according to Primary Date Column')
    df.sort_values(by=[primaryDate],inplace=True)
    df.reset_index(inplace=True,drop=True)

    # Exploiting Date Column
    print('\nObtaining Year,Quarter,Month,Week,Day of Week,Day,Hour,Minute,Seconds,Weekend or not\n')
    df['Year'] = df[primaryDate].dt.year
    df['Quarter'] = df[primaryDate].dt.quarter
    df['Month'] = df[primaryDate].dt.month
    df['Week'] = df[primaryDate].dt.week
    df['Day of Week'] = df[primaryDate].dt.dayofweek
    df['Day'] = df[primaryDate].dt.day
    df['Hour'] = df[primaryDate].dt.hour
    df['Minute'] = df[primaryDate].dt.minute
    df['Seconds'] = df[primaryDate].dt.second
    df['Weekend_or_not'] = df['Day of Week'].apply(lambda x: 1 if x in [5,6,0,1] else 0)

    print('\nThings yet to be done')
    print('1. Find Space between dates and equally space them!')
    print('3. Get Near Holiday/not(+-5 days)')
    print('5. Get business hours/not')
    print('6. Get Morning/Afternoon/Evening/Night\n')

    # Removing columns that have absolutely one level
    print('\nRemoving columns with only one level')
    for column in df:
        if df[column].nunique() == 1:
            print('Column \'{}\' will be dropped since it has 1 level'.format(column))
            df.drop(column,axis=1,inplace=True)

    print('\nPrinting DataFrame Head\n')
    print(df.head())

    print('\n#### PACKING/UPDATING PROPS ####\n')
    props['df'] = df
    print('#### TIME ENGINEERING DONE ####')
    return props