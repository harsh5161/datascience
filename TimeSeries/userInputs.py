import pandas as pd
import xlrd,csv


def importFile(path,nrows=None):

    print('#### RUNNING WAIT ####')

    # IF THE EXTENSION IS CSV
    def importCsv(path):

        print('We have a csv file')
        try:
            df = pd.read_csv(path,low_memory=False,nrows=nrows)
            if df.shape[1] == 1:
                df = pd.read_csv(path,low_memory=False,sep=';',nrows=nrows)
            print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
            return df

        except FileNotFoundError:
            print('File not found, Check the name, path, spelling mistakes')
            return None

        except UnicodeDecodeError:
            try:
                enc = 'unicode_escape'
                df = pd.read_csv(path,encoding=enc,low_memory=False,nrows=nrows)
                print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
                return df

            except UnicodeDecodeError:
                try:
                    enc = 'ISO-8859-1'
                    df = pd.read_csv(path,encoding=enc,low_memory=False,nrows=nrows)
                    print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
                    return df
                except:
                    pass

        except:
            try:
                df= pd.read_csv(path,nrows=nrows)
                separators= ["~","!","@","#","$","%","^","&","*",":","|","/",";"]     # all possible separators
                if len(df.columns)<=3 :                                               # if separator was "," we would have more than 1 columns
                    cols = df.columns[0]
                    possibleSep = []
                    for i in separators:                                    # checking all the separators present in column names
                        if i in cols:
                            possibleSep.append(i)

                    for j in possibleSep:                                   # iterate through possible seprators till we get the correct one
                        df_sep = pd.read_csv(path,sep=j,nrows=nrows)
                        if len(df_sep.columns)>3:
                            print('This file has {} columns and {} rows'.format(df_sep.shape[1],df_sep.shape[0]))
                            return df_sep
            except:
                try:
                    if len(pd.read_csv(path,sep=None).columns,nrows=nrows)>3  :                   # for tab ie "\" tsv files
                        df = pd.read_csv(path,sep=None,nrows=nrows)
                        print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
                        return df
                except:
                    pass

    # IF THE EXTENSION IS JSON
    def importJSON(path):
        try:
            print('We have a JSON file')
            df = pd.read_json(path)
            print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
            return df
        except Exception:
            try:
                df = pd.read_json(path,lines=True)
                print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
                return df

            except ValueError:
                print('File not found, Check the name, path, spelling mistakes')
                return None

    # IF THE EXTENSION IS XL
    def importExcel(path):
        try:
            print('We have an Excel file')
            #######
            # opening workbook
            wb = xlrd.open_workbook(path)

            # Tracking Sheets inside workbook
            sheet_names = wb.sheet_names()
            if len(sheet_names)==1:
              sheet_selected = sheet_names[0]
            else:
              print("\nFollowing Are The sheets Found in the workbook\n {}".format(sheet_names))
              sheet_selected = input("Type the sheet name:  ")

            # open workbook by and get sheet by index
            sheet = wb.sheet_by_name(sheet_selected)

            # # writer object is created
            col = csv.writer(open("SheetSheetSheet.csv", 'w', newline=""))

            # writing the data into csv file
            for row in range(sheet.nrows):
                col.writerow(sheet.row_values(row))
            print('\nXlrd Done')

            # read csv file and convert into a dataframe object
            return pd.read_csv("SheetSheetSheet.csv")
            #######
        except FileNotFoundError:
            print('File not found, Check the name, path, spelling mistakes')
            return None

    def importTable(path):
        try:
            print('We have General Table File')
            df = pd.read_table(path,nrows=nrows)
            if df.shape[1] == 1:
                df = pd.read_table(path,sep=',',nrows=nrows)
            print('This file has {} columns and {} rows'.format(df.shape[1],df.shape[0]))
            return df
        except FileNotFoundError:
            print('File not found, Check the name, path, spelling mistakes')
            return None

    try:
        ext = path.split('.')[-1].strip()
        print('extension is {}'.format(ext))
        if ext == 'csv' or ext == 'tsv':
            df = importCsv(path)
            return df,None
        elif ext == 'json':
            df = importJSON(path)
            return df,None
        elif 'xl' in ext:
            df = importExcel(path)
            return df,'SheetSheetSheet.csv'
        elif ext == 'data':
            df = importTable(path)
            return df,None
        else:
            print('File format not supported\n')
    except Exception as e:
        print('We ran into some Error!')
        print('The error message is {}'.format(e))
        return None,None

def getForecastPeriod():
    inp = input('Enter the forecast period in the format years,months,days : ').strip().split(',')
    return (int(inp[0]) * 365 + int(inp[1]) * 30 + int(inp[2]))

def getInfo(cols,datecols,test=False):
    if not test:
        print('The columns found are :')
        print(cols)
        target = input('Enter the target column : ')
        print('\nThe date columns found are :')
        print(datecols)
        primaryDate = input('Enter the primary date column : ')
        forecastPeriod = getForecastPeriod()
        print('The forecast period is : {} days'.format(forecastPeriod))
        if target =='' or primaryDate == '' or forecastPeriod == None:
            print('Empty entries')
            return None
        elif (target not in cols) or (primaryDate not in datecols):
            print('Entry not found in respective columns')
            return None
        else:
            info = {'Target':target,'PrimaryDate':primaryDate,'ForecastPeriod':forecastPeriod}
            return info
    else:
        # Pass all test parameters into info
        pass

    