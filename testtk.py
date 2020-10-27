import os
import pandas as pd
import Training

# Get Size of a specific file using /path/filename.extension
def getSize(filename):
    return os.stat('./test/' + filename).st_size

# Main Function definition
def main():
    files = [] # Creating a list of valid files based on extension
        # Only CSV,XLSX,XLS and JSON files will be tested
        for i in os.listdir('./test/'):
            try:
                ext = i.split('.')[1].lower()
                if ext == 'csv' or ext == 'xlsx' or ext == 'xls' or ext == 'json':
                    files.append(i)
            except:
                pass
    print(files)

    # Create a dataframe with filename and size as columns
    # Sort the dataframe in ascending order of size
    # To test smaller files first and then moving on to the bigger files
    files_df = pd.DataFrame({'Files':files})
    files_df['Size'] = files_df['Files'].apply(getSize)
    files_df.sort_values('Size',inplace=True)
    files_df.reset_index(drop=True,inplace=True)

    print('\nPrinting files present in the test folder : \n')
    print(files_df)

    # Use this indexNumber to edit other columns of the dataframe
    indexNumber = 0 

    # For every file in the files list
    for filename in files_df['Files']:
        print('Testing {}\n'.format(filename))
        try:
            # Test file
            ret = Training.main(test=True,Path='./test/' + filename)
            # Edit dataframe to know if it ran succesfully
            if ret:files_df.loc[indexNumber,'Result'] = 'Success'
            else:files_df.loc[indexNumber,'Result'] = 'Error'
        except Exception as e:
            # Print Exceptions
            print(e)
            print('\n\n')
            files_df.loc[indexNumber,'Result'] = 'Error'
        # Increment index to write on the next row of the dataframe
        indexNumber += 1

    # Save the DataFrame to CSV as a report of the test
    files_df.to_csv('./test/TestReport.report')

# Main Function Call
if __name__ == '__main__':
    main()