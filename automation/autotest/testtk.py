import os,sys
import pandas as pd
import Training
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get Size of a specific file using /path/filename.extension
def getSize(filename):
    try:
        return os.stat('./test/' + filename).st_size
    except FileNotFoundError:
        print('File \'{}\' does not exist in the directory!'.format(filename))
        return None
    except Exception:
        print('Something went wrong for the file \'{}\''.format(filename))

class Logger(object):
    def __init__(self,filena):
        self.terminal = sys.stdout
        self.log = open(filena, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    
# Main Function definition
def main():
    # Creating a dataFrame for listing files,size,target and/or ID
    
    files_df = pd.read_csv('./test/TEST_LIST.csv')
    files_df['Size'] = files_df['Files'].apply(getSize)
    files_df.dropna(subset=['Size'],inplace=True)
    files_df.sort_values('Size',inplace=True)
    files_df.reset_index(drop=True,inplace=True)

    print('\nPrinting files present in the test folder : \n')
    print(files_df)

    # Use this indexNumber to edit other columns of the dataframe
    indexNumber = 0
    print('#### RUNNING WAIT ####')
    # For every file in the files list
    for filename in files_df['Files']:
        # sys.stdout = open('./test/' + filename.split('.')[0] + '_log.txt','w')
        filena = './test/' + filename.split('.')[0] + '_log.txt'
        sys.stdout = Logger(filena)
        print('Testing {}\n'.format(filename))
        # print("Enter the Target variable:")
        try:
            # props is a parameter that carries a list of Target and ID of one file
            props = [files_df.loc[indexNumber,'Target'],files_df.loc[indexNumber,'ID']]
            # Test file
            ret = Training.main(props,test=True,Path='./test/' + filename)
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
        # sys.stdout.close()

    # Save the DataFrame to CSV as a report of the test
    files_df.to_csv('./test/TestReport.report')

# Main Function Call
if __name__ == '__main__':
    main()