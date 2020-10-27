import glob
import os
import pandas as pd
import sys

def getSize(filename):
    return os.stat('./test/' + filename).st_size

# Create a dataframe with filename and size as columns
# Sort the dataframe in ascending order of size
files_df = pd.DataFrame({'Files':os.listdir('./test')})
files_df['Size'] = files_df['Files'].apply(getSize)
files_df.sort_values('Size',inplace=True)
files_df.reset_index(drop=True,inplace=True)

print('Printing files present in the test folder : ')
print(files_df)

for filename in files_df['Files']:
    print('{} will be tested'.format(filename))
