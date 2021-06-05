import pandas as pd
import numpy as np
import re 
from userInputs import *

def web_init():
	path = 'datasets/titanic.csv' #Plug in the path to the valid dataset
	df, _ = importFile(None,None,path)
	df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
	df = duplicateHandler(df)
	df,update = dataHandler(df) 
	df = duplicateHandler(df)

	if len(df)>1000:
		df = df.sample(n=1000,random_state=42)

	# print(f'No. of NAs in dataset before transformation {df.isna().sum()}')

	numlist = list(df.select_dtypes(include=['int64','float64']).columns)
	objectlist = list(df.select_dtypes(include=['object']).columns)
	df[numlist]=df[numlist].fillna(df.mode().iloc[0])
	# get the dataframe at this point for preview
	for col in numlist:
		df[col] = df[col].clip(lower=df[col].quantile(0.1),upper=df[col].quantile(0.9))
	df[objectlist] = df[objectlist].fillna('missing',axis=1)

	# print(f'No. of NAs in dataset after transformation {df.isna().sum()}')
	print(f'Returning DataFrame of shape {df.shape}')
	df.to_csv("websupport_test.csv",index=False)	#the dataframe at this point would only be useful for the plots
	return df


if __name__ == '__main__':
	web_init()	#You can choose to call this file as it is or even just call that function from somewhere else
