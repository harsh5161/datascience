import pandas as pd
import numpy as np
import re 
from userInputs import *

def web_init():
	path = 'datasets/telco.csv' #Plug in the path to the valid dataset
	df, _ = importFile(path)
	df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
	df = duplicateHandler(df)
	df,update = dataHandler(df) 
	df = duplicateHandler(df)

	if len(df)>1000:
		df = df.sample(n=1000,random_state=42)
    
	print(f'Returning DataFrame of shape {df.shape}')
	return df


if __name__ == '__main__':
	web_init()	#You can choose to call this file as it is or even just call that function from somewhere else
