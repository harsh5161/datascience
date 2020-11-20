import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ClusterMags(segdata):
    
    ## table
    print("\nThe following table shows cluster magnitudes...")
    cmags = segdata['K-means Segments'].value_counts().to_frame()
    cmags= cmags.reset_index()                               # to rename columns
    cmags.columns= ['Cluster number', 'Frequency']           # renaming columns
    cmags= cmags.sort_values('Cluster number', ascending =True, ignore_index=True)
    cmags['Percent']= round((cmags['Frequency']/cmags['Frequency'].sum())* 100, 2)
    cmags['Cummulative Frequency'] = cmags['Frequency'].cumsum()
    cmags['Cummulative Percent'] = round(((cmags['Frequency']/cmags['Frequency'].sum())* 100).cumsum(),2)
    display(cmags)
    
    ## bar chart
    sns.barplot(x=cmags['Cluster number'], y = cmags['Frequency'],  
            data = cmags, palette="husl").set_title('Cluster Magnitudes')
    plt.show()

    
def ClusterProfiling(segdata, num_df, disc_df):
    
    ## for numeric variables
    num_cp=pd.DataFrame(segdata.groupby('K-means Segments')[num_df.columns.to_list()].agg(np.mean))
    num_cp=num_cp.reset_index()                                           # to rename columns
    num_cp.columns= num_cp.columns[:1].tolist()+ num_df.columns.to_list()   # renaming columns
    print("\nCluster Profiles Using Numeric Variables...")
    display(num_cp)
    
    ## for categorical variables  
    print("\nCluster profiles for each categorical variable...")
    for col in disc_df.columns:
        cat_cp = pd.crosstab(index=segdata['K-means Segments'], 
                         columns=segdata[col],
                         margins=True, margins_name ='Total')

    #     cat_cp.columns = [segdata[col].unique().tolist()+['rowtotal']]
    #     cat_cp.index= [segdata['K-means Segments'].unique().tolist()+['coltotal']]

    #     cat_cp.reset_index(inplace= True )
    #     cat_cp.columns = ['K-means Segments'] + segdata[col].unique().tolist() +['rowtotal']
        display(cat_cp)