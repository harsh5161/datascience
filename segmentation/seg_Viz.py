import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi


########## CSS for styling dataframe ################

# Set CSS properties for th elements in dataframe
th_prop = [
    ('padding', '5px'),
    ('font-family', 'arial'),
    ('font-size', '100%'),
    ('color', 'Black'),
    ('border', '0.1px') ,
    ('border', 'solid black'),
    ('text-align', 'center')
  ]

# Set CSS properties for td elements in dataframe
td_prop = [
#     ('background', 'rgb(232, 247, 252)'),
    ('border', '0.1px'),
    ('border','solid black'),
    ('color', 'black'),
    ('font-family', 'arial')
  ]

# Set table styles
styls = [
  dict(selector="th", props=th_prop),
  dict(selector="td", props=td_prop),
  dict(selector="caption", 
       props=[("text-align", "center"),
              ("font-size", "110%"),
              ("font-weight", "bold"),
              ("color", 'black'), 
              ('border', '0.1px'), 
              ('border', 'solid black')])
  ]

##########################################################################

def ClusterMags(segdata):
    
    ## table
    print("\nThe following table shows cluster magnitudes...")
    cmags = segdata['Segments (Clusters)'].value_counts().to_frame()
    cmags= cmags.reset_index()                               # to rename columns
    cmags.columns= ['Cluster numbers', 'Frequency']           # renaming columns
    cmags= cmags.sort_values('Cluster numbers', ascending =True, ignore_index=True)
    cmags['Percent']= round((cmags['Frequency']/cmags['Frequency'].sum())* 100, 2)
    cmags['Cummulative Frequency'] = cmags['Frequency'].cumsum()
    cmags['Cummulative Percent'] = round(((cmags['Frequency']/cmags['Frequency'].sum())* 100).cumsum(),2)
    styled_cmags =cmags.style.set_table_styles(styls).set_precision(2).hide_index() #styling
    dfi.export(styled_cmags,'Cmags_table.png')  # storing as image
    display(styled_cmags)
    
    ## bar chart
    sns_plot = sns.barplot(x=cmags['Cluster numbers'], y = cmags['Frequency'],  
            data = cmags, palette="husl").set_title('Cluster Magnitudes')
    sns_plot.figure.savefig('Cmags_barchart.png')
    plt.show()

    
def ClusterProfiling(segdata, num_df, disc_df):
    
    ## for numeric variables
    num_cp=pd.DataFrame(segdata.groupby('Segments (Clusters)')[num_df.columns.to_list()].agg(np.mean))  #calculating means per cluster
    num_cp=num_cp.reset_index()                                           # to rename columns
    num_cp.columns= num_cp.columns[:1].tolist()+ ['Mean of ' + i for i in num_df.columns.to_list()]   # renaming columns

####### this commented block can be used for testing quantile logic ###########################
#     columns = [i[8:] for i in num_cp.columns[1:]]  #adding quantiles 1 and 3 of each column in num_cp
#     quant3=[]
#     quant1=[]
#     for col in columns:
#         quant3.append(np.quantile(segdata[col].values,0.75))
#         quant1.append(np.quantile(segdata[col].values,0.25))
#     q1=['quantile 1']+quant1
#     q3=['quantile 3']+quant3
#     q=pd.DataFrame([q1,q3],columns= num_cp.columns)
#     num_cp=pd.concat([num_cp,q], ignore_index=True)
###################################################################################################

    overall_means = segdata[num_df.columns.to_list()].mean(axis=0).to_list()  # calculating means in overall dataset
    overall_row = ['Overall Dataset'] + overall_means
    num_cp.loc[-1] = overall_row           # adding overall means row at the top of the table
    num_cp.index = num_cp.index + 2 
    num_cp.sort_index(inplace=True)
    
    def highlight(x):  # function for highlighting cells  
        return ['background-color: #CC0000' if v > np.quantile(segdata[x.name[8:]].values,0.75) 
                else ('background-color: #FFCC66' if v < np.quantile(segdata[x.name[8:]].values,0.25) else '' ) for v in x]
    
    styled_num_cp =num_cp.style.set_table_styles(styls).apply(highlight, subset= num_cp.columns[1:]).set_precision(2).hide_index()  # styling df
    dfi.export(styled_num_cp,'Numeric var cluster profiles.png')  # storing as image
    print("\nCluster Profiles Using Numeric Variables...")
    display(styled_num_cp)  # displaying df
    
    ## for categorical variables     
    print("\nCluster profiles for each categorical variable...")
    for col in disc_df.columns:
        cat_cp = pd.crosstab(index=segdata['Segments (Clusters)'], 
                         columns=segdata[col],
                         margins=True, margins_name ='Total')

    #     cat_cp.columns.name=cat_cp.index.name  
        cat_cp.rename_axis(None, axis=1, inplace=True)    # removing cat_cp.columns.name
        cat_cp.rename(columns = {'Total':'Row total'}, inplace = True) # renaming total column
        cat_cp=(cat_cp.div(cat_cp["Row total"], axis=0)*100)  # dividing by row total and getting percentage of frequency
        cat_cp = cat_cp[:-1]   #deleting last row because we dont want column totals
        
        def cat_highlight(x):
            return ['background-color: #CC0000' if v > 80 
                    else ('background-color: #FF9966' if (v < 20 and v >0) 
                          else ('background-color: #CCFFFF' if v==0 else '')) for v in x]

        styled_cat_cp =cat_cp.style.set_table_styles(styls).apply(cat_highlight, subset= cat_cp.columns[:-1]).set_caption(str(col)+" (%)").set_precision(2) #styling
        
        dfi.export(styled_cat_cp, 'cluster profiles for '+str(col)+ '.png')# save as image
        display(styled_cat_cp)
        