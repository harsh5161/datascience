import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi

from collections import defaultdict

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
styls = [ dict(props = [('border-collapse','collapse'), ('border-spacing','0px')]),
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
    
    ## creating a dict to store cluster numbers and their respective percentages
    clp=pd.DataFrame(cmags[['Cluster numbers','Percent']])
    clp.set_index('Cluster numbers', inplace= True)
    cluster_percentages=clp.to_dict()
    cluster_percentages=cluster_percentages['Percent']
    print(cluster_percentages)
    return cluster_percentages
    
    
def ClusterProfiling_Tables(segdata, num_df, disc_df):
    # print("###################################################")
    # print(segdata.head(20))
    # print("###################################################")
    # print(num_df.head(20))
    # print("###################################################")
    # print(disc_df.head(20))
    # print("###################################################")
    ## for numeric variables
    high_mean_vals= defaultdict(list) # dictionary to store which cluster has high values of which numeric column
    low_mean_vals= defaultdict(list)  # dictionary to store which cluster has low values of which numeric column
    if not num_df.empty:   # only runs if numerical varibales are present in list of columns used for profiling
        num_cp=pd.DataFrame(segdata.groupby('Segments (Clusters)')[num_df.columns.to_list()].agg(np.mean))  #calculating means per cluster
        num_cp=num_cp.reset_index()                                           # to rename columns
        num_cp.columns= num_cp.columns[:1].tolist()+ ['Mean of ' + i for i in num_df.columns.to_list()]   # renaming columns

        overall_means = segdata[num_df.columns.to_list()].mean(axis=0).to_list()  # calculating means in overall dataset
        overall_row = ['Overall Dataset'] + overall_means
        num_cp.loc[-1] = overall_row           # adding overall means row at the top of the table
        num_cp.index = num_cp.index + 2 
        num_cp.sort_index(inplace=True)

    #     display(num_cp) #before scaling # can be used for testing

        for c in num_cp.columns[1:]:          # scaling overall dataset row to 100 for easy comparison
                ov= num_cp[c].iloc[0]
                num_cp[c]= (num_cp[c]/ov)*100


        def highlight(x):  # function for highlighting cells   
            return ['background: #fffcb0' if v== x.iloc[0] 
                    else ('background-color: #0e7a8f' if v > 150 
                    else ('background-color: #a1d6e2' if v < 50
                          else '' )) for v in x]

        styled_num_cp =num_cp.style.set_table_styles(styls).apply(highlight, subset= num_cp.columns[1:]).set_precision(2).hide_index()  # styling df
        dfi.export(styled_num_cp,'Numeric var cluster profiles.png', max_cols=-1)  # storing as image
        print("\nCluster Profiles Using Numeric Variables...")
        display(styled_num_cp)  # displaying df

        ## recording high and low mean values of numeric variables per cluster        
        num_cp_copy=num_cp.set_index('Segments (Clusters)')

        hmv_T={}
        lmv_T={}
        for c in num_cp_copy.columns:          # getting which cluster numbers have high and low values for a numeric column       
            hmv_T[c[8:]]=num_cp_copy[num_cp_copy[c]>150].index.to_list() 
            lmv_T[c[8:]]=num_cp_copy[num_cp_copy[c]<50].index.to_list()

        for node, neighbours in hmv_T.items():  # loop to invert hmv_T dictionary mapping so that cluster numbers become keys and column names become values
            for neighbour in neighbours:
                high_mean_vals[neighbour].append(node)                  

        for node, neighbours in lmv_T.items():  # loop to invert lmv_T dictionary mapping
            for neighbour in neighbours:
                low_mean_vals[neighbour].append(node)                  

        for i in num_cp_copy.index.values[1:]:  # adding missing cluster numbers in keys
            if i not in high_mean_vals.keys():
                high_mean_vals[i]=[]

            if i not in low_mean_vals.keys():
                low_mean_vals[i]=[]

        high_mean_vals=dict(sorted(high_mean_vals.items()))  # sorting 
        low_mean_vals=dict(sorted(low_mean_vals.items()))

        print("\nhigh_mean_vals::", high_mean_vals)
        print("\nlow_mean_vals::", low_mean_vals)
    
    ## for categorical variables  
    high_percent_levels={} # dictionary to store which cluster has high percentage of a categorical column's level
    zero_percent_levels={} # dictionary to store which cluster has zero percentage of a categorical column's level

    if not disc_df.empty:    # only runs if categorical varibales are present in list of columns used for profiling
        print("\nCluster profiles for each categorical variable...")
        
    # generating one table for each variable in which column names are catgories present in the variable and indexes are the cluster numbers
        for col in disc_df.columns:       
            cat_cp = pd.crosstab(index=segdata['Segments (Clusters)'],  # gives frequency of each category in the variable for each cluster
                             columns=segdata[col],
                             margins=True, margins_name ='Total')

        #     cat_cp.columns.name=cat_cp.index.name  
            cat_cp.rename_axis(None, axis=1, inplace=True)    # removing cat_cp.columns.name
            cat_cp.rename(columns = {'Total':'Row total'}, index= {'Total':'Overall Dataset'}, inplace = True) # renaming total column and index
            cat_cp=(cat_cp.div(cat_cp["Row total"], axis=0)*100)  # dividing by row total and getting percentage of frequency
            cat_cp = cat_cp.iloc[np.arange(-1, len(cat_cp)-1)]   #shifting the last row to first position
            cat_cp.drop(['Row total'], axis= 1, inplace =True)   #dropping row total column

            for c in cat_cp.columns:          # scaling overall dataset % to 100 for easy comparison
                ov= cat_cp[c].loc['Overall Dataset']
                cat_cp[c]= (cat_cp[c]/ov)*100

            def cat_highlight(x):   # function to highlight cells 
                if x.name == "Overall Dataset":
                    return ['background: #fffcb0' for v in x]
                else:
                    return ['background-color: #0e7a8f' if v > 150 
                            else ('background-color: #46abc2' if (v <= 150 and v >100) 
                                  else ('background-color: #bcbabe' if v==0 else '')) for v in x]

            # if column names in cat_cp contain special characters(which throw a unicodeerror) then they are encoded
            try:  
                styled_cat_cp =cat_cp.style.set_table_styles(styls).apply(cat_highlight, axis = 1).set_caption(str(col)+" (%)").set_precision(2) #styling
                dfi.export(styled_cat_cp, 'cluster profiles for '+str(col)+ '.png',max_cols=-1)# save as image     
            except UnicodeEncodeError:
                cat_cp.columns=pd.Series(cat_cp.columns).apply(lambda x: str(x.encode('utf-8'))[2:-1] if type(x)==str else x)
                styled_cat_cp =cat_cp.style.set_table_styles(styls).apply(cat_highlight, axis = 1).set_caption(str(col)+" (%)").set_precision(2) #styling
                dfi.export(styled_cat_cp, 'cluster profiles for '+str(col)+ '.png',max_cols=-1)# save as image
           
            display(styled_cat_cp)

            ## for text cluster profiles
            cat_cp_copy=cat_cp.copy()

            hpl_T={}
            zpl_T={}

            for c in cat_cp_copy.columns: # getting which cluster numbers have high and low values for a category in the variable 
                hpl_T[c]=cat_cp_copy[cat_cp_copy[c]>150].index.to_list()
                zpl_T[c]=cat_cp_copy[cat_cp_copy[c]==0].index.to_list()

            high_percent_levels[col] = defaultdict(list)
            for node, neighbours in hpl_T.items():  # loop to invert hpl_T dictionary mapping 
                for neighbour in neighbours:
                    high_percent_levels[col][neighbour].append(node)                  

            zero_percent_levels[col] = defaultdict(list)        
            for node, neighbours in zpl_T.items():  # loop to invert zpl_T dictionary mapping 
                for neighbour in neighbours:
                    zero_percent_levels[col][neighbour].append(node)                  

            for i in cat_cp_copy.index.values[1:]:   #loop to add missing cluster numbers to keys
                if i not in high_percent_levels[col].keys():
                    high_percent_levels[col][i]=[]

                if i not in zero_percent_levels[col].keys():
                    zero_percent_levels[col][i]=[]

            high_percent_levels[col]=dict(sorted(high_percent_levels[col].items()))  #sorting
            zero_percent_levels[col]=dict(sorted(zero_percent_levels[col].items()))

    #         print("\nhigh_percent_levels:::\n",high_percent_levels) 
    #         print("\nzero_percent_levels:::\n",zero_percent_levels) 
     
    return high_mean_vals, low_mean_vals, high_percent_levels, zero_percent_levels

def ClusterProfiling_Text(cluster_percentages, high_mean_vals, low_mean_vals, high_percent_levels, zero_percent_levels):
    cp_text={}
    print("\n\n\n\t\t_____________CLUSTER PROFILES_______________\n")
    for i in range(0,len(cluster_percentages)):
        cp_text[i]=str("\033[1m\033[4m"+"\nCLUSTER {}:".format(i)+"\033[0m")
        cp_text[i]+=str("\nCluster {} represents {}% of the Overall dataset.".format(i,cluster_percentages[i]))
        
        if high_mean_vals[i]:
            cp_text[i]+=str("\nIt contains relatively high values of the numeric variables- {}".format(str(high_mean_vals[i])[1:-1]))
            
        if low_mean_vals[i]:
            cp_text[i]+=str("\nIt contains relatively low values of the numeric variables- {}".format(str(low_mean_vals[i])[1:-1]))
        
        for k in high_percent_levels.keys():
            if high_percent_levels[k][i]: 
                cp_text[i]+=str("\nFor the variable '{}', it contains more of categories- {}.".format(k, str(high_percent_levels[k][i])[1:-1]))
            if zero_percent_levels[k][i]:
                cp_text[i]+=str("\nFor the variable '{}', it does not contain categories- {}.".format(k, str(zero_percent_levels[k][i])[1:-1]))
        print(cp_text[i])