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
    ('border', '0.1px'),
    ('border', 'solid black'),
    ('text-align', 'center')
]

# Set CSS properties for td elements in dataframe
td_prop = [
    #     ('background', 'rgb(232, 247, 252)'),
    ('border', '0.1px'),
    ('border', 'solid black'),
    ('color', 'black'),
    ('font-family', 'arial')
]

# Set table styles
styls = [dict(props=[('border-collapse', 'collapse'), ('border-spacing', '0px')]),
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


# function to plot scatter plot showing clusters
def clusters_scatter_plot(pca_components, cluster_labels):
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink',
              'olive', 'goldenrod', 'lightcyan', 'navy', 'coral', 'olive', 'turquoise']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    plt.scatter(pca_components.iloc[:, 0], pca_components.iloc[:, 1], c=vectorizer(
        cluster_labels), s=5)
    plt.show()


def ClusterMags(segdata, algo):

    # table
    #     print("\nThe following table shows cluster magnitudes...")
    cmags = segdata['Segments (Clusters)'].value_counts().to_frame()
    cmags = cmags.reset_index()                               # to rename columns
    # renaming columns
    cmags.columns = ['Cluster numbers', 'Frequency']
    cmags = cmags.sort_values(
        'Cluster numbers', ascending=True, ignore_index=True)
    cmags['Percent'] = round(
        (cmags['Frequency']/cmags['Frequency'].sum()) * 100, 2)
    cmags['Cummulative Frequency'] = cmags['Frequency'].cumsum()
    cmags['Cummulative Percent'] = round(
        ((cmags['Frequency']/cmags['Frequency'].sum()) * 100).cumsum(), 2)
    styled_cmags = cmags.style.set_table_styles(
        styls).set_precision(2).hide_index()  # styling
    dfi.export(styled_cmags, 'Cmags_table_'+str(algo)+'.png',
               table_conversion='matplotlib')  # storing as image
    display(styled_cmags)

    # bar chart
    sns_plot = sns.barplot(x=cmags['Cluster numbers'], y=cmags['Frequency'],
                           data=cmags, palette="husl").set_title('Cluster Magnitudes')
    sns_plot.figure.savefig('Cmags_barchart_'+str(algo)+'.png')
    plt.show()

    # creating a dict to store cluster numbers and their respective percentages
    clp = pd.DataFrame(cmags[['Cluster numbers', 'Percent']])
    clp.set_index('Cluster numbers', inplace=True)
    cluster_percentages = clp.to_dict()
    cluster_percentages = cluster_percentages['Percent']
    return cluster_percentages


def ClusterProfiling_Tables(segdata, num_df, disc_df, algorithm):
    # print("###################################################")
    # print(segdata.head(20))
    # print("###################################################")
    # print(num_df.head(20))
    # print("###################################################")
    # print(disc_df.head(20))
    # print("###################################################")
    # for numeric variables
    # dictionary to store which cluster has high values of which numeric column
    high_mean_vals = defaultdict(list)
    # dictionary to store which cluster has low values of which numeric column
    low_mean_vals = defaultdict(list)
    if not num_df.empty:   # only runs if numerical varibales are present in list of columns used for profiling
        num_cp = pd.DataFrame(segdata.groupby('Segments (Clusters)')[
                              num_df.columns.to_list()].agg(np.mean))  # calculating means per cluster
        # to rename columns
        num_cp = num_cp.reset_index()
        num_cp.columns = num_cp.columns[:1].tolist(
        ) + ['Mean of ' + i for i in num_df.columns.to_list()]   # renaming columns

        overall_means = segdata[num_df.columns.to_list()].mean(
            axis=0).to_list()  # calculating means in overall dataset
        overall_row = ['Overall Dataset'] + overall_means
        # adding overall means row at the top of the table
        num_cp.loc[-1] = overall_row
        num_cp.index = num_cp.index + 2
        num_cp.sort_index(inplace=True)

        print("\n\033[1mCluster Profiles Using Numeric Variables...\033[0m")
        print("\nThe following table shows the mean of each numeric variable in the overall dataset as well as in the formed clusters. If the mean of a variable is significantly higher in a cluster than the overall dataset, then it indicates that, that cluster has more percentage of high values for that variable as compared to the overall dataset.")
        # before scaling # can be used for testing
        display(num_cp.style.set_table_styles(styls).set_precision(2))
        num_cp.to_csv(f"{algorithm}_NumCP.csv", index=False)

        # scaling overall dataset row to 100 for easy comparison
        for c in num_cp.columns[1:]:
            ov = num_cp[c].iloc[0]
            num_cp[c] = (num_cp[c]/ov)*100

        def highlight(x):  # function for highlighting cells
            return ['background: #fffcb0' if v == x.iloc[0]
                    else ('background-color: #0e7a8f' if v > 150
                    else ('background-color: #a1d6e2' if v < 50
                          else '')) for v in x]

        styled_num_cp = num_cp.style.set_table_styles(styls).apply(
            highlight, subset=num_cp.columns[1:]).set_precision(2).hide_index()  # styling df
        dfi.export(styled_num_cp, f'{algorithm}_Numeric var cluster profiles.png',
                   max_cols=-1, table_conversion='matplotlib')  # storing as image
        print("\nFor easier comparison the overall dataset row is scaled to 100 and all other values are scaled accordingly in the table below. This table can be interpreted like so: For example, if the value of mean of a variable in a cluster is 150, that means, in that particular cluster the mean of said variable is 50% higher than its mean in the overall dataset. ")
        display(styled_num_cp)  # displaying df

        # recording high and low mean values of numeric variables per cluster
        num_cp_copy = num_cp.set_index('Segments (Clusters)')

        hmv_T = {}
        lmv_T = {}
        for c in num_cp_copy.columns:          # getting which cluster numbers have high and low values for a numeric column
            hmv_T[c[8:]] = num_cp_copy[num_cp_copy[c] > 150].index.to_list()
            lmv_T[c[8:]] = num_cp_copy[num_cp_copy[c] < 50].index.to_list()

        for node, neighbours in hmv_T.items():  # loop to invert hmv_T dictionary mapping so that cluster numbers become keys and column names become values
            for neighbour in neighbours:
                high_mean_vals[neighbour].append(node)

        for node, neighbours in lmv_T.items():  # loop to invert lmv_T dictionary mapping
            for neighbour in neighbours:
                low_mean_vals[neighbour].append(node)

        # adding missing cluster numbers in keys
        for i in num_cp_copy.index.values[1:]:
            if i not in high_mean_vals.keys():
                high_mean_vals[i] = []

            if i not in low_mean_vals.keys():
                low_mean_vals[i] = []

        high_mean_vals = dict(sorted(high_mean_vals.items()))  # sorting
        low_mean_vals = dict(sorted(low_mean_vals.items()))

        print("\nhigh_mean_vals::", high_mean_vals)
        print("\nlow_mean_vals::", low_mean_vals)

    # for categorical variables
    # dictionary to store which cluster has high percentage of a categorical column's level
    high_percent_levels = {}
    # dictionary to store which cluster has zero percentage of a categorical column's level
    zero_percent_levels = {}

    if not disc_df.empty:    # only runs if categorical varibales are present in list of columns used for profiling
        print(
            "\n\033[1mCluster profiles for each categorical variable...\033[0m")

    # generating one table for each variable in which column names are catgories present in the variable and indexes are the cluster numbers
        for col in disc_df.columns:
            cat_cp = pd.crosstab(index=segdata['Segments (Clusters)'],  # gives frequency of each category in the variable for each cluster
                                 columns=segdata[col],
                                 margins=True, margins_name='Total')

        #     cat_cp.columns.name=cat_cp.index.name
            # removing cat_cp.columns.name
            cat_cp.rename_axis(None, axis=1, inplace=True)
            cat_cp.rename(columns={'Total': 'Row total'}, index={
                          'Total': 'Overall Dataset'}, inplace=True)  # renaming total column and index
            # dividing by row total and getting percentage of frequency
            cat_cp = (cat_cp.div(cat_cp["Row total"], axis=0)*100)
            # shifting the last row to first position
            cat_cp = cat_cp.iloc[np.arange(-1, len(cat_cp)-1)]

            print("\nThe following table shows the share of each category of the variable \033[1m" + str(
                col) + "\033[0m in a cluster. It also shows the percentage of each category present in the overall dataset. For example, if there is variable 'Grade' which contains the categories 'A', 'B' and 'C', and for cluster 0 the value of category 'A' is 40 that means 40% of all the data points in cluster 0 belong to catgery 'A'. If the percentage of category 'A' in the overall dataset is 20% then that means cluster 0 has disproportionately larger share of category 'A'.")
            display(cat_cp.style.set_table_styles(styls).set_caption(
                str(col)+" (%)").set_precision(2))  # before scaling # can be used for testing
            cat_cp.to_csv(f"{algorithm}_CatCP_{str(col)}.csv", index=False)

            # dropping row total column
            cat_cp.drop(['Row total'], axis=1, inplace=True)

            for c in cat_cp.columns:          # scaling overall dataset % to 100 for easy comparison
                ov = cat_cp[c].loc['Overall Dataset']
                cat_cp[c] = (cat_cp[c]/ov)*100

            def cat_highlight(x):   # function to highlight cells
                if x.name == "Overall Dataset":
                    return ['background: #fffcb0' for v in x]
                else:
                    return ['background-color: #0e7a8f' if v > 150
                            else ('background-color: #46abc2' if (v <= 150 and v > 100)
                                  else ('background-color: #bcbabe' if v == 0 else '')) for v in x]

            # if column names in cat_cp contain special characters(which throw a unicodeerror) then they are encoded
            # try:
                # styled_cat_cp =cat_cp.style.set_table_styles(styls).apply(cat_highlight, axis = 1).set_caption(str(col)+" (%)").set_precision(2) #styling
                # dfi.export(styled_cat_cp, 'cluster profiles for '+str(col)+ '.png',max_cols=-1,table_conversion='matplotlib')# save as image
            # except UnicodeEncodeError:
                #cat_cp.columns=pd.Series(cat_cp.columns).apply(lambda x: str(x.encode('utf-8'))[2:-1] if type(x)==str else x)
                # styled_cat_cp =cat_cp.style.set_table_styles(styls).apply(cat_highlight, axis = 1).set_caption(str(col)+" (%)").set_precision(2) #styling
                # dfi.export(styled_cat_cp, 'cluster profiles for '+str(col)+ '.png',max_cols=-1,table_conversion='matplotlib')# save as image

            #print("\nFor easier comparison the overall dataset row is scaled to 100 and all other values are scaled accordingly in the table below. This table can be interpreted like so: For example, if the value of a category in a cluster is 150, that means, in that particular cluster the share of that category is 50 times more than its share in the overall dataset.")
            # display(styled_cat_cp)

            # for text cluster profiles
            cat_cp_copy = cat_cp.copy()

            hpl_T = {}
            zpl_T = {}

            for c in cat_cp_copy.columns:  # getting which cluster numbers have high and low values for a category in the variable
                hpl_T[c] = cat_cp_copy[cat_cp_copy[c] > 150].index.to_list()
                zpl_T[c] = cat_cp_copy[cat_cp_copy[c] == 0].index.to_list()

            high_percent_levels[col] = defaultdict(list)
            for node, neighbours in hpl_T.items():  # loop to invert hpl_T dictionary mapping
                for neighbour in neighbours:
                    high_percent_levels[col][neighbour].append(node)

            zero_percent_levels[col] = defaultdict(list)
            for node, neighbours in zpl_T.items():  # loop to invert zpl_T dictionary mapping
                for neighbour in neighbours:
                    zero_percent_levels[col][neighbour].append(node)

            # loop to add missing cluster numbers to keys
            for i in cat_cp_copy.index.values[1:]:
                if i not in high_percent_levels[col].keys():
                    high_percent_levels[col][i] = []

                if i not in zero_percent_levels[col].keys():
                    zero_percent_levels[col][i] = []

            high_percent_levels[col] = dict(
                sorted(high_percent_levels[col].items()))  # sorting
            zero_percent_levels[col] = dict(
                sorted(zero_percent_levels[col].items()))

    #         print("\nhigh_percent_levels:::\n",high_percent_levels)
    #         print("\nzero_percent_levels:::\n",zero_percent_levels)

    return high_mean_vals, low_mean_vals, high_percent_levels, zero_percent_levels


def ClusterProfiling_Text(cluster_percentages, high_mean_vals, low_mean_vals, high_percent_levels, zero_percent_levels):
    cp_text = {}
    print("\n\n\n\t\t_____________CLUSTER PROFILES_______________\n")
    for i in range(0, len(cluster_percentages)):
        cp_text[i] = str("\033[1m\033[4m"+"\nCLUSTER {}:".format(i)+"\033[0m")
        cp_text[i] += str("\nCluster {} represents {}% of the Overall dataset.".format(
            i, cluster_percentages[i]))

        if high_mean_vals[i]:
            cp_text[i] += str("\nIt contains relatively high values of the numeric variables- {}".format(
                str(high_mean_vals[i])[1:-1]))

        if low_mean_vals[i]:
            cp_text[i] += str("\nIt contains relatively low values of the numeric variables- {}".format(
                str(low_mean_vals[i])[1:-1]))

        for k in high_percent_levels.keys():
            if high_percent_levels[k][i]:
                cp_text[i] += str("\nFor the variable '{}', it contains more of categories- {}.".format(
                    k, str(high_percent_levels[k][i])[1:-1]))
            if zero_percent_levels[k][i]:
                cp_text[i] += str("\nFor the variable '{}', it does not contain categories- {}.".format(
                    k, str(zero_percent_levels[k][i])[1:-1]))
        print(cp_text[i])
