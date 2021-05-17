from sklearn.cluster import KMeans, DBSCAN
# import faiss
import hdbscan
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.style as style
from sklearn.metrics import silhouette_samples, silhouette_score #, davies_bouldin_score
from sklearn.datasets import make_blobs
import time
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
# from cdbw import CDbw
# import joblib
import gap_statistic
from seg_Viz import *

###### End Imports #######

class Segmentation:
    def silplots(X, range_n_clusters):
        
        silhouette_avg_n_clusters = {}         # to store all avg silhoutte scores corresponding to no of clusters
        sil_score_best = -1 #this is the minimum possible score  

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, ax2 = plt.subplots()
#             fig.set_size_inches(18, 7)

#             # The 1st subplot is the silhouette plot
#             # The silhouette coefficient can range from -1, 1 but in this example its only from -0.1 to 1
#             ax1.set_xlim([-0.1, 1])
#             # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly. 
#             ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
            s=time.time()
            clusterer = KMeans(n_clusters=n_clusters, init = 'k-means++',max_iter=100, random_state=42)
            cluster_labels = clusterer.fit_predict(X)
            e=time.time()
            print("\n  Time taken to run kmeans inside silplots: ", time.strftime("%H:%M:%S", time.gmtime(e-s)))

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters 
            s=time.time()
            silhouette_avg = silhouette_score(np.array(X), cluster_labels)
            e=time.time()
            print("\n  Time taken to calculate silhouette_avg: ", time.strftime("%H:%M:%S", time.gmtime(e-s)))
            
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is : %0.4f" %silhouette_avg)

            silhouette_avg_n_clusters[n_clusters]=silhouette_avg #storing in dict
    
#             # Compute the silhouette scores for each sample
#             s=time.time()
#             sample_silhouette_values = silhouette_samples(X, cluster_labels)
#             e=time.time()
#             print("\n  Time taken to calculate silhouette sample: ", time.strftime("%H:%M:%S", time.gmtime(e-s)))

#             y_lower = 10
#             s=time.time()
#             for i in range(n_clusters):
#                 # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
#                 ith_cluster_silhouette_values = \
#                     sample_silhouette_values[cluster_labels == i]
#                 ith_cluster_silhouette_values.sort()
                
#                 size_cluster_i = ith_cluster_silhouette_values.shape[0]
#                 y_upper = y_lower + size_cluster_i
#                 color = cm.nipy_spectral(float(i) / n_clusters)
#                 ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                                   0, ith_cluster_silhouette_values,
#                                   facecolor=color, edgecolor=color, alpha=0.7)

#                 # Label the silhouette plots with their cluster numbers at the middle
#                 ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#                 # Compute the new y_lower for next plot
#                 y_lower = y_upper + 10  # 10 for the 0 samples
#             e=time.time()
#             print("\n  Time taken in small for loop: ", time.strftime("%H:%M:%S", time.gmtime(e-s)))

#             ax1.set_title("The silhouette plot for the various clusters.")
#             ax1.set_xlabel("The silhouette coefficient values")
#             ax1.set_ylabel("Cluster label")

#             # The vertical line for average silhouette score of all the values
#             ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#             ax1.set_yticks([])  # Clear the yaxis labels / ticks
#             ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("KMeans clustering with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()
        
        # function to return key for any value in a dictionary
        def get_key(val):
            for key, value in silhouette_avg_n_clusters.items():
                 if val == value:
                    return key
        # out of the 3 possible optimal no of clusters picking the one with the highest sil score or the one where max_sil_score-sil_score<=0.1 and n < n_for_max_sil_score        
        s=time.time()       
        sil_score_best=max(silhouette_avg_n_clusters.values())
        selected_score= sil_score_best
        best_n_clusters= get_key(sil_score_best)
        for key,value in silhouette_avg_n_clusters.items():
            if sil_score_best-value<=0.1 and key<get_key(sil_score_best):
                selected_score=value
                best_n_clusters=key
        e=time.time()
        print("\n  Time taken to select best_n_clusters ", time.strftime("%H:%M:%S", time.gmtime(e-s)))
        
        return best_n_clusters,selected_score     #returning optimal no of clusters and corresponding silhoutte score 

    def Kmeans(pca_components,df):
        kmeans_dict={}  #dict to store kmeans details, keys:['no_of_clusters','sil_score','cluster_labels', 'segdata','cluster_percentages']

#             #Creating 10 K-Mean models while varying the number of clusters (k)
#             #Generating the inertials for 10 clusters 
#             print("\nGenerating the inertias for 10 clusters...")
#             st=time.time()
#             inertias = []
#             for k in range(1,11):
#                 model = KMeans(n_clusters= k, init = 'k-means++',max_iter=100, random_state = 42)
#                 model.fit(pca_components)
#                 inertias.append(model.inertia_)

#             et= time.time()    
#             print("\n ..........Done........")
#             print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(et-st))) 

#             #Generating Elbow Plot for user to see how the cluster intertias are
#             print("\nGenerating the elbow plot...")
#             plt.figure(figsize=(10,8))
#             fig1 = plt.plot(range(1,11),inertias,marker = 'o', linestyle='--')
#             fig1= plt.xlabel('Number of Clusters')
#             fig1 = plt.ylabel('Inertias')
#             fig1 = plt.title('K-Means with Elbow Method')
#             fig1.figure.savefig("Elbow.png")
#             plt.show()

        #finding possible optimal no of clusters(K) using 3 gap methods then sending these 3 Ks through silplots() to get best_n_clusters
        optimalK = gap_statistic.OptimalK() #n_jobs=4, parallel_backend='joblib'
        n_clusters = optimalK(pca_components, cluster_array=np.arange(2, 11))
#         print(optimalK.gap_df)
        optimalK.plot_results()         
        n_max_gap=n_clusters # method1: n which maximises gap value
        n_max_gapstar=int(optimalK.gap_df[optimalK.gap_df['gap*']==optimalK.gap_df['gap*'].max()]['n_clusters']) # method2: n which maximise gap* value
        n_min_posdiff= int(optimalK.gap_df[optimalK.gap_df['diff']>0]['n_clusters'].min())  # method3: minimum n for which diff is positive
        possibleKs=sorted(list(set([n_max_gap, n_max_gapstar, n_min_posdiff]))) # sorting and removing duplicate n values and storing in list
        s=time.time()
        best_n_clusters,selected_score= Segmentation.silplots(pca_components,possibleKs) #finding sil scores only for n in possibleKs
        e=time.time()
        print("\nTime taken for entire silplots func: ", time.strftime("%H:%M:%S", time.gmtime(e-s)))
  
        # running kmeans 
        model = KMeans(n_clusters= best_n_clusters, init = 'k-means++', random_state = 42)
        model.fit(pca_components)
        
        #storing cluster labels with dataset
        df_new = df.reset_index(drop =True)
        df_new['Segments (Clusters)'] = model.labels_
        df_new.to_csv(f"KmeansResult.csv",index=False)
        #storing all results
        kmeans_dict['no_of_clusters']=best_n_clusters
        kmeans_dict['sil_score']= selected_score
        kmeans_dict['cluster_labels']= model.labels_
        kmeans_dict['segdata']=df_new

        return kmeans_dict   #returns kmeans details

    
    def HDbscan(pca_components,df):
        hdbscan_dict={} # keys: ['no_of_clusters','val_index_score','cluster_labels','segdata','noise','cluster_percentages']
        s=time.time()        
        model= hdbscan.HDBSCAN(min_cluster_size=int(0.05*pca_components.shape[0]), min_samples=1 )
        model.fit(pca_components)
        clusters = model.labels_ 
        e=time.time()
        print("\n  Time taken to perform hdbscan ", time.strftime("%H:%M:%S", time.gmtime(e-s)))
        s=time.time()        
        val_index= hdbscan.validity_index(np.array(pca_components), clusters)
        e=time.time()
        print("\n  Time taken to calculate validity index ", time.strftime("%H:%M:%S", time.gmtime(e-s)))
        df_new = df.reset_index(drop=True)
        df_new['Segments (Clusters)'] = clusters  #storing cluster labels with dataset
        noise= df_new[df_new['Segments (Clusters)']==-1]  
        df_new= df_new[df_new['Segments (Clusters)']!=-1]  # removing noise (where cluster number is -1)
        df_new.to_csv(f"DBSCANResult.csv",index=False)

        if -1 in np.unique(clusters):
            hdbscan_dict['no_of_clusters']=len(np.unique(clusters))-1 #excluding noise if present
        else: 
            hdbscan_dict['no_of_clusters']=len(np.unique(clusters))
        hdbscan_dict['val_index_score']= val_index
        hdbscan_dict['cluster_labels'] = clusters #includes noise (cluster_label = -1)
        hdbscan_dict['segdata'] = df_new
        hdbscan_dict['noise']= noise

#         silhouette_avg= silhouette_score(np.array(pca_components), clusters)
#         cdbw_score = CDbw(np.array(pca_components), clusters, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False)
#         print("\nSilhoutte score for hdbscan(higher the better, ranges from -1 to 1): ", silhouette_avg) 
#         print("Davies Bouldin score for hdbscan(smaller the better): ", davies_bouldin_score(np.array(pca_components), clusters))  
#         print("**warning: Silhoutte score and Davies Bouldin score only work well when the clusters are shaped round")  
#         print("\nvalidity index(higher the better, ranges from -1 to 1): ",val_index)
#         print("\nCDbw index(higher the better): " , cdbw_score)       

        return hdbscan_dict    #returns hdbscan details

    def clustering_algorithms(self,pca_components,df):
        clustering_algos = {}   # dict to store all algo details
        
        #Performing KMeans and printing silhouette score plots
        print('\nPerforming KMeans...')
        starttime=time.time()
        clustering_algos['KMEANS'] = Segmentation.Kmeans(pca_components,df) #returns kmeans details in a dictionary
        endtime=time.time()
        print("\nTime taken for KMEANS: ", time.strftime("%H:%M:%S", time.gmtime(endtime-starttime)))
        #Performing HDBSCAN 
        print('\nPerforming HDBSCAN...')
        starttime=time.time()
        clustering_algos['HDBSCAN'] = Segmentation.HDbscan(pca_components,df) #returns HDBSCAN details in a dictionary
        endtime=time.time()
        print("\nTime taken for HDBSCAN: ", time.strftime("%H:%M:%S", time.gmtime(endtime-starttime)))
        
        # printing results for user
        print("\n\033[1m---------------- RESULTS FOR KMEANS -------------------\033[0m")
        print("Number of clusters being generated: ", clustering_algos['KMEANS']['no_of_clusters'])
        print("Cluster magnitudes: ")
        clustering_algos['KMEANS']['cluster_percentages']=ClusterMags(clustering_algos['KMEANS']['segdata'],'KMEANS')
        print('Scatter plot showing the formed clusters: ')
        clusters_scatter_plot(pca_components, clustering_algos['KMEANS']['cluster_labels'])
        print("Silhouette score(higher the better, ranges from -1 to 1): ",clustering_algos['KMEANS']['sil_score'])
#         print("**Silhoutte score only works well when the clusters are shaped round")  
        print("\n\033[1m---------------- END OF KMEANS RESULTS -------------------\033[0m")
        
        print("\n\033[1m---------------- RESULTS FOR HDBSCAN -------------------\033[0m")
        print("Number of clusters being generated: ", clustering_algos['HDBSCAN']['no_of_clusters'])
        print("Cluster magnitudes: ")
        clustering_algos['HDBSCAN']['cluster_percentages']=ClusterMags(clustering_algos['HDBSCAN']['segdata'],'HDBSCAN')
        print('Scatter plot showing the formed clusters: ')
        clusters_scatter_plot(pca_components, clustering_algos['HDBSCAN']['cluster_labels'])
        print("\nValidity index(higher the better, ranges from -1 to 1): ",clustering_algos['HDBSCAN']['val_index_score'])
        print("**Validity index only is used in density-based clustering, works well with arbitrarily shaped clusters")
        print("**Silhoutte score does not work well with arbitrarily shaped clusters.")
        if not clustering_algos['HDBSCAN']['noise'].empty:
            print("\nThe following "+ str(clustering_algos['HDBSCAN']['noise'].shape[0])+" rows were dropped because they were classified as noise:\n\n",clustering_algos['HDBSCAN']['noise'])
        print("\n\033[1m---------------- END OF HDBSCAN RESULTS -------------------\033[0m")
        
        print("\n\033[1mTips for choosing algorithm:\033[0m\n1) Good clusters are well seperated and do not overlap much. If the scatter plot shows two or more distinct clusters that means the algorithm has successfully classified all data points into distinct clusters which have different characteristics. If the clusters are overlapping then the data points in those clusters may not have very different characteristics and therefore may not be very useful. \n2) Look at the validation metrics to see how well that clustering algorithm has performed. Validation metrics are useful in giving a basic idea about how well seperated the clusters are. But we should remember that a validation metric alone is not the sole determinant of good clustering.\n3) Sometimes all the data points maybe very close together(densely packed) or may have uniform density throughout and may not form very distinguishable clusters. In such cases density-based clustering algorithms like HDBSCAN may classify almost all data points into one big cluster and other clusters(if formed) may contain a very small amount of data points. In this case centroid based algorithms like K-MEANS may give a better result. However it should be kept in mind that it depends on the dataset as well. Some datasets just dont have any clusters or may only have one big cluster and one very small cluster.\n4) In some cases HDBSCAN may only detect small clusters and classify a majority of data points as noise. This again depends on the dataset. If this is not the type of result expected, then K-MEANS may be more useful.\n5) Validation metrics, Scatter plots, Cluster magnitudes, and the dataset, all should be kept in mind while selecting the algorithm." ) 
              
        return clustering_algos

        
        
