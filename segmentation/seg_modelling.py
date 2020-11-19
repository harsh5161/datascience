from sklearn.cluster import KMeans 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.style as style
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
import time
###### End Imports #######

class Segmentation:
    def silplots(X):
        range_n_clusters = [i for i in range(2,11)]
        silhouette_avg_n_clusters = []
        
        sil_score_max = -1 #this is the minimum possible score

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, init = 'k-means++',max_iter=100, random_state=42)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is : %0.2f" %silhouette_avg)
            
            if silhouette_avg > sil_score_max:   #selecting optimal number of clusters where silhoutte score is highest
                sil_score_max = silhouette_avg
                best_n_clusters = n_clusters

            silhouette_avg_n_clusters.append(silhouette_avg)
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

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

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()


        style.use("fivethirtyeight")
        plt.plot(range_n_clusters, silhouette_avg_n_clusters)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("silhouette score")
        plt.show() 
        
        return best_n_clusters

    
      
    def clustering_algorithms(self,pca_components,df):
        #Creating 10 K-Mean models while varying the number of clusters (k)
        #Generating the inertials for 10 clusters 
        print("\nGenerating the inertias for 10 clusters...")
        st=time.time()
        inertias = []
        for k in range(1,11):
            model = KMeans(n_clusters= k, init = 'k-means++',max_iter=100, random_state = 42)
            model.fit(pca_components)
            inertias.append(model.inertia_)
        
        et= time.time()    
        print("\n ..........Done........")
        print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(et-st))) 

        #Generating Elbow Plot for user to see how the cluster intertias are
        print("\nGenerating the elbow plot...")
        plt.figure(figsize=(10,8))
        fig1 = plt.plot(range(1,11),inertias,marker = 'o', linestyle='--')
        fig1= plt.xlabel('Number of Clusters')
        fig1 = plt.ylabel('Inertias')
        fig1 = plt.title('K-Means with Elbow Method')
        fig1.figure.savefig("Elbow.png")
        plt.show()

       
        # Generating avg silhoutte scores and plots
        print("\nGenerating silhoutte plots...")
        st=time.time()
        best_n_clusters= Segmentation.silplots(pca_components)
        et=time.time()
        print("..........Done..........")
        print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(et-st)))
        
        print("\033[1m" + "\nNumber of clusters being generated: ", best_n_clusters)
        print ('\033[0m')  #to remove bold
        
        print("\nPerforming clustering...")
        #Actually generating the clusters
        st=time.time()
        model = KMeans(n_clusters= best_n_clusters, init = 'k-means++', random_state = 42)
        model.fit(pca_components)
        et=time.time()
        print("\n Time taken: ", time.strftime("%H:%M:%S", time.gmtime(et-st))) 

#         df_new = pd.concat([df.reset_index(drop =True),pd.DataFrame(pca_components)], axis =1)
        df_new = df.reset_index(drop =True)
#         pca_list = df_new.columns.values[-14:].tolist()
#         num = pca_components.shape[1]
#         df_new.columns.values[-num:] = [f'Principal-Component {i}' for i in range(0,num)]
        df_new['K-means Segments'] = model.labels_
        df_new.to_csv("Segmentation.csv")

        
        
