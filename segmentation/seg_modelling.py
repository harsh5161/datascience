from sklearn.cluster import KMeans 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
###### End Imports #######

class Segmentation:
    def clustering_algorithms(self,pca_components,df):
        #Creating 10 K-Mean models while varying the number of clusters (k)
        #Generating the inertials for 10 clusters 
        inertias = []
        for k in range(1,11):
            model = KMeans(n_clusters= k, init = 'k-means++', random_state = 42)
            model.fit(pca_components)
            inertias.append(model.inertia_)

        #Generating Elbow Plot for user to see how the cluster intertias are
        plt.figure(figsize=(10,8))
        fig1 = plt.plot(range(1,11),inertias,marker = 'o', linestyle='--')
        fig1= plt.xlabel('Number of Clusters')
        fig1 = plt.ylabel('Inertias')
        fig1 = plt.title('K-Means with Elbow Method')
        fig1.figure.savefig("Elbow.png")
        plt.show()

        sil_score_max = -1 #this is the minimum possible score
        for n_clusters in range(2,11):
            model = KMeans(n_clusters = n_clusters, init='k-means++',random_state = 42)
            labels = model.fit_predict(pca_components)
            sil_score = silhouette_score(pca_components, labels)
            print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
            if sil_score > sil_score_max:
                sil_score_max = sil_score
                best_n_clusters = n_clusters

        #Actually generating the clusters
        model = KMeans(n_clusters= best_n_clusters, init = 'k-means++', random_state = 42)
        model.fit(pca_components)

        df_new = pd.concat([df.reset_index(drop =True),pd.DataFrame(pca_components)], axis =1)
        pca_list = df_new.columns.values[-14:].tolist()
        num = pca_components.shape[1]
        df_new.columns.values[-num:] = [f'Principal-Component {i}' for i in range(0,num)]
        df_new['K-means Segments'] = model.labels_
        df_new.to_csv("Segmentation.csv")
