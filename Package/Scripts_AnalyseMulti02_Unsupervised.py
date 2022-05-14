#!/usr/bin/env python
# coding: utf-8
# %%

# # Présentation générale du jeu de données

# Installation des différents packages :

# pip install -r requirements.txt

# Chargement des librairies suivantes qu'on utilisera pour l'analyse de données :

# %%
import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats
from scipy.stats import pearsonr
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.features import ParallelCoordinates
from plotly.graph_objects import Layout
import jenkspy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import PCA as pca_yellow
from yellowbrick.style import set_palette
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

# %% [markdown]
# # Classification ascendante hiérarchique

# %%
def amulti_cah_linkage(data):
    """
        Trace le dendogramme de la CAH
    
    Args:
        data ([type]): données
    """
    std_scale = preprocessing.StandardScaler().fit(data)
    X_scaled = std_scale.transform(data)
    #générer la matrice des liens
    Z = linkage(X_scaled,method='ward',metric='euclidean')
    #affichage du dendrogramme
    plt.title("CAH")
    dendrogram(Z,labels=data.index,color_threshold=0)
    plt.show()
    return Z


# %%
#amulti_cah_create_classe(Z, test, test.index, 350, 12, "nutrition_grade_fr" )
def amulti_cah_create_classe(Z, data, data_index, threshold, n_last_clusters, group, group_exist ):
    """
    Trace le dendogramme tronqué de la CAH et ajoute le nouveau classement dans une base cible
    
    Args:
        Z([type]): données transformées
        data ([type]): données avec les index
        threshold ([type]): seuil choisi
        n_last_clusters ([type]): nb des p derniers clusters fusionnés 
        group ([type]): variable groupe
    
    """   
    #matérialisation des 5 classes (hauteur t = 350)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(Z,labels=data_index,orientation='left',color_threshold=threshold,
               truncate_mode = 'lastp' ,   # afficher uniquement les p derniers clusters fusionnés 
        p = n_last_clusters ,   # afficher uniquement les p derniers clusters fusionnés 

        leaf_font_size = 12. , 
        show_contracted = True
              )
    plt.show()

    #découpage à la hauteur t = 350 ==> identifiants de 5 groupes obtenus
    groupes_cah = fcluster(Z,t=threshold,criterion='distance')
    print(groupes_cah)
    #index triés des groupes
    idg = np.argsort(groupes_cah)
    #affichage des observatbbions et leurs groupes
    d_index=data_index
    info_groupe=pd.DataFrame(d_index[idg],groupes_cah[idg])
    WGI_pca_k2 = data.assign(classe = info_groupe.index)
    
    if group_exist==1:
        t = pd.crosstab(WGI_pca_k2[group], WGI_pca_k2.classe, normalize = "columns")
        t = t.assign(col_group = t.index)
        tm = pd.melt(t, id_vars = "col_group")
        tm = tm.assign(value = 100 * tm.value)

        sns.catplot("col_group", y = "value", col = "classe", data = tm, kind = "bar")
    else:
        return WGI_pca_k2

# %% [markdown]
# # Kmeans

# %%
#WGI_num0
def amulti_kmeans_elbow(data):  
    """
    Trace le graphique pour choisir le k grâce à la méthode du coude
    
    Args:
        data ([type]): données
    
    """
    X = data
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=(2,15), metric='calinski_harabasz', timings=False, locate_elbow=False
    )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    
    # Generate synthetic dataset with 8 random clusters
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,15))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure


# %%
#kmeans2=amulti_kmeans_visualizer(WGI_num0,data_study, 5, "nutrition_grade_fr")
def amulti_kmeans_visualizer(data,data_study, nb_clusters, group):
    """
    Trace le graphique pour visualiser les résultats des classes en fonction d'une variable groupe
    
    Args:
        data ([type]): données avec les index
        data_study ([type]): données de l'étude
        nb_clusters ([type]): nombre de clusters choisis
        group ([type]): variable groupe
    
    """   
    kmeans2 = KMeans(n_clusters = nb_clusters)
    kmeans2.fit(scale(data))
    print(kmeans2.labels_)
    pd.Series(kmeans2.labels_).value_counts()
    print(kmeans2.cluster_centers_)
    WGI_k2 = data.assign(classe = kmeans2.labels_)
    display(WGI_k2.groupby("classe").mean())
    WGI_pca_k2 = data_study.assign(classe = kmeans2.labels_)
    t = pd.crosstab(WGI_pca_k2[group], WGI_pca_k2.classe, normalize = "columns")
    t = t.assign(group = t.index)
    tm = pd.melt(t, id_vars = "group")
    tm = tm.assign(value = 100 * tm.value)

    sns.catplot(x="group", y = "value", col = "classe", data = tm, kind = "bar")
    return kmeans2
    


# %%
#amulti_kmeans_visualizer_acp(WGI_pca_df,kmeans2)
def amulti_kmeans_visualizer_acp(data_acp,kmeans_dt):
    """
    Tracer le nuage de point de l'acp en fonction des nouvelles classes créées avec kmeans
    Args:
        data_acp ([type]): données de l'acp
        kmeans_dt ([type]): données de kmeans
    
    """   
     #acp
    WGI_pca_k2 = data_acp.assign(classe = kmeans_dt.labels_)
    WGI_pca_k2.plot.scatter(x = "Dim1", y = "Dim2", c = "classe", cmap = "Accent")
    plt.show()
    #acp 3d
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    WGI_pca_k2 = WGI_pca_df.assign(classe = kmeans_dt.labels_)
    ax.scatter(
        xs=WGI_pca_k2["Dim1"], 
        ys=WGI_pca_k2["Dim2"],
        c=WGI_pca_k2["classe"],
        cmap='tab10'
    )
    ax.set_xlabel('Dim 1 (21%)')
    ax.set_ylabel('Dim 2 (24%)')
    plt.show()


# %% [markdown]
# # Mix Kmeans / CAH

# %%
#amulti_mix_kmeans_cah_calcul(WGI_num0, nb_clusters)
def amulti_mix_kmeans_cah_calcul_kmeans(data, nb_clusters):
    """
    calculs avec kmeans pour retourner les centroids
    Args:
        data_acp ([type]): données de l'acp
        kmeans_dt ([type]): données de kmeans
    
    """   
    kmeans2 = KMeans(n_clusters = nb_clusters)
    kmeans2.fit(scale(data))
    labels=kmeans2.labels_
    centroid=kmeans2.cluster_centers_
    return kmeans2
    
def amulti_mix_kmeans_cah_calcul_cah(centroid, threshold):
    """
    Tracer du dendogramme obtenu sur les centroids 
    Args:
        centroid ([type]): centroid de kmeans
        threshold ([type]): seuil du dendogramme
    
    """
    Z = linkage(centroid,method='ward',metric='euclidean')
    #affichage du dendrogramme
    plt.title("CAH")
    dendrogram(Z,color_threshold=threshold)
    plt.show()
    return Z



# %%
#WGI_num0= data
#  labels=kmeans2.labels_
#  centroid=kmeans2.cluster_centers_
def amulti_mix_kmeans_cah_create_classe(data, data_study, Z, kmeans2, threshold, nb_clusters, group, group_exist=0):
    """
    calculer les données avec les classes obtenues du mix kmeans-cah
    Args:
        data([type]): données
        data_study([type]): données de l'étude
        Z([type]): données transformées
        kmeans2 ([type]): résultat de kmeans
        threshold ([type]): seuil
        group ([type]): nom de la variable groupe pour comparer les résultats
    
    """   
    range_clusters=[str(i) for i in range(0,nb_clusters)]
    #découpage à la hauteur t = 11 ==> identifiants de 5 groupes obtenus
    groupes_cah = fcluster(Z,t=threshold,criterion='distance')
    print(groupes_cah)
    centroid = kmeans2.cluster_centers_
    #index triés des groupes
    idg = np.argsort(groupes_cah)
    #affichage des observatbbions et leurs groupes
    info_groupe=pd.DataFrame(centroid[idg],groupes_cah[idg], columns=range_clusters)
    
    #classe kmeans et index des data
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data.index
    cluster_map['cluster'] = kmeans2.labels_
    
    #kmeans class et centroid
    cols=list(np.concatenate((["cluster"],range_clusters), axis=None))
    centroid_map = pd.DataFrame(columns=cols)
    for i in range(0,max(cluster_map["cluster"])+1):
        centroid_map.loc[i] = i, kmeans2.cluster_centers_[i][0],kmeans2.cluster_centers_[i][1], kmeans2.cluster_centers_[i][2], kmeans2.cluster_centers_[i][3],kmeans2.cluster_centers_[i][4],kmeans2.cluster_centers_[i][5]

    #classe cah
    info_groupe["classe_cah"]=info_groupe.index
    
    #merge entre centroid et classe cah
    data_merge = pd.merge(centroid_map, info_groupe, how="left", on=range_clusters, indicator=True,  suffixes=('', '_del'))
    data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
    cah=data_result.copy()
    
    #merge entre cah et individus avec centroid
    data_merge = pd.merge(cluster_map, cah, how="left", on=["cluster"], indicator=True,  suffixes=('', '_del'))
    data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
    mix_kmeans_cah=data_result.copy()
    
    #assign les classes cah au data
    WGI_k2 = data_study.assign(classe = mix_kmeans_cah.classe_cah)
    WGI_k2.groupby("classe").mean()

    if group_exist==1:
        #graphique en fonction d'un groupe
        t = pd.crosstab(WGI_k2[group], WGI_k2.classe, normalize = "columns")
        t = t.assign(group = t.index)
        tm = pd.melt(t, id_vars = "group")
        tm = tm.assign(value = 100 * tm.value)

        sns.catplot("group", y = "value", col = "classe", data = tm, kind = "bar")
    

# %%
def Quali_to_quanti(data,column):
    #creating labelEncoder
    ple = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    data[column]=ple.fit_transform(data[column])+1
    return data


# %% [markdown]
# # Knn classifier

# %%
#features : list quanti
#var_to_pred : Nutgrade_encoded
#nb_neighbors:5
def knn_classifier_result_accuracy(features, var_to_pred, nb_neighbors):
    """
    calculer les résultats knn classifier sur des data
    Args:
        features([type]): listes variables
        var_to_pred([type]): variable à prédire
        nb_neighbors([type]): nombre de k
    
    """   
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(features, var_to_pred, test_size=0.3) # 70% training and 30% test

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=nb_neighbors)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#features_test = data quanti in predict
#pred=knn_classifier_predict_data(data_knn_test,features,features_test, Nutgrade_encoded,"nutrition_grade_fr", 5)
def knn_classifier_predict_data(data_knn_test,features,features_test, var_to_pred,var_to_pred_name, nb_neighbors):
    """
    calculer une prediction avec knn classifier
    Args:
        data_knn_test([type]): données à prédire
        features([type]): listes variables
        features_test([type]): listes variables contenues dans les data à prédire
        var_to_pred([type]): données de la variable à prédire
        var_to_pred_name([type]): nom de la variable à prédire
        nb_neighbors([type]): nombre de k
    
    """   
    model = KNeighborsClassifier(n_neighbors=5)
    # Train the model using the training sets
    model.fit(features,var_to_pred)
    predicted= model.predict(features_test) # 0:Overcast, 2:Mild
    data_knn_test=data_knn_test.assign(var_to_pred_name=predicted)
    return data_knn_test


def kmsplus_iter(stand, X, cluster, km_max_iter, silhouette_visu):
    if stand == 1:
        std_scale = preprocessing.StandardScaler().fit(X)
        X = std_scale.transform(X)
    # 1. Choisissons le clustering avec kmeans 
    myclust = KMeans(init='k-means++', n_clusters=cluster, 
                     random_state=0, max_iter=km_max_iter) # default 300
    myclust.fit(X)
    # 2. Visualisation
    # On commence par réduire la dimension des données avec tSNE. On scale d’abord les données :
    # Puis on applique tSNE aux données scalées :
    #from sklearn import manifold
    #tsne = manifold.TSNE(n_components=2, init='pca')
    #X_trans = tsne.fit_transform(X_scaled)
    # 3. Évaluation
    # Pour l’évaluation intrinsèque, je choisis le coefficient de silhouette :
    silhouette = metrics.silhouette_score(X, myclust.labels_)
    print("Silhouette Coefficient: ",silhouette, " - nb clusters : ", cluster)
    if silhouette_visu == 1:
        visualizer = SilhouetteVisualizer(myclust)
        visualizer.fit(X)    # Fit the data to the visualizer
        visualizer.poof()    # Draw/show/poof the data
        # Pour la comparaison aux étiquettes, je choisis l’indice de Rand ajusté :
        #print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, myclust.labels_))
    print("-----------------------------------------------------------------------------------")
    return silhouette, cluster


#kmplus_assignclass_graphs(X, 4, 500,
#["RevenueCluster", "FrequencyCluster", "RecencyCluster", "review_score_mean_mean"])
def kmplus_assignclass(df, X, km_n_clusters, km_max_iter):
    myclust = KMeans(init='k-means++',n_clusters=km_n_clusters, random_state=0, max_iter=km_max_iter) # default 300
    myclust.fit(X)
    km_df = df.assign(classe = myclust.labels_)
    return km_df


#graphs_profils(df_mix_cah_km,1,4, ["RevenueCluster", "FrequencyCluster", "RecencyCluster", "review_score_mean_mean"])
def graphs_profils(df, min_cluster, n_clusters, graph_categories):
    for i in range(min_cluster,n_clusters+1):       
        categories = graph_categories
        info=df[graph_categories].loc[df["classe"]==i].mean()
        df_graph = pd.DataFrame(dict(
            r=info,
            theta=categories))
        fig = px.line_polar(df_graph, r='r', theta='theta',
                            line_close=True)
        fig.update_traces(fill='toself')
        fig.show()

# %%
# lda_word_corpus_texts(reviews["words_token_lem"])
def lda_word_corpus_texts(df_var):
    """
    construire les variables pour la LDA
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
         function(params)"""
    # Create a dictionnaire
    id2word = corpora.Dictionary(df_var)
    # Create a Corpus
    texts = df_var
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1])
    return id2word, texts, corpus


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    calculer le score de cohérence pour choisir le nombre de topics en général, il faut prendre 
    le nombre après le premier pic
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
        compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=10, step=1)"""
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=1, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def choose_topics_lda(id2word, corpus, texts):
    """
    calculer le score de cohérence pour choisir le nombre de topics en général, il faut prendre 
    le nombre après le premier pic
    + graphique
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
         function(params)"""
    # Calculer les scores de cohérence entre 2 et 10 topics (Cela peut prendre un certain temps)
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=10, step=1)

    # Afficher un graphique
    limit=10
    start=2
    step=1
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Affchier les coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))  #le 4 correspond au nombre de décimales du c

# %%
# model_lda(corpus, id2word, num_topics, alpha, 10)
def model_lda(corpus, id2word, num_topics, alpha, passes):
    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=passes,
                                            alpha='auto',
                                            per_word_topics=True)
    print(lda_model.print_topics())
    return lda_model

# %%
#compute_results(lda_model, corpus, reviews["words_token_lem"], id2word, 'c_v')
def compute_results(lda_model, corpus, texts, id2word, coherence):
    """
    calcul des metriques pour la lda
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
         function(params)"""
    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence=coherence)
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return perplexity, coherence_lda


# %% 
def launch_model(corpus, texts, id2word, passes, coherence, n_topics, resultat_lda):
    """
    lancer la lda
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
         function(params)"""
    for topic in n_topics:
        lda_model = model_lda(corpus, id2word, topic, 'auto', passes)
        perplexity, coherence_lda = compute_results(lda_model, corpus, texts, id2word, coherence)
        resultat_lda.append({ 'topic':topic, 'perplexity': perplexity, 'coherence_lda': coherence_lda })    
