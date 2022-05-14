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
from sklearn.manifold import TSNE
#import umap.umap_ as umap
import time

# %% [markdown]
# # ACP

# %%
def amulti_acp_standard(data):
    """
    Centrer et réduire nos données
    
    Args:
        data ([type]): données
    
    """
    #instanciation
    sc = StandardScaler()
    #transformation – centrage-réduction
    Z = sc.fit_transform(data)
    return Z


def amulti_acp_choice_dim(data, Z):
    """
    Réaliser les calculs de l'acp et de tracer le graphique du % cumulé de la variance expliqué par dimension
    
    Args:
        data ([type]): données
        Z ([type]): données transformées
    
    """   
    pca = PCA()

    #calculs
    coord = pca.fit_transform(Z)
    pca.fit_transform(Z)
    print(pca.explained_variance_.shape[0])
    length_i=pca.explained_variance_.shape[0]
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    n = data.shape[0] # nb individus
    p = data.shape[1] # nb variables
    eig = pd.DataFrame(
        {
            "Dimension" : ["Dim" + str(x + 1) for x in range(length_i)], 
            "Valeurs propres" : (n-1) / n * pca.explained_variance_,
            "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
            "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
        }
    )
    print(eig)
    eig.plot.bar(x = "Dimension", y = "% cum. var. expliquée", color="g") # permet un diagramme en barres
    plt.text(5, 18, "17%") # ajout de texte
    plt.axhline(y = 17, linewidth = .5, color = "red", linestyle = "--") # ligne 17 = 100 / 6 (nb dimensions)
    plt.show()
    pca.explained_variance_ratio_
    #cumul de variance expliquée
    plt.plot(np.arange(0,length_i),np.cumsum(pca.explained_variance_ratio_))
    plt.title("Explained variance vs. # of factors")
    plt.ylabel("Cumsum explained variance ratio")
    plt.xlabel("Factor number")
    plt.show()


# %%
#data_study["product_name"] data_study["nutrition_grade_fr"]
#amultiacp_visualizer(WGI_num0, Z, 3, data_study,data_study["product_name"], data_study["nutrition_grade_fr"], "nutrition_grade_fr")
def amultiacp_visualizer(data_acp, Z, dim, data_study,data_stickers, data_group, column_group, cercle_correlation):
    """
    Trace les graphiques pour visualiser les dimensions soit en 3d soit du premier plan factoriel
    
    Args:
        data_acp ([type]): données
        Z ([type]): données transformées
        dim([type]): nombre de dimensions retenues
        data_study([type]): toutes les données (qui contient les étiquettes)
        data_stickers([type]): les données de la variable qui contient les étiquettes des individus
        data_group([type]): toutes les données de la variable groupe
        column_group([type]): nom de la variable groupe
    
    """   
    pca = PCA()
    #calculs
    coord = pca.fit_transform(Z)
    pca.fit_transform(Z)
    WGI_pca = pca.transform(Z)
    # Transformation en DataFrame pandas
    WGI_pca_df = pd.DataFrame({
        "Dim1" : WGI_pca[:,0], 
        "Dim2" : WGI_pca[:,1],
        "Dim3" : WGI_pca[:,2],
        "product": data_stickers,
        "nutrition_grade_fr" : data_group
    })

    # Résultat (premières lignes)
    WGI_pca_df.head()
    WGI_pca_df.plot.scatter("Dim1", "Dim2") # nuage de points
    plt.xlabel("Dimension 1 ") # modification du nom de l'axe X
    plt.ylabel("Dimension 2 ") # idem pour axe Y
    plt.suptitle("Premier plan factoriel") # titre général
    plt.show()
    
    ax = plt.figure(figsize=(16,10))
    ax = plt.axes(projection='3d')
    ax.scatter(
        xs=WGI_pca_df["Dim1"], 
        ys=WGI_pca_df["Dim2"], 
        zs=WGI_pca_df["Dim3"]
    )
    ax.set_xlabel('Dim 1 ')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.show()
    
    data_study2 = data_study.copy()
    # Append the principle components for each entry to the dataframe
    for i in range(0, 3):
        data_study2['PC' + str(i + 1)] = coord[:, i]

    #display(data_acp.head())

    # Show the points in terms of the first two PCs
    g = sns.lmplot(x='PC1',
                   y='PC2',
                   hue=column_group,data=data_study2,
                   fit_reg=False,
                   scatter=True,
                   height=7)
    plt.suptitle("Premier plan factoriel") # titre général

    plt.show()

    if cercle_correlation == 1:
        # Plot a variable factor map for the first two dimensions.
        (fig, ax) = plt.subplots(figsize=(8, 8))
        for i in range(0, pca.components_.shape[1]):
            ax.arrow(0,
                    0,  # Start the arrow at the origin
                    pca.components_[0, i],  #0 for PC1
                    pca.components_[1, i],  #1 for PC2
                    head_width=0.1,
                    head_length=0.1)

            plt.text(pca.components_[0, i] + 0.05,
                    pca.components_[1, i] + 0.05,
                    data_study.columns.values[i])
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis('equal')
        ax.set_title('Variable factor map')
        plt.show()
        
        n = data_acp.shape[0] # nb individus
        p = data_acp.shape[1] # nb variables
        eigval = (n-1) / n * pca.explained_variance_ # valeurs propres
        sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
        corvar = np.zeros((p,p)) # matrice vide pour avoir les coordonnées
        for k in range(p-1):
            corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
        # on modifie pour avoir un dataframe
        cos2var = corvar**2
        print(pd.DataFrame({'id':data_study.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))

        #contributions
        ctrvar = cos2var
        for k in range(p):
            ctrvar[:,k] = ctrvar[:,k]/eigval[k]
        #on n'affiche que pour les trois premiers axes
        print(pd.DataFrame({'id':data_study.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))

        # Load the concrete dataset
        plt.figure()
        visualizer = pca_yellow(scale=True, proj_features=True)
        visualizer.fit_transform(data_acp)
        visualizer.poof()

# %%
def Quali_to_quanti(data,column):
    #creating labelEncoder
    ple = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    data[column]=ple.fit_transform(data[column])+1
    return data


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def tsne_graph(df_tsne, group=0, by=""):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df_tsne)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_tsne['tsne-2d-one'] = tsne_results[:,0]
    df_tsne['tsne-2d-two'] = tsne_results[:,1]
    if group==1:
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=by,
            palette=sns.color_palette("rocket", 5),
            data=df_tsne,
            legend="full",
            alpha=0.3
        )
    else:  
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            #hue="RecencyCluster",
            #palette=sns.color_palette("hls", 10),
            data=df_tsne,
            legend="full",
            alpha=0.3
        )


def umap_graphs(df_acp2):
    features = df_acp2.copy()

    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    umap_3d = UMAP(n_components=3, init='random', random_state=0)

    proj_2d = umap_2d.fit_transform(features)
    proj_3d = umap_3d.fit_transform(features)

    fig_2d = px.scatter(
        proj_2d, x=0, y=1
        #color=df.species, labels={'color': 'species'}
    )
    fig_3d = px.scatter_3d(
        proj_3d, x=0, y=1, z=2
        #color=df.species, labels={'color': 'species'}
    )
    fig_3d.update_traces(marker_size=5)

    fig_2d.show()
    fig_3d.show()