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
from sklearn.metrics import roc_auc_score, roc_curve, auc


def pipelineModel_TrainingSet(X,Y, stand, test_size_i):
    """
        Diviser le jeu de données en train/test avec option de standardisation
    
        Args:
            X (_type_): _description_
            Y (_type_): _description_
            stand (_type_): _description_
            test_size_i (_type_): _description_
    
        Returns:
            _type_: _description_
    
        Example:
            function(params)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
    if stand==1:
        std_scale = preprocessing.StandardScaler().fit(X_train)

        X_train_std = std_scale.transform(X_train)
        X_test_std = std_scale.transform(X_test)
        return X_train_std, X_test_std, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test


## Metrics
def plot_roc_curve(fper, tper):
    """
    _summary_
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
         function(params)"""
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe Roc')
    plt.legend()
    plt.show()


def pipeline_training_metrics(y_prob, y_test):
    """
    Calculs de metrics
                
    Args:
         X (_type_): _description_
         Y (_type_): _description_
         stand (_type_): _description_
         test_size_i (_type_): _description_
               
    Returns:
         _type_: _description_
                
    Example:
        yhat = model.predict_proba(X_train)[:,1]
        y_pred = np.where(yhat > 0.5, 1, 0)
        pipeline_training_metrics(y_pred, y_train)"""
    # On créé un vecteur de prédiction à partir du vecteur de probabilités
    y_pred = np.where(y_prob > 0.5, 1, 0) 

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    #print("false positive rate : ", false_positive_rate)
    #print("true positive rate : ", true_positive_rate)
    x_rate = false_positive_rate
    y_rate = true_positive_rate 

    # This is the ROC curve
    plot_roc_curve(x_rate,y_rate)
    
    CM = metrics.confusion_matrix(y_test, y_pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    metier = (FN*0.9+FP*0.1)/(FN+FP+TN+TP)
    
    print('FULL Métrique Métier : ', metier)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    #Matrice de confusion
    conf = metrics.confusion_matrix(y_test, y_pred)
    conf
    sns.heatmap(conf, square=True, annot=True, cbar=False)
            #, xticklabels=list(iris.target_names)
            #, yticklabels=list(iris.target_names))
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    plt.title('Matrice de confusion %')
    plt.show()
    
    sns.heatmap(conf/np.sum(conf), annot=True, 
            fmt='.2%', cmap='Blues')
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    plt.title('Matrice de confusion %')
    plt.show()
    
    print("\nSur le jeu de test auc : {:.3f}".format(metrics.roc_auc_score(y_test, y_pred)))
    print("\nSur le jeu de test f1_score : {:.3f}".format(metrics.f1_score(y_test, y_pred)))
    print("\nSur le jeu de test precision : {:.3f}".format(metrics.precision_score(y_test, y_pred)))
    print("\nSur le jeu de test recall : {:.3f}".format(metrics.recall_score(y_test, y_pred)))
