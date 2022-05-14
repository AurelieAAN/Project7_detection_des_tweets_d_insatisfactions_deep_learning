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


# %%

def data_transformcol_string(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str:
            try:
                data[col]=data[col].astype('|S')
            except:
                print("error - caractères spéciaux : "+col)
                continue
                
def data_fillNA_string(data, value, excluded):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str and col not in excluded:
            try:
                data[col].fillna(value, inplace=True)
            except:
                print("error - update na : "+col)
                continue
                
def data_uniqueone_string(data):
    """
    Supprimer toutes les variables avec une colonne unique
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    result=data
    for col in data.columns:
        if data[col].nunique()<= 1:
            print("deleted unique colonne : "+col)
            result=data.drop(col, axis=1,inplace=True)
        else:
            print(col+" - count unique : "+str(data[col].nunique()))
    return result

# %%
def data_missingTab(data):
    """
    Tableau de données manquantes en %
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': round(percent_missing,2)})
    
    return missing_value_df


# %%
def data_majuscule(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    result = data.applymap(lambda s:s.upper() if type(s) == str else s)
    return result


# %%


def del_Nan(data, seuil, delete_i, all_freq):
    """
    Détection des données manquantes en fonction d'un seuil
    
    
    
    Args:
        data ([type]): données
        seuil ([type]): entre 0 et 1
        delete ([type]): 0 ou 1
        all_freq ([type]): resultat de toutes les colonnes
    
    Exemple:
    del_Nan(data, 0.8,0,0)
    
    """
    tab = pd.isnull(data).sum()/len(data)
    s=tab.to_frame(name='freq').reset_index()
    if all_freq==0 and delete_i==0:
        return s[s['freq']>=seuil]
    elif all_freq==1 and delete_i==0:
        return s
    elif delete_i==1:
        s=s[s['freq']>=seuil]
        col_del=list(s['index'])
        data=data.drop(list(s['index']), axis=1, inplace=True)
        print("Columns deleted : ", col_del)
# %%
#créons une fonction
def matrix_vm(data_i, fig_i, color_i):
    """
    Graphique Matrice des données manquantes
    
    
    
    Args:
        data_i ([type]): données
        fig_i ([type]): taille du graphique
        color_i ([type]): couleur
    
    Exemple:
    matrix_vm(data, (14,8), (0.564, 0.823, 0.631))
    
    """
    missingno.matrix(data_i, figsize=fig_i, fontsize=12, color=color_i, sparkline=False).set_title("Matrice des données manquantes")
    gray_patch = mpatches.Patch(color=color_i, label='Données présentes')
    white_patch = mpatches.Patch(color='white', label='Données manquantes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=[gray_patch, white_patch])


# %%

def graph_stripplot(data_i,column_i, title_i, scale_i, color_i):
    """
    Nuage de point d'une variable quantitative (volume de données important)
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        title_i ([type]): titre
        scale_i ([type]): taille du graphique
    
    
    
    """
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=scale_i)
    ax = sns.stripplot(y=column_i, data=data_i,
                       size=4, color=color_i).set_title(title_i)




# %%



def graph_boxenplot(data_i, column_i, color_i,title_i,scale_i):
    """
    boxenplot : plus quantile que le boxplot (même principe: valeur de lettre)
    
    
    
    Args:
        data_i ([type]): data
        column_i ([type]): colonne
        color_i ([type]): couleur
        title_i ([type]): titre
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_boxenplot(data, "circonference_cm", "g","Boxenplot de la circonférence en cm des arbres",(8,6))
    
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=scale_i)
    sns.boxenplot(x=column_i, 
                  color=color_i,
                  scale="linear", data=data_i)
    plt.title(title_i)



# %%


def outliers(data, var, delete_i):
    """
    Détection des valeurs aberrantes.
    Construit un tableau avec :
    - l'écart-interquartile de la valeur haute et basse
    - le nombre et le % d'outliers détectés
    
    
    Args:
        data : données
        s ([type]): données d'une colonne
        delete_i ([type]): 1/0 : yes/no
    
    Exemple:
    outliers(data.circonference_cm)
    
    
    """
    s= var.loc[pd.isna(var)==False]
    iqr = (np.quantile(s, 0.75))-(np.quantile(s, 0.25))
    upper_bound = np.quantile(s, 0.75)+(1.5*iqr)
    print(np.quantile(s, 0.25))
    lower_bound = np.quantile(s, 0.25)-(1.5*iqr)
    f = []
    for i in s:
        if i > upper_bound:
            f.append(i)
        elif i < lower_bound:
            f.append(i)
    sums = len(f)
    pros = len(f)/len(s)*100
    d = {'IQR':iqr,
         'Upper Bound':upper_bound,
        'Lower Bound':lower_bound,
        'Sum outliers': sums,'percentage outliers':pros}
    d = pd.DataFrame(d.items(),columns = ['sub','values'])
    if delete_i==1:
        data=data.loc[(var <= upper_bound) | pd.isna(var)==True]
        data=data.loc[(var >= lower_bound) | pd.isna(var)==True]
        print("data deleted => upper bound and lower bound")
        return data
    else:
        return(d)

# %%

def delete_outliers_UPPER(data_i, s, value_i):
    """
    Suppression des valeurs aberrantes supérieures à une valeur.
    
    
    Args:
        data_i ([type]): toutes les données
        s ([type]): données d'une colonne
        value ([type]): valeur maximum
    
    Exemple:
    outliers(data.circonference_cm, 700)
    
    
    """
    data_i=data_i.loc[(s <= value_i) | pd.isna(s)==True]
    return data_i
    
def delete_outliers_LOWER(data_i, s, value_i):
    """
    Suppression des valeurs aberrantes supérieures à une valeur.
    
    
    Args:
        data_i ([type]): toutes les données
        s ([type]): données d'une colonne
        value ([type]): valeur minimum
    
    Exemple:
    outliers(data.circonference_cm, 0)
    
    
    """
    data_i=data_i.loc[(s >= value_i) | pd.isna(s)==True]
    return data_i


def graph_hist(var, bins_r, title_i,color_h, xmin,xmax, xscale, ymin, ymax, xlabel, ylabel, scale_i, group_i, label_i ):
    """
    Création d'un histogramme
    
    
    Args:
        var ([type]): données de la colonne
        bins_r ([type]): plage de l'axe x
        title_i ([type]): titre
        color_h ([type]): couleur
        xmin ([type]): min de l'axe x
        xmax ([type]): max de l'axe x
        xscale ([type]): taille du graphique
        ymin ([type]): min de l'axe y
        ymax ([type]): max de l'axe y
        xlabel ([type]): nom de l'axe x
        ylabel ([type]): nom de l'axe y
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_hist(data.circonference_cm,[0,50,100,150,200,250,350,450,700], "Répartition des arbres en fonction de leur circonference en cm", (0.564, 0.823, 0.631),
          0,700, 150, 0, 70000, 'circonference_cm', 'Fréquences',(11,7))
    
    """
    plt.figure(figsize=scale_i)
    if group_i==1:
        plt.hist(var,bins=bins_r, color=color_h, label=label_i)
        plt.legend(loc='upper right')
    else:
        plt.hist(var,bins=bins_r, color=color_h)

    # Etiquetage
    plt.title(title_i)
    plt.xlim(xmin, xmax, xscale)
    plt.ylim(ymin,ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%
def graph_hist_interval_auto(var,title_i,color_h, xmin,xmax, xscale, ymin, ymax, xlabel, ylabel, scale_i, group_i, label_i ):
    """
    Création d'un histogramme
    
    
    Args:
        var ([type]): données de la colonne
        bins_r ([type]): plage de l'axe x
        title_i ([type]): titre
        color_h ([type]): couleur
        xmin ([type]): min de l'axe x
        xmax ([type]): max de l'axe x
        xscale ([type]): taille du graphique
        ymin ([type]): min de l'axe y
        ymax ([type]): max de l'axe y
        xlabel ([type]): nom de l'axe x
        ylabel ([type]): nom de l'axe y
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_hist(data.circonference_cm,[0,50,100,150,200,250,350,450,700], "Répartition des arbres en fonction de leur circonference en cm", (0.564, 0.823, 0.631),
          0,700, 150, 0, 70000, 'circonference_cm', 'Fréquences',(11,7))
    
    """
    plt.figure(figsize=scale_i)
    if group_i==1:
        plt.hist(var, color=color_h, label=label_i)
        plt.legend(loc='upper right')
    else:
        plt.hist(var,bins=bins_r, color=color_h)

    # Etiquetage
    plt.title(title_i)
    plt.xlim(xmin, xmax, xscale)
    plt.ylim(ymin,ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%


def graph_barplot(data, title, color_i,ylim_min, ylim_max, xlabel_i, ylabel_i, rotate_i, graph_vertical, fig_i):
    """
    Création d'un diagramme en barre
    
    
    
    Args:
        data ([type]): données
        title ([type]): titre
        color_i ([type]): couleur
        ylim_min ([type]): min axe y
        ylim_max ([type]): max axe y
        xlabel_i ([type]): nom de l'axe x
        ylabel_i ([type]): nom de l'axe y
        rotate_i ([type]): rotation des noms de l'axe x
        graph_vertical ([type]): diagramme en barre vertical = 1
        fig_i ([type]): taille du graphique

    Exemple:
    graph_barplot(data['reg_circonference_cm'], "Répartition des arbres en fonction des circonferences en cm (regroupés en intervalle)", 
                  (0.564, 0.823, 0.631),
              0, 40, "Intervalle - circonference_cm", "Fréquence en %",70, 1, (14,8))
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(var = t.index, freq = 100 * t.freq)
    plt.figure(figsize=fig_i)
    if graph_vertical==1:
        sns.barplot(x = "var", y = "freq", data = t, color=color_i)
        plt.ylim(ylim_min,ylim_max)
    else:
        sns.barplot(x = "freq", y = "var", data = t, color=color_i)
        plt.xlim(ylim_min,ylim_max)
    plt.title(title)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.xticks(rotation=rotate_i)
        


# %%




def graph_violinplot(data_col_i, title_i, ylabel_i, color_i,fig_i):
    """
    Création d'un violinplot :

    
    Args:
        data_col_i ([type]): données d'une colonne
        title_i ([type]): titre
        ylabel_i ([type]): nom de l'axe des y 
        color_i ([type]): couleur
        fig_i ([type]): taille du graphique
    
    Exemple: 
    graph_violinplot(data["hauteur_cm"], "Violinplot - Variable hauteur_cm", "hauteur_cm", (0.564, 0.823, 0.631), (14,7))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    ax = sns.violinplot(y=data_col_i, color=color_i)
    ax.set_title(title_i)
    ax.set_ylabel(ylabel_i)


# %%




# %%

def graph_boxplot(data_i, column_i, title_i, color_i, fig_i):
    """
    Boite à moustache
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        title_i ([type]): titre
        color_i ([type]): couleur
        fig_i ([type]): taille
    
    Exemple:
    graph_boxplot(data, "circonference_cm", "test", "g", (14,8))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    sns.boxplot(y=column_i, color=color_i, data=data_i)
    plt.title(title_i)

# %%

def graph_boxplot_by_group(data_i, column_i, groupby_i, title_i, color_i, fig_i):
    """
    Boite à moustache par catégorie
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        by_i ([type]): colonne qualitative
        title_i ([type]): titre
        color_i ([type]): couleur
        fig_i ([type]): taille
    
    Exemple:
    graph_boxplot_by(data, "circonference_cm","category", "test", "g", (14,8))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    sns.boxplot(y=column_i,x=groupby_i, palette=color_i, data=data_i)
    plt.title(title_i)

# %%

def graph_bubbleplot(data_i, color_i,xlabel_i, ylabel_i, title_i, ylim_min, ylim_max,
                    color_text, align, fontweight_i, scale_i, rotate_i):
    """
    Grphique bubbleplot
    
    
    
    Args:
        data_i ([type]): données de la colonne à représenter
        color_i ([type]): couleur des bubbles
        xlabel_i ([type]): nom de l'axe x 
        ylabel_i ([type]): nom de l'axe y
        title_i ([type]): titre
        ylim_min ([type]): min de l'axe y
        ylim_max ([type]): max de l'axe y
        color_text ([type]): couleur du texte
        align ([type]): alignement du texte
        fontweight_i ([type]): style du texte
        scale_i ([type]): Taille du graphique    
    
    Exemple:
    graph_bubbleplot(data['domanialite'], 'g',"Domanialite", "Fréquence en %", 
                 "Répartition des arbres par catégorie des domanialites", 0, 90,
                    "black", "center", "bold", (8,5), 70)
    
    """
    t = data_i
    data_var=t.value_counts(normalize=True).reset_index(name="values")
    data_var['values']=100*data_var['values']
    plt.figure(figsize=scale_i)
    sns.scatterplot(data=data_var, x='index', y='values', s=100*data_var['values'], color=color_i).margins(0.4)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.title(title_i)
    plt.ylim(ylim_min,ylim_max)
    plt.xticks(rotation=rotate_i)
    for i,txt in enumerate(data_var['values']):
        plt.text(i,txt, str(round(txt,2))+" %", color=color_text, ha=align, fontweight=fontweight_i, size=12)


# %%


def graph_circle(data, column, title):
    """
    Graphique circulaire
    
    
    
    Args:
        data ([type]): données de la colonne à representer
        column ([type]): nom de la colonne
        title ([type]): titre du graphique
    
    Exemple:
    graph_circle(data["libelle_francais"], "libelle_francais", "Répartition des arbres en fonction de leur appellation")
    
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(column = t.index, freq = 100 * t.freq)
    plt.figure(figsize=(18,8))
    plt.pie(t.freq, labels = t.column,  autopct='%.0f%%')

    plt.title(title)
    my_circle = plt.Circle((0,0), 0.4, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)


# Réalisons un diagramme circulaire représentant nos données de la variable libelle_francais.

# %%


def graph_pie(data, column, title, colors_i, fig_i):
    """
    graphique de type camembert
    
    
    
    Args:
        data ([type]): données de la colonne à représenter
        column ([type]): colonne
        title ([type]): titre
        colors_i ([type]): couleur
    
    Exemple:
    graph_pie(data["remarquable"], "remarquable", "Répartition des arbres selon s'ils sont, ou non, remarquable",['#8FD1A0', '#46A964'])
    
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(column = t.index, freq = 100 * t.freq)
    color_palette_list = colors_i
    plt.figure(figsize=fig_i)
    plt.pie(t.freq, labels = t.column,  autopct='%.2f%%', colors=color_palette_list)
    plt.title(title)
    


# %%
def graph_barplot_by_group(data, column, group, color_i, title_i): 
    """
    Tracer  un diagramme en barred'une variable en fonction d'une autre
    
    Args:
        data ([type]): données 
        column ([type]): colonne à analyser 
        group ([type]): colonne groupe 
        color_i ([type]): couleur
        title_i ([type]): titre du graphique
    
    Exemple:
     graph_barplot_by_group(data, 'reg_additives', 'group', '#6D8260', 'Répartition des aliments en fonction des additifs contenus et du nutrition grade')
     
    """
    grouped = data.groupby([group], sort=False)
    reg_carbohydrates_counts = grouped[column].value_counts(normalize=True, sort=False)

    occupation_data = [
        {'reg': column, 'group': group, 'percentage': percentage*100} for 
        (group, column), percentage in dict(reg_carbohydrates_counts).items()
    ]

    df_occupation = pd.DataFrame(occupation_data)

    p = sns.barplot(x="reg", y="percentage", hue="group", data=df_occupation)
    
    p.set(xlabel='Intervalle - '+str(column), ylabel='Fréquences - %', title=title_i)
    _ = plt.setp(p.get_xticklabels(), rotation=90)  # Rotate labels

# %%





# Dans notre jeu de données, nous savons dans quel arrondissement se trouve les arbres. Nous pouvons par exemple illustrer le nombre d'arbres que nous avons par arrondissement.
# 
# Pour cela, nous pouvons récupérer les données geographiques des arrondissements qui se trouve sur le site de Paris.
# source : https://opendata.paris.fr/explore/dataset/arrondissements/export/?disjunctive.c_ar&disjunctive.c_arinsee&disjunctive.l_ar&location=13,48.85156,2.32327

# %%

def carte_paris_arr(data_i, columns_i,  title_i, fill_color_i):
    """
    Carte de Paris : Visualiser une colonne en fonction des arrondissements
    
    
    
    Args:
        data_i ([type]): [description]
        columns_i ([type]): [description]
        title_i ([type]): [description]
        fill_color_i ([type]): [description]
    
    Exemple:
    carte_paris_arr(data, ["arrondissement_num", "hauteur_cm"], 'Représentation graphique de la hauteur (en cm) des arbres par arrondissement de Paris', "YlOrRd")
    
    """
    geo = json.load(open("arrondissements.geojson"))

    paris = folium.Map(location = [48.856578, 2.351828], zoom_start = 12)
    loc = title_i
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format(loc) 
    folium.Choropleth(geo_data = geo, key_on = "feature.properties.c_ar",
                    data = data, columns = columns_i,
                    fill_color = fill_color_i).add_to(paris)
    paris.get_root().html.add_child(folium.Element(title_html))
    paris


# Nous remarquons que les arbres les plus haut se trouve au nord-ouest de Paris.

# Il serait intéressant de savoir aussi où se trouve les arbres les plus "gros".

# %%

# %%


def plot_quanti(data_i, column_x, column_y, scale_i, color_i, title_i, ylabel_i, xlabel_i): 
    """
    [summary]
    
    
    
    Args:
        data_i ([type]): données
        column_x ([type]): nom de la colonne sur l'axe x
        column_y ([type]): nom de la colonne sur l'axe y
        scale_i ([type]): taille du graphique
        color_i ([type]): couleur
        title_i ([type]): titre
        ylabel_i ([type]): nom de l'axe y
        xlabel_i ([type]): nom de l'axe x
    
    Exemple:
    
    plot_quanti(data, "circonference_cm", "hauteur_cm", (8,5),
                "g", "Nuage de point entre la circonference (cm) et la hauteur (cm) des arbres",
            "hauteur_cm", "circonference_cm")
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=scale_i)
    ax=sns.relplot(x=column_x, y=column_y, data=data_i, color=color_i, marker="+", linewidth=1);
    plt.title(title_i)
    plt.ylabel(ylabel_i)
    plt.xlabel(xlabel_i)


# %% [markdown]
# # graphs interactifs

# %%
def graph_intv_bar(data_i, x_i, y_i, x_label_i, y_label_i, palette_color_i, title_i):
    """
    Tracer un diagramme en barre intéractif    
    
    Args:
        data_i ([type]): données
        x_i ([type]): colonne de l'axe x
        y_i ([type]): colonne de l'axe y
        x_label_i ([type]): label de l'axe x
        y_label_i ([type]): label de l'axe y
        palette_color_i ([type]): palette de couleur
        title_i ([type]): Titre du graphique
    
    Exemple:
    graph_int_bar(ms, 'column_name', 'percent_missing', "Variables", "% - Valeurs manquantes", "fall", "Variables avec plus de 50% de valeurs manquantes")
    
    """
    fig = px.bar(data_i, x=x_i, y=y_i, color=y_i, 
             labels={x_i:x_label_i,y_i:y_label_i},
              color_continuous_scale=palette_color_i)
    fig.update_layout(
        title_text=title_i, # title of plot
        plot_bgcolor= 'rgba(0, 0, 0, 0)'
    )
    #fig.write_html(name_html)
    #file=name_html+".html"
    #plot(fig, filename = file, auto_open=False)
    fig.show()



# %%
def boxplot_intv(data_var_i,y_label_i, palette_color_i , sd_i):
    """
    Tracer une boite à moustache interactive   
    
    Args:
        data_var_i ([type]): données de la variable à analyser
        y_label_i ([type]): label de l'axe y
        palette_color_i ([type]): palette de couleur
        title_i ([type]): Titre du graphique
        sd_i ([type]): ajout moyenne + Ecart-type
    Exemple:
    boxplot_int(data.proteins_100g, "Proteins_100g", "indianred",0 )
    """
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(layout=layout)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    if sd_i==1:
        fig.add_trace(go.Box(y=data_var_i, name=" ",
                        marker_color = palette_color_i, boxmean='sd'))
    else:
        fig.add_trace(go.Box(y=data_var_i, name=" ",
                        marker_color = palette_color_i))
    
    fig.update_layout(
        title_text="Boite à moustache de la variable "+y_label_i, # title of plot
        yaxis_title=y_label_i
    )
    
    fig.show()


# %%
def graph_intv_violin(data_var_i, fillcolor_i, x_label_i):
    """
    Tracer un violinplot interactif   
    
    Args:
        data_var_i ([type]): données de la variable à analyser
        fillcolor_i ([type]): palette de couleur
        x_label_i ([type]): label de l'axe x
    Exemple:
    graph_int_violin(data['saturated_fat_100g'], "#6D8260", "saturated_fat_100g")
    """
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(data=go.Violin(y=data_var_i, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor=fillcolor_i, opacity=0.6,
                               x0=" "), layout=layout)


    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    
    fig.update_layout(title_text="Violinplot de la variable "+x_label_i, 
                      yaxis_zeroline=False,
                          yaxis_title=x_label_i)
    fig.show()


# %%
def hist_intv(data_var_i, color_i, title_i, x_label_i, y_label_i):
    """
    Tracer  un histogramme intéractif
    
    Args:
        data_var_i ([type]): données de la variable à analyser
        color_i ([type]): couleur
        title_i ([type]): titre du graphique
        x_label_i ([type]): label de l'axe x
        y_label_i ([type]): label de l'axe y
    
    Exemple:
    hist_int(data['col'], "#AB2300", "Distribution des aliments en fonction de la variable ", "col", "Fréquence")
    
    """
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Histogram(
        x=data_var_i,
        name='control', # name used in legend and hover labels
        xbins=dict( # bins used for histogram
            start=0,
            end=24,
            size=4
        ),
        marker_color=color_i,
        opacity=0.75
    ))

    fig.update_layout(
        title_text=title_i, # title of plot
        xaxis_title_text=x_label_i, # xaxis label
        yaxis_title_text=y_label_i, # yaxis label
        bargap=0, # gap between bars of adjacent location coordinates
        bargroupgap=0 # gap between bars of the same location coordinates
    )
    

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)
    
    fig.show()


# %%
def graph_barplot_by_group(data, column, group, color_i, title_i): 
    """
    Tracer  un diagramme en barred'une variable en fonction d'une autre
    
    Args:
        data ([type]): données 
        column ([type]): colonne à analyser 
        group ([type]): colonne groupe 
        color_i ([type]): couleur
        title_i ([type]): titre du graphique
    
    Exemple:
     graph_barplot_by_group(data, 'reg_additives', 'group', '#6D8260', 'Répartition des aliments en fonction des additifs contenus et du nutrition grade')
     
    """
    grouped = data.groupby([group], sort=False)
    reg_carbohydrates_counts = grouped[column].value_counts(normalize=True, sort=False)

    occupation_data = [
        {'reg': column, 'group': group, 'percentage': percentage*100} for 
        (group, column), percentage in dict(reg_carbohydrates_counts).items()
    ]

    df_occupation = pd.DataFrame(occupation_data)

    p = sns.barplot(x="reg", y="percentage", hue="group", data=df_occupation)
    
    p.set(xlabel='Intervalle - '+str(column), ylabel='Fréquences - %', title=title_i)
    _ = plt.setp(p.get_xticklabels(), rotation=90)  # Rotate labels


# %%
def reg_fisher_jenks(data, colonne, new_colonne, nb_bin):
    """
    Créons une variable regroupement grâce à l'algorithme de Fisher-Jenkins
    
    ex:reg_fisher_jenks(data, "colonne1", "new_colonne1", "10")
    """
    data_add=data.loc[pd.isna(data[colonne])==False]
    breaks = jenkspy.jenks_breaks(data_add[colonne], nb_class=nb_bin)
    #label = [1,2,3,4,5]
    #breaks=list(set(breaks))
    data[new_colonne] = pd.cut(data[colonne] , bins=breaks,  include_lowest=True).to_numpy()



# %%
def del_quali_norelevant(data,identifiant,  delete_i, del_seuilcat_i, del_seuilfreq_i, all_i):
    """
    Créons une fonction qui supprime les variables qui ont un seuil de catégorie important et qui sont très dispersées.
    
    ex:del_quali_norelevant(data,["code", "created_datetime", "product_name", "countries", "countries_tags",
                          "ingredients_text"], 1, 500, 0.05, 0)
    """
    info=""
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str and col not in identifiant:
                data_autres=data.loc[data[col]!="AUTRES"]
                if delete_i==1 and max(data_autres[col].value_counts(normalize=True).head(10))<del_seuilfreq_i and (data[col].nunique()>del_seuilcat_i):
                    print("deleted "+col)
                    del data[col]
                print("Variable :"+ col + "----Max freq :"+str(max(data_autres[col].value_counts(normalize=True))))      

    

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
def amultiacp_visualizer(data_acp, Z, dim, data_study,data_stickers, data_group, column_group):
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

    display(data_acp.head())

    # Show the points in terms of the first two PCs
    g = sns.lmplot(x='PC1',
                   y='PC2',
                   hue=column_group,data=data_study2,
                   fit_reg=False,
                   scatter=True,
                   height=7)
    plt.suptitle("Premier plan factoriel") # titre général

    plt.show()

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
                 data_acp.columns.values[i])

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
    for k in range(p):
        corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
    # on modifie pour avoir un dataframe
    cos2var = corvar**2
    print(pd.DataFrame({'id':data_acp.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))

    #contributions
    ctrvar = cos2var
    for k in range(p):
        ctrvar[:,k] = ctrvar[:,k]/eigval[k]
    #on n'affiche que pour les trois premiers axes
    print(pd.DataFrame({'id':data_acp.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))

    # Load the concrete dataset
    plt.figure()
    visualizer = pca_yellow(scale=True, proj_features=True)
    visualizer.fit_transform(data_acp)
    visualizer.poof()



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
    dendrogram(Z,labels=test2.index,color_threshold=0)
    plt.show()
    return Z


# %%
#amulti_cah_create_classe(Z, test.index, 350, 12, "nutrition_grade_fr" )
def amulti_cah_create_classe(Z, data_index, threshold, n_last_clusters, group ):
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
    WGI_pca_k2 = test.assign(classe = info_groupe.index)
    t = pd.crosstab(WGI_pca_k2[group], WGI_pca_k2.classe, normalize = "columns")
    t = t.assign(col_group = t.index)
    tm = pd.melt(t, id_vars = "col_group")
    tm = tm.assign(value = 100 * tm.value)

    sns.catplot("col_group", y = "value", col = "classe", data = tm, kind = "bar")


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
        model, k=(4,12), metric='calinski_harabasz', timings=False, locate_elbow=False
    )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    
    # Generate synthetic dataset with 8 random clusters
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4,11))

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
def amulti_mix_kmeans_cah_create_classe(data, data_study, Z, kmeans2, threshold, group):
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
    #découpage à la hauteur t = 11 ==> identifiants de 5 groupes obtenus
    groupes_cah = fcluster(Z,t=threshold,criterion='distance')
    print(groupes_cah)
    centroid = kmeans2.cluster_centers_
    #index triés des groupes
    idg = np.argsort(groupes_cah)
    #affichage des observatbbions et leurs groupes
    info_groupe=pd.DataFrame(centroid[idg],groupes_cah[idg], columns=['0','1','2','3','4','5'])
    
    #classe kmeans et index des data
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data.index
    cluster_map['cluster'] = kmeans2.labels_
    
    #kmeans class et centroid
    centroid_map = pd.DataFrame(columns=["cluster", "0", "1", "2", "3", "4", "5"])
    for i in range(0,max(cluster_map["cluster"])+1):
        centroid_map.loc[i] = i, kmeans2.cluster_centers_[i][0],kmeans2.cluster_centers_[i][1], kmeans2.cluster_centers_[i][2], kmeans2.cluster_centers_[i][3],kmeans2.cluster_centers_[i][4],kmeans2.cluster_centers_[i][5]

    #classe cah
    info_groupe["classe_cah"]=info_groupe.index
    
    #merge entre centroid et classe cah
    data_merge = pd.merge(centroid_map, info_groupe, how="left", on=["0", "1", "2","3","4","5"], indicator=True,  suffixes=('', '_del'))
    data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
    cah=data_result.copy()
    
    #merge entre cah et individus avec centroid
    data_merge = pd.merge(cluster_map, cah, how="left", on=["cluster"], indicator=True,  suffixes=('', '_del'))
    data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
    mix_kmeans_cah=data_result.copy()
    
    #assign les classes cah au data
    WGI_k2 = data_study.assign(classe = mix_kmeans_cah.classe_cah)
    WGI_k2.groupby("classe").mean()

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


# %%
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

# %%

# %%
