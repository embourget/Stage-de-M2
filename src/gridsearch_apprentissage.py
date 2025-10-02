import numpy as np
import pandas as pd
import time
from scipy.special import xlogy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomTreesEmbedding, BaggingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, LassoLars, OrthogonalMatchingPursuit 
from sklearn.linear_model import BayesianRidge, LogisticRegression, TweedieRegressor, SGDRegressor, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, RadiusNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR, OneClassSVM, LinearSVC
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from pathlib import Path

import os

from modele_apprentissage import *

def max_distribution_pred(df_test, y_pred, dim, distribution):
    """
    Calcule la valeur de la fonction duale pour chaque prédiction de mu sur un DataFrame de test.

    Pour chaque ligne du DataFrame de test, la fonction :
    1. Extrait les vecteurs et scalaires nécessaires (Sst, Srs, Qr, Qs, Qt, r, s, t)
       via `get_vecteur`.
    2. Passe la prédiction correspondante `mu_pred` à la fonction de distribution
       (fonction duale) pour obtenir la valeur duale.

    Parameters
    ----------
    df_test : pandas.DataFrame
        DataFrame contenant les données de test (statistiques, contraintes, etc.).
    y_pred : np.ndarray, shape (n_samples, dim)
        Prédictions des paramètres mu pour chaque ligne.
    dim : int
        Dimension du problème (taille des vecteurs mu et Sst).
    distribution : callable
        Fonction duale de la distribution à évaluer. Doit avoir la signature :
        distribution(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)

    Returns
    -------
    list of float
        Liste des valeurs de la fonction duale calculées pour chaque ligne du DataFrame de test.

    Notes
    -----
    - Utilise `get_vecteur` pour extraire les paramètres nécessaires à partir de chaque ligne.
    - Chaque prédiction `mu_pred` est évaluée directement avec la fonction duale de la distribution.
    - Les valeurs retournées correspondent à l’évaluation de D(mu_pred) pour chaque ligne.
    """
    max_dual_pred = []
    for rows in range (df_test.shape[0]):
        Sst, Srs, Qr, Qt, Qs, t, s, r = get_vecteur(df_test, rows, dim)
        mu_pred = y_pred[rows]
        max_dual_pred += [distribution(mu_pred, Sst, Srs, Qr, Qs, Qt, r, s, t)]
    return max_dual_pred


def get_df_test(X_test, y_pred, dim, df, distribution):
    """
    Construit un DataFrame de test enrichi avec les valeurs du maximum de la fonction duale 
    calculées à partir des prédictions optimales de mu.

    Pour chaque ligne de X_test, la fonction :
    1. Sélectionne la ligne correspondante dans le DataFrame complet `df`.
    2. Calcule la valeur de la fonction duale associée à la prédiction `mu_pred` en utilisant 
       `max_distribution_pred`.
    3. Ajoute une colonne "max_dual_pred" contenant ces valeurs.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Sous-ensemble de données utilisées comme variables explicatives pour le test.
    y_pred : np.ndarray, shape (n_samples, dim)
        Prédictions des paramètres mu pour chaque ligne de X_test.
    dim : int
        Dimension du problème (taille des vecteurs mu et Sst).
    df : pandas.DataFrame
        DataFrame complet contenant les données initiales (dont X_test est extrait).
    distribution : callable
        Fonction duale de la distribution à évaluer. Doit avoir la signature :
        distribution(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)

    Returns
    -------
    pandas.DataFrame
        Copie de `df` restreinte aux indices de `X_test`, avec une colonne supplémentaire :
        - "max_dual_pred" : valeur de la fonction duale calculée pour chaque ligne à partir 
          de la prédiction de mu.

    Raises
    ------
    AssertionError
        Si le nombre de valeurs calculées ne correspond pas au nombre de lignes de `df_test`.

    Notes
    -----
    - La fonction utilise `max_distribution_pred` pour évaluer les valeurs duales.
    - Vérifie la cohérence entre le nombre de prédictions et le nombre de lignes dans `X_test`.
    """
    test_indices = X_test.index
    df_test = df.loc[test_indices].copy()
    max_dual_pred = max_distribution_pred(df_test, y_pred, dim, distribution)
    if len(max_dual_pred) != len(df_test):
        raise AssertionError("La longueur des prédictions et du DataFrame ne correspond pas")
    df_test["max_dual_pred"] = max_dual_pred
    return df_test

def MSE_score(df_test) :
    """
    Calcule l'erreur quadratique moyenne (MSE) entre les valeurs maximales réelles
    du dual optimisé et les valeurs du dual maximum calculé à partir de l'argmax prédit.

    Parameters
    ----------
    df_test : pandas.DataFrame
        DataFrame contenant au moins deux colonnes :
        - "dual_mu_opt" : valeurs réelles du dual issues de l'optimisation
        - "max_dual_pred" : valeurs prédites du dual par le modèle

    Returns
    -------
    float
        Valeur de l'erreur quadratique moyenne (MSE).
    """
    MSE_max_dual = ((df_test["dual_mu_opt"]-df_test["max_dual_pred"]) ** 2).mean()
    #MSE_max_dual = ((df_test["D_poiss(mu_opt)"]-df_test["max_dual_pred"]) ** 2).mean()
    return MSE_max_dual

def score(X_test, y_pred, dim, df, grid_search, df_test):
    """
    Évalue les performances du modèle à la fois en classification binaire (pruning)
    et en régression (dual et paramètres mu), en s’appuyant sur les résultats
    de GridSearchCV et sur les prédictions effectuées.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Jeu de données de test (features).
    y_pred : np.ndarray
        Prédictions du modèle pour X_test (vecteurs mu_pred de taille [n_samples, dim]).
    dim : int
        Dimension des vecteurs mu.
    df : pandas.DataFrame
        DataFrame complet contenant toutes les données initiales.
    grid_search : sklearn.model_selection.GridSearchCV
        Objet GridSearchCV entraîné, utilisé pour extraire les meilleurs
        hyperparamètres et les scores de validation croisée.
    df_test : pandas.DataFrame
        DataFrame aligné sur X_test, enrichi avec :
        - "max_dual_pred" : valeurs duales prédites
        - "dual_mu_opt" : valeurs duales issues de l’optimisation
        - "pruning" : étiquette binaire réelle pour la classification

    Returns
    -------
    results : list of str
        Liste de chaînes formatées contenant :
        - Hyperparamètres optimaux (best_params_)
        - MSE entre 'dual_mu_opt' et 'max_dual_pred'
        - Matrice de confusion (TP, TN, FP, FN)
        - Scores de classification : Accuracy, Recall, Precision, F-score, Specificity
        - Scores de régression issus de GridSearchCV :
            * R² global (et R² par mu si dim == 2)
            * MSE, RMSE
        - Temps moyens et écarts-types d’entraînement et de test

    Notes
    -----
    - La classification binaire est définie par le signe de "max_dual_pred"
      (positif → pruning = 1, négatif → pruning = 0).
    - Les métriques de régression proviennent directement de GridSearchCV,
      calculées lors de la validation croisée.
    - Cette fonction retourne une liste de chaînes prêtes à être affichées
      ou loguées, et non un dictionnaire structuré.
    """
    best_param = grid_search.best_params_

    MSE_max_dual = MSE_score(df_test)

    #prédiction pruning
    df_test["pruning_pred"] = (df_test["max_dual_pred"] >= 0).astype(int)

    #score TP, TN, FP, FN
    pruning = df_test["pruning"]
    pruning_pred = df_test["pruning_pred"]

    TP = ((pruning == 1) & (pruning_pred == 1)).sum()
    TN = ((pruning == 0) & (pruning_pred == 0)).sum()
    FP = ((pruning == 0) & (pruning_pred == 1)).sum()
    FN = ((pruning == 1) & (pruning_pred == 0)).sum()

    #score Accuracy, Recall, Precision, Fscore, Specificity
    Accuracy = (TP + TN)/ (TP + TN + FN + FP)
    Recall = TP / (FN + TP)
    Precision = TP / (FP + TP)
    F_score = 2 * (Precision * Recall) / (Precision + Recall)
    Specificity = TN / (TN + FP)

    #tableau résultats gridsearch
    best_index = grid_search.best_index_
    cv_results = pd.DataFrame(grid_search.cv_results_)

    #R2 score argmax (moyenne des R2 scores pour chaque mu)
    best_mean_score_r2 = cv_results['mean_test_r2'][best_index] 
    best_std_score_r2 = cv_results['std_test_r2'][best_index]

    #score MSE
    best_mean_test_mse = cv_results['mean_test_mse'][best_index] 
    best_std_test_mse = cv_results['std_test_mse'][best_index]

    #score RMSE
    best_mean_test_rmse = -cv_results['mean_test_rmse'][best_index]
    best_std_test_rmse = cv_results['std_test_rmse'][best_index]

    #temps entrainement
    mean_fit_time = cv_results['mean_fit_time'][best_index]
    std_fit_time = cv_results['std_fit_time'][best_index]

    #temps test
    mean_score_time = cv_results['mean_score_time'][best_index]
    std_score_time = cv_results['std_score_time'][best_index]

    results = [
        f"best_param = {best_param}", 
        f"MSE_max_dual = {MSE_max_dual}", 
        f"TP = {TP}", f"TN = {TN}", f"FP = {FP}", f"FN = {FN}", 
        f"Accuracy = {Accuracy}", f"Recall = {Recall}", f"Precision = {Precision}", 
        f"F_score = {F_score}", f"Specificity = {Specificity}", 
        f"best_mean_score_r2 = {best_mean_score_r2}", 
        f"best_std_score_r2 = {best_std_score_r2}", 
        f"best_mean_test_mse = {best_mean_test_mse}", 
        f"best_std_test_mse = {best_std_test_mse}",
        f"best_mean_test_rmse = {best_mean_test_rmse}",
        f"best_std_test_rmse = {best_std_test_rmse}",
        f"mean_fit_time = {mean_fit_time}",
        f"std_fit_time = {std_fit_time}",
        f"mean_score_time = {mean_score_time}",
        f"std_score_time = {std_score_time}"
        ]
    
    if dim == 2 :
        #R2 score pour mu_opt_1
        best_mean_score_r2_mu_opt1 = cv_results['mean_test_r2_mu_opt1'][best_index]
        best_std_score_r2_mu_opt1 = cv_results['std_test_r2_mu_opt1'][best_index]

        #R2 score pour mu_opt_2
        best_mean_score_r2_mu_opt2 = cv_results['mean_test_r2_mu_opt2'][best_index]
        best_std_score_r2_mu_opt2 = cv_results['std_test_r2_mu_opt2'][best_index]

        results += [
            f"best_mean_score_r2_mu_opt1 = {best_mean_score_r2_mu_opt1}", 
            f"best_std_score_r2_mu_opt1 = {best_std_score_r2_mu_opt1}", 
            f"best_mean_score_r2_mu_opt2 = {best_mean_score_r2_mu_opt2}", 
            f"best_std_score_r2_mu_opt2 = {best_std_score_r2_mu_opt2}", 
        ]

    return results

def r2_mu_opt1(y_true, y_pred):
    """
    Calcule le R² uniquement pour la première sortie (mu_opt1).

    Parameters
    ----------
    y_true : array-like or pandas.DataFrame
        Valeurs réelles des sorties. Doit contenir au moins 2 colonnes.
    y_pred : array-like or pandas.DataFrame
        Valeurs prédites correspondantes.

    Returns
    -------
    float
        Coefficient de détermination R² pour la première sortie.
    """
    if hasattr(y_true, "iloc"):
        y_true_col = y_true.iloc[:, 0]
    else:
        y_true_col = y_true[:, 0]
    y_pred_col = y_pred[:, 0]
    return r2_score(y_true_col, y_pred_col)

def r2_mu_opt2(y_true, y_pred):
    """
    Calcule le R² uniquement pour la deuxième sortie (mu_opt2).

    Parameters
    ----------
    y_true : array-like or pandas.DataFrame
        Valeurs réelles des sorties. Doit contenir au moins 2 colonnes.
    y_pred : array-like or pandas.DataFrame
        Valeurs prédites correspondantes.

    Returns
    -------
    float
        Coefficient de détermination R² pour la deuxième sortie.
    """
    if hasattr(y_true, "iloc"):
        y_true_col = y_true.iloc[:, 1]
    else:
        y_true_col = y_true[:, 1]
    y_pred_col = y_pred[:, 1]
    return r2_score(y_true_col, y_pred_col)

# def gridsearch(model, model_name, param_grid, df, dim, distribution, X, y) :
#     """
#     Effectue une recherche d'hyperparamètres par validation croisée (GridSearchCV) 
#     avec plusieurs métriques de scoring, en intégrant un prétraitement conditionnel
#     (scaling / PCA) selon le modèle choisi.

#     Parameters
#     ----------
#     model : estimator object
#         Modèle compatible scikit-learn (par ex. DecisionTreeRegressor, RandomForest, etc.).
#     model_name : str
#         Nom du modèle (permet de décider si un prétraitement est appliqué).
#     param_grid : dict
#         Dictionnaire des hyperparamètres à tester pour GridSearchCV.
#     df : pandas.DataFrame
#         DataFrame initial contenant les données complètes (non utilisé directement ici).
#     dim : int
#         Dimension du problème (par ex. 2 si sortie = (mu_opt1, mu_opt2)).
#     distribution : function
#         Fonction associée à la distribution considérée (non utilisée directement ici).
#     X : pandas.DataFrame or ndarray
#         Variables explicatives.
#     y : pandas.DataFrame or ndarray
#         Variables cibles (doit contenir au moins 2 colonnes : mu_opt1 et mu_opt2).

#     Returns
#     -------
#     X_test : pandas.DataFrame or ndarray
#         Jeu de test retenu pour l'évaluation.
#     y_pred : ndarray
#         Prédictions du meilleur modèle sur X_test.
#     grid_search : GridSearchCV
#         Objet GridSearchCV entraîné, contenant tous les résultats de la recherche.
#     """
#     scoring1 = {
#         'r2' : 'r2',
#         'mse' : 'neg_mean_squared_error',
#         'rmse': 'neg_root_mean_squared_error',
#     }
    
#     if dim == 2 :
#         scoring2 = {
#         'r2_mu_opt1': make_scorer(r2_mu_opt1),
#         'r2_mu_opt2': make_scorer(r2_mu_opt2),
#         }
#     else :
#         scoring2 = {}

#     preprocess_models = [
#         'SVR', 'NuSVR', 'KNeighborsRegressor', 'MLPRegressor', 'LinearRegression', 
#         'Ridge', 'Lasso', 'MultiTaskLasso', 'ElasticNet', 'LassoLars', 'OrthogonalMatchingPursuit',
#         'BayesianRidge', 'TweedieRegressor', 'SGDRegressor', 'PassiveAggressiveRegressor', 
#         'TheilSenRegressor', 'KernelRidge'
#     ]

#     if model_name in preprocess_models :
#         preprocess_options = [
#             StandardScaler(),
#             MinMaxScaler(),
#             PCA(n_components = dim),
#             Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=dim))]),
#             Pipeline([('scaler', MinMaxScaler()), ('pca', PCA(n_components=dim))])
#         ]
    
#     else :
#         preprocess_options = ['passthrough']
    
#     full_pipeline = Pipeline([
#         ('preprocessor', 'passthrough'),
#         ('estimator', model)
#     ]) 

#     param_grid_full = {'preprocessor': preprocess_options}  

#     param_grid_final = {}

#     for k, v in param_grid.items():
#         if k.startswith('estimator__'):
#             param_grid_final[f'estimator__{k}'] = v  # <-- double préfixe
#         else:
#             param_grid_final[f'estimator__estimator__{k}'] = v

#     param_grid_full.update(param_grid_final)
    
#     scoring = scoring1 | scoring2

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     grid_search = GridSearchCV(
#         estimator=full_pipeline,
#         param_grid=param_grid_full,
#         cv=5,
#         scoring=scoring,
#         refit='mse'
#     )

#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     y_pred = best_model.predict(X_test)

#     mu_opt1_pred = y_pred[:, 0]
#     mu_opt2_pred = y_pred[:, 1]

#     return X_test, y_pred, grid_search



def gridsearch(model, model_name, param_grid, df, dim, distribution, X, y) :
    """
    Effectue une recherche d'hyperparamètres par validation croisée (GridSearchCV) 
    avec plusieurs métriques de scoring, y compris des scores personnalisés si dim == 2.

    Parameters
    ----------
    model : estimator object
        Modèle scikit-learn à entraîner (ex. DecisionTreeRegressor, RandomForest, etc.).
    param_grid : dict
        Grille d’hyperparamètres pour la recherche.
    X : pandas.DataFrame or ndarray
        Variables explicatives.
    y : pandas.DataFrame or ndarray
        Variables cibles (mu_opt).
    dim : int
        Dimension du problème (ex. 2 si y contient (mu_opt1, mu_opt2)).

    Returns
    -------
    X_test : pandas.DataFrame or ndarray
        Sous-ensemble de test retenu pour l'évaluation.
    y_pred : ndarray
        Prédictions du meilleur modèle sur X_test.
    grid_search : GridSearchCV
        Objet GridSearchCV entraîné, contenant tous les résultats.
    """
    scoring1 = {
        'r2' : 'r2',
        'mse' : 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
    }
    
    if dim == 2 :
        scoring2 = {
        'r2_mu_opt1': make_scorer(r2_mu_opt1),
        'r2_mu_opt2': make_scorer(r2_mu_opt2),
        }
    else :
        scoring2 = {}
    
    scoring = scoring1 | scoring2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit='mse'
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mu_opt1_pred = y_pred[:, 0]
    mu_opt2_pred = y_pred[:, 1]

    return X_test, y_pred, grid_search


def get_X_y_newcolumns(df, dim):
    """
    Construit les matrices de variables explicatives (X) et de variables cibles (y)
    à partir d'un DataFrame, en ajoutant de nouvelles colonnes 'quotient'.

    Pour chaque dimension i, une colonne 'quotient{i}' est calculée comme :
        quotient_i = (Qs - Qrs_i) / (s - r_i)

    Les colonnes de X comprennent :
        - Sst1, ..., Sst_dim
        - Srs1, ..., Srs_{2*dim}
        - quotient1, ..., quotient_dim

    Les colonnes de y comprennent :
        - mu_opt1, ..., mu_opt_dim

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes nécessaires (Qs, Qrs_i, s, r_i, Sst_i, Srs_i, mu_opt_i).
    dim : int
        Dimension du problème (par exemple 2 si les sorties sont mu_opt1 et mu_opt2).

    Returns
    -------
    X : pandas.DataFrame
        Variables explicatives construites à partir du DataFrame.
    y : pandas.DataFrame
        Variables cibles (mu_opt_i).
    df : pandas.DataFrame
        Le DataFrame original, enrichi des colonnes 'quotient'.
    """
    for i in range(1, dim+1) :
        df[f'quotient{i}'] = (df.Qs - df[f'Qrs{i}']) / (df.s - df[f'r{i}'])
    
    colonnes = [f"Sst{i}" for i in range(1, dim + 1)] \
        + [f"Srs{i}" for i in range(1, 2 * dim + 1)] \
        + [f"quotient{i}" for i in range(1, dim +  1)]
    X = df[colonnes]

    col = [f"mu_opt{i}" for i in range(1, dim+1)]
    y = df[col]
    #y = df[['mu_opt1', 'mu_opt2']]
    return X, y, df


def main(fpath, model_name, dual, dim) :
    """
    Retourne un estimateur scikit-learn et son dictionnaire d'hyperparamètres 
    (param_grid) en fonction du nom du modèle demandé.

    Parameters
    ----------
    model_name : str
        Nom du modèle à instancier. Doit être parmi la liste suivante :
        
        - Régressions à noyau et voisins :
            * "SVR"
            * "NuSVR"
            * "KNeighborsRegressor"
            * "KernelRidge"
        
        - Régressions linéaires et régularisées :
            * "LinearRegression"
            * "Ridge"
            * "Lasso"
            * "ElasticNet"
            * "MultiTaskLasso"
            * "MultiTaskElasticNet"
            * "LassoLars"
            * "OrthogonalMatchingPursuit"
            * "BayesianRidge"
            * "SGDRegressor"
            * "PassiveAggressiveRegressor"
            * "TheilSenRegressor"
            * "TweedieRegressor"
        
        - Arbres et ensembles :
            * "DecisionTreeRegressor"
            * "ExtraTreeRegressor"
            * "RandomForestRegressor"
            * "ExtraTreesRegressor"
            * "GradientBoostingRegressor"
            * "BaggingRegressor"
        
        - Réseaux de neurones :
            * "MLPRegressor"

    Returns
    -------
    param_grid : dict
        Dictionnaire des hyperparamètres à tester dans GridSearchCV,
        adapté au modèle choisi.
    model : estimator
        Instance du modèle scikit-learn correspondant.

    Raises
    ------
    ValueError
        Si le `model_name` fourni n'est pas reconnu parmi la liste ci-dessus.

    Notes
    -----
    - Chaque modèle est instancié avec des hyperparamètres par défaut,
      et `param_grid` propose des plages d’exploration pour GridSearchCV.
    - Les modèles sensibles à l’échelle des données (ex. SVR, KNN, MLP, modèles linéaires)
      peuvent être combinés avec des préprocesseurs (StandardScaler, MinMaxScaler, PCA).
    """
    df = get_csv(fpath)
    param_grid, model = get_model(model_name)
    distribution = get_Distribution(dual)

    X, y, df = get_X_y_newcolumns(df, dim)

    X_test, y_pred, grid_search = gridsearch(model, model_name, param_grid, df, dim, distribution, X, y)
    
    df_test = get_df_test(X_test, y_pred, dim, df, distribution)

    res1 = [
        f"model : {model_name}"
        ]
    res2 = score(X_test, y_pred, dim, df, grid_search, df_test)
    results = res1 + res2
    return results


