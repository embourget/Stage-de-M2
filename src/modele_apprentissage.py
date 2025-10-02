import numpy as np
import pandas as pd
import time
from scipy.special import xlogy
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomTreesEmbedding, BaggingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, LassoLars, OrthogonalMatchingPursuit 
from sklearn.linear_model import BayesianRidge, LogisticRegression, TweedieRegressor, SGDRegressor, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, RadiusNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR, OneClassSVM, LinearSVC
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from pathlib import Path

import os

from dual import *


def get_csv(fpath: Path):
    """
    Charge un fichier CSV dans un DataFrame pandas.

    Parameters
    ----------
    fpath : Path
        Chemin du fichier CSV.

    Returns
    -------
    pd.DataFrame
        Contenu du fichier CSV sous forme de DataFrame.

    Notes
    -----
    Le CSv doit contenir une ligne d'en-tête (utilisée comme noms des colonnes)
    """
    df = pd.read_csv(fpath, header=0)
    return df



def get_model(model) :
    """
    Retourne un estimateur de régression supervisée et une grille 
    d’hyperparamètres pour l’optimisation par recherche de grille
    adaptée à ce modèle.

    Selon la chaîne de caractères fournie, la fonction :
    - crée un estimateur de régression simple ou multi-sorties,
      en utilisant `MultiOutputRegressor` si nécessaire,
    - définit une grille `param_grid` pour l’optimisation
      d’hyperparamètres (utile pour `GridSearchCV` ou `RandomizedSearchCV`).

    Parameters
    ----------
    model : str
        Nom du modèle à instancier. Valeurs possibles :
        {"SVR", "NuSVR", "LinearRegression", "KNeighborsRegressor", 
        "MLPRegressor", "RandomForestRegressor", "GradientBoostingRegressor",
        "BaggingRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor",
        "ExtraTreeRegressor", "Ridge", "Lasso", "MultiTaskLasso", 
        "ElasticNet", "MultiTaskElasticNet", "LassoLars",
        "OrthogonalMatchingPursuit", "BayesianRidge", "TweedieRegressor",
        "SGDRegressor", "PassiveAggressiveRegressor", "TheilSenRegressor",
        "KernelRidge"}.

    Returns
    -------
    tuple
        - param_grid : dict
            Dictionnaire des hyperparamètres à tester pour le modèle.
            Pour les modèles multi-sorties, les clés utilisent le préfixe
            `estimator__` afin de fonctionner dans `MultiOutputRegressor`.
        - model : sklearn estimator
            Estimateur scikit-learn initialisé. Peut être utilisé directement
            dans un pipeline ou dans une recherche d’hyperparamètres.

    Notes
    -----
    - Les modèles ne supportant pas nativement plusieurs sorties sont encapsulés
      dans `MultiOutputRegressor`.
    - La grille `param_grid` est adaptée aux conventions scikit-learn et permet
      une intégration directe avec `GridSearchCV`.
    - Cette fonction ne teste pas la validité du nom du modèle : si le modèle
      fourni n’est pas reconnu, elle retournera une erreur.
    """

    if model == 'SVR' :
        param_grid = {
            'estimator__kernel': ['linear', 'rbf', 'sigmoid'],
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__epsilon': [0.01, 0.1, 0.5],
            'estimator__gamma': ['scale', 'auto']
        }
        model = MultiOutputRegressor(SVR())
    
    elif model == 'LinearRegression' :
        param_grid = {
            'fit_intercept' : [True, False],
            'positive': [True, False]
        }
        model = LinearRegression()
    
    elif model == 'NuSVR' :
        param_grid = {
            'estimator__kernel': ['linear', 'rbf', 'sigmoid'],
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__nu': [0.1, 0.3, 0.5, 0.7],
            'estimator__degree': [2, 3, 4],        
            'estimator__coef0': [0.0, 0.1, 0.5]
        }
        model = MultiOutputRegressor(NuSVR())

    elif model == 'KNeighborsRegressor' :
        param_grid = {
            'n_neighbors' : [3, 5, 7, 9, 11, 15, 21, 31],
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40],
            'p': [1, 2]
        }
        model = KNeighborsRegressor()

    elif model == 'MLPRegressor' :
        param_grid = {
            'estimator__hidden_layer_sizes': [
                (128,), (256,),         
                (128, 64), (256, 128), 
                (128, 64, 32), (256, 128, 64),
                (256, 256, 128, 64)
            ],
            'estimator__activation': ['relu', 'tanh'],
            'estimator__alpha': [0.0001, 0.001],
            'estimator__learning_rate': ['constant', 'adaptive', 'invscaling'],
            'estimator__learning_rate_init': [0.001, 0.01],
            'estimator__max_iter': [200, 300, 500]
        }
        model = MultiOutputRegressor(MLPRegressor())

    elif model == 'RandomForestRegressor' :
        param_grid = {
            'n_estimators': [100, 300, 500, 700],  
            'max_depth': [None, 5, 10], 
            'bootstrap': [True, False]                             
        }
        model = RandomForestRegressor()

    elif model == 'GradientBoostingRegressor':
        param_grid = {
            'estimator__n_estimators': [100, 300, 500, 700],
            'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'estimator__max_depth': [None, 3, 5, 10], 
            'estimator__subsample': [0.5, 0.7, 1.0]                        
        }
        model = MultiOutputRegressor(GradientBoostingRegressor())
    
    elif model == 'BaggingRegressor' :
        param_grid = {
            'estimator__n_estimators': [100, 300, 500, 700],  
            'estimator__max_samples': [0.5, 0.7, 1.0],            
            'estimator__max_features': [0.5, 0.7, 1.0],    
            'estimator__bootstrap': [True, False],
            'estimator__bootstrap_features': [True, False]                              
        }
        model = BaggingRegressor()

    # elif model == 'ExtraTreesRegressor' :
    #     param_grid = {
    #     'n_estimators' : [100, 300, 500, 700],
    #     'max_depth' : [5, 10, None],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'bootstrap': [False, True] 
    #     }
    #     model = ExtraTreesRegressor()
    
    elif model == 'ExtraTreesRegressor' :
        param_grid = {
        'n_estimators' : [700],
        'max_depth' : [5, 15],
        'max_features' : ['sqrt'],
        'bootstrap': [False] 
        }
        model = ExtraTreesRegressor()


    # elif model == 'DecisionTreeRegressor' :
    #     param_grid = {
    #         'estimator__splitter' : ['best', 'random'],           
    #         'estimator__max_depth' : [5, 10, None], 
    #         'estimator__max_features': ['sqrt', 'log2', None],
    #         'estimator__max_leaf_nodes': [None, 10, 20, 50],
    #         'estimator__ccp_alpha': [0.0, 0.01, 0.1]                            
    #     }
    #     model = MultiOutputRegressor(DecisionTreeRegressor())
    
    elif model == 'DecisionTreeRegressor' :
        param_grid = {
            'estimator__splitter' : ['best'],           
            'estimator__max_depth' : [10], 
            'estimator__max_features': ['sqrt'],
            'estimator__max_leaf_nodes': [20],
            'estimator__ccp_alpha': [0.0]                            
        }
        model = MultiOutputRegressor(DecisionTreeRegressor())

    elif model == 'ExtraTreeRegressor' :
        param_grid = {
            'splitter' : ['best', 'random'],
            'max_depth' : [5, 10, None],
            'max_features': ['sqrt', 'log2', 0.5, 1.0, None],
            'max_leaf_nodes': [None, 10, 50, 100], 
            'ccp_alpha': [0.0, 0.01, 0.1]
        }
        model = ExtraTreeRegressor()
    
    elif model == 'Ridge' :
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'fit_intercept' : [True, False],
        }
        model = Ridge()
    
    elif model == 'Lasso' :
        param_grid = {
            'estimator__alpha' : [0.01, 0.1, 1.0, 10.0, 100.0],
            'estimator__fit_intercept': [True, False],
            'estimator__selection' : ['cyclic', 'random']
        }
        model = MultiOutputRegressor(Lasso())
    
    elif model == 'MultiTaskLasso' :
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'selection' : ['cyclic', 'random']
        }
        model = MultiTaskLasso()
    
    elif model == 'ElasticNet' :
        param_grid = {
            'estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'estimator__l1_ratio' : [0.0, 0.2, 0.5, 0.7, 0.9, 1.0],
            'estimator__fit_intercept': [True, False],
            # 'tol' ?
            'estimator__selection' : ['cyclic', 'random']
        }
        model = MultiOutputRegressor(ElasticNet())
    
    elif model == 'MultiTaskElasticNet' :
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0,  100.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
            'fit_intercept': [True, False],
            'tol': [1e-3, 1e-4],
            'max_iter': [1000, 5000]
        }
        model = MultiTaskElasticNet()
    
    elif model == 'LassoLars' :
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'fit_intercept': [True, False],
            'precompute': [True, False],
            'max_iter': [500, 1000, 5000],
            'eps': [1e-3, 1e-4]
        }
        model = LassoLars()
    
    elif model == 'OrthogonalMatchingPursuit' :
        param_grid = {
            'n_nonzero_coefs': [2, 4, 6, 8, 10],
            'tol': [1e-6, 1e-4, 1e-2],
            'fit_intercept': [True, False]
        } 
        model = OrthogonalMatchingPursuit()
    
    elif model == 'BayesianRidge' :
        param_grid = {
            'estimator__alpha_1': [1e-6, 1e-4, 1e-2, 1.0],
            'estimator__alpha_2': [1e-6, 1e-4, 1e-2, 1.0],
            'estimator__lambda_1': [1e-6, 1e-4, 1e-2, 1.0],
            'estimator__lambda_2': [1e-6, 1e-4, 1e-2, 1.0],
            'estimator__fit_intercept': [True, False]

        }
        model = MultiOutputRegressor(BayesianRidge())
    
    elif model == 'TweedieRegressor' :
        param_grid = {
            'estimator__alpha' : [0.01, 0.1, 1.0],
            'estimator__fit_intercept' : [True, False],
            'estimator__link' : ['auto', 'identity', 'log'],
            'estimator__solver' : ['lbfgs', 'newton-cholesky'],
            'estimator__max_iter': [100, 1000, 10000],
            'estimator__tol': [1e-3, 1e-4, 1e-5],
            'estimator__warm_start': [False, True]
        }
        model = MultiOutputRegressor(TweedieRegressor())
    
    elif model == 'SGDRegressor' : 
        param_grid = {
            'estimator__loss': ['squared_error', 'huber', 'epsilon_insensitive'],
            'estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'estimator__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'estimator__l1_ratio': [0.15, 0.5, 0.9],  
            'estimator__max_iter': [1000, 2000, 5000],
            'estimator__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'estimator__eta0': [1e-4, 1e-3, 1e-2], 
            'estimator__early_stopping': [True, False],
            'estimator__tol': [1e-4, 1e-3, 1e-2],
            'estimator__fit_intercept': [True, False]
        }
        model = MultiOutputRegressor(SGDRegressor())
    
    elif model == 'PassiveAggressiveRegressor' :
        param_grid = {
            'estimator__C': [0.01, 0.1, 1.0, 10.0],          
            'estimator__fit_intercept': [True, False],
            'estimator__max_iter': [500, 1000, 5000],
            'estimator__tol': [1e-4, 1e-3, 1e-2],
            'estimator__early_stopping': [True, False],
            'estimator__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'estimator__epsilon': [0.01, 0.1, 1.0] 
        }
        model = MultiOutputRegressor(PassiveAggressiveRegressor())
    
    elif model == 'TheilSenRegressor' :
        param_grid = {
            'estimator__fit_intercept': [True, False],
            'estimator__max_subpopulation': [1e4, 1e5],  
            'estimator__n_subsamples': [None, 100, 200],
            'estimator__max_iter': [100, 300, 1000],
            'estimator__tol': [1e-3, 1e-2]
        }
        model = MultiOutputRegressor(TheilSenRegressor())
    
    elif model == 'KernelRidge' :
        param_grid = {
            'alpha': [0.01, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'], 
            'gamma': [0.1, 1, 10],
            'degree': [2, 3, 4], 
            'coef0': [1, 3, 5] 
        }
        model = KernelRidge()
    
    return param_grid, model


def get_Distribution(distribution: str):
    """
    Récupère la fonction duale correspondant à une distribution donnée pour la partie 
    Machine Learning.

    Parameters
    ----------
    distribution : str
        Nom de la distribution. Par exemple :
        {"poisson", "gauss", "exp", "geom", "bern", "negbin"}.

    Returns
    -------
    function
        La fonction duale associée à la distribution.

    Raises
    ------
    ValueError
        Si la distribution n'est pas reconnue.
    """

    fonctions = {
        "poisson": D_poisson,
        "gauss": D_gauss,
        "exp": D_exp,
        "geom": D_geom,
        "bern": D_bern,
        "negbin": D_negbin,
    }

    if distribution not in fonctions:
        raise ValueError(
            f"Distribution '{distribution}' inconnue. Choix parmi : {list(fonctions.keys())}"
        )

    return fonctions[distribution]


def get_vecteur(df, row : int, dim : int)-> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, np.ndarray] :
    """
    Extrait les vecteurs et scalaires nécessaires pour formuler le problème dual
    à partir d'une ligne spécifique d'un DataFrame.

    Cette fonction lit les valeurs d'une ligne du DataFrame et construit :
    - les vecteurs et matrices des coefficients S_st, S_rs, Q_r et r,
    - les scalaires Qt, Qs, t et s, nécessaires pour la formulation duale
      d'un problème d'optimisation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant toutes les données du problème.
    row : int
        Index de la ligne à extraire.
    dim : int
        Dimension du problème (nombre de variables).

    Returns
    -------
    tuple
        Tuple contenant dans l'ordre :
        - Sst : np.ndarray, shape (dim,)
            Vecteur des coefficients S_st.
        - Srs : np.ndarray, shape (dim, dim)
            Matrice des coefficients S_rs.
        - Qr : np.ndarray, shape (dim,)
            Vecteur des coefficients Q_r.
        - Qt : int
            Scalaire Q_t.
        - Qs : int
            Scalaire Q_s.
        - t : int
            Scalaire associé au temps/distribution t.
        - s : int
            Scalaire associé au temps/distribution s.
        - r : np.ndarray, shape (dim,)
            Vecteur des coefficients r.
    """
    Sst = df.iloc[row, 0:dim].values

    Srs = np.empty((dim, dim))

    for col in range(dim):
        start = dim + col * dim
        end = start + dim
        Srs[:, col] = df.iloc[row, start:end].values

    Qr = df.iloc[row, (dim + 1) * dim + 2 : (dim + 2) * dim + 2].values
    Qt = int(df.iloc[row, (dim + 1) * dim])
    Qs = int(df.iloc[row, (dim + 1) * dim + 1])
    t = int(df.iloc[row, dim**2 + 2 * dim + 2])
    s = int(df.iloc[row, dim**2 + 2 * dim + 3])
    r = df.iloc[row, dim**2 + 2 * dim + 4 : dim**2 + 3 * dim + 4].values
    vect = (Sst, Srs, Qr, Qt, Qs, t, s, r)
    return vect
