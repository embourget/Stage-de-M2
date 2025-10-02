from pathlib import Path

import pandas as pd
from scipy.optimize import (Bounds, LinearConstraint, NonlinearConstraint,
                            minimize)
from scipy.optimize import linprog

from constraints import *
from dual import *
from decision import *
from decision_constraints import *

def get_D(distribution: str):
    """
    Récupère la fonction duale et son opposée et les contraintes associées
    pour une distribution donnée.

    Parameters
    ----------
    distribution : str
        Nom de la distribution. Doit être parmi :
        {"poisson", "gauss", "exp", "geom", "bern", "negbin"}.

    Returns
    -------
    tuple
        - fonctions : tuple
            (fonction D, fonction négative de D).
        - contraintes : tuple
            Ensemble de fonctions générant les matrices (A, c)
            représentant les contraintes linéaires associées à la distribution.

    Raises
    ------
    ValueError
        Si la distribution n'est pas reconnue.
    """

    fonctions = {
        "poisson": (D_poisson, neg_D_poisson),
        "gauss": (D_gauss, neg_D_gauss),
        "exp": (D_exp, neg_D_exp),
        "geom": (D_geom, neg_D_geom),
        "bern": (D_bern, neg_D_bern),
        "negbin": (D_negbin, neg_D_negbin),
        "poisson_decision": (Decision_poisson, neg_Decision_poisson),
        "gauss_decision": (Decision_gauss, neg_Decision_gauss),
        "exp_decision": (Decision_exp, neg_Decision_exp),
        "geom_decision": (Decision_geom, neg_Decision_geom),
        "bern_decision": (Decision_bern, neg_Decision_bern),
        "negbin_decision": (Decision_negbin, neg_Decision_negbin),
    }

    if distribution not in fonctions:
        raise ValueError(
            f"Distribution '{distribution}' inconnue. Choix parmi : {list(fonctions.keys())}"
        )

    contraintes = {
        "poisson": (
            get_A1_and_c1,
            get_A5_and_c5,
            get_A6_and_c6,
        ),
        "gauss": (
            get_A5_and_c5, 
            get_A6_and_c6
        ),
        "exp": (
            get_A1_and_c1,
            get_A5_and_c5,
            get_A6_and_c6,
        ),
        "geom": (
            get_A1_and_c1,
            get_A2_and_c2,
            get_A5_and_c5,
            get_A6_and_c6,
        ),
        "bern": (
            get_A1_and_c1,
            get_A4_and_c4,
            get_A5_and_c5,
            get_A6_and_c6,
        ),
        "negbin": (
            get_A1_and_c1,
            get_A3_and_c3,
            get_A5_and_c5,
            get_A6_and_c6,
        ),
        "poisson_decision": (
            decision_get_A1_and_c1,
            decision_get_A3_and_c3,
        ),
        "gauss_decision": (
            decision_get_A3_and_c3,
        ),
        "exp_decision": (
            decision_get_A1_and_c1,
            decision_get_A3_and_c3,
        ),
        "geom_decision": (
            decision_get_A1_and_c1,
            decision_get_A2_and_c2,
            decision_get_A3_and_c3,
        ),
        "bern_decision":(
            decision_get_A1_and_c1,
            decision_get_A3_and_c3,
            decision_get_A4_and_c4,
        ),
        "negbin_decision": (
            decision_get_A1_and_c1,
            decision_get_A3_and_c3,
            decision_get_A5_and_c5,
        ),
    }

    return fonctions[distribution], contraintes[distribution]


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

def find_feasible_solution(A: np.ndarray, c: np.ndarray):
    # A, shape (n_constraints, n_dims)
    # c, shape (n_constraints,)
    # mu (output), shape (n_dims,)
    
    # find a mu such that A @ mu >= c
    
    # Define a dummy objective: minimize 0*x
    objective = np.zeros(A.shape[1])
    res = linprog(c=objective, A_ub=-A, b_ub=-c, method='highs')

    if res.success:
        return res.x
    else:
        return None

def ajout_colonnes(df, dim : int)-> tuple[list]:
    """
    Ajoute au DataFrame les colonnes nécessaires à l’optimisation duale.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame initial contenant les données du problème.
    dim : int
        Dimension du problème, c’est-à-dire le nombre de colonnes `mu_opt`
        à ajouter.

    Returns
    -------
    tuple
        - columns : list of str
            Noms des colonnes `mu_opt1, mu_opt2, ..., mu_optdim`.
        - df : pandas.DataFrame
            Nouveau DataFrame enrichi avec les colonnes suivantes :
            
            * `mu_opt1 ... mu_optdim` : colonnes initialisées à 0 pour stocker les solutions optimales,
            * `dual_mu_opt` : valeur duale initialisée à 0,
            * `pruning` : colonne de contrôle (0 = conservé, 1 = élagué),
            * `distance_euclidienne` : colonne initialisée à 0 pour stocker une distance ultérieure.
    """
    columns = []
    for col in range(dim):
        col_name = f"mu_opt{col + 1}"
        df = df.assign(**{col_name: np.zeros(df.shape[0])})
        columns.append(col_name)

    col_name = "dual_mu_opt"
    df = df.assign(**{col_name: np.zeros(df.shape[0])})

    col_name = "pruning"
    df = df.assign(**{col_name: np.zeros(df.shape[0])})

    col_name = "distance_euclidienne"
    df = df.assign(**{col_name: np.zeros(df.shape[0])})
    return columns, df
    

def vecteur(df, row : int, dim : int)-> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, np.ndarray] :
    """
    Extrait du DataFrame les vecteurs et scalaires nécessaires
    pour formuler le problème dual à partir d'une ligne donnée.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données du problème.
    row : int
        Index de la ligne à partir de laquelle extraire les valeurs.
    dim : int
        Dimension du problème (nombre de variables).

    Returns
    -------
    tuple
    Contient les éléments extraits du DataFrame, dans l'ordre :
        - Sst : np.ndarray, shape (dim,)
            Vecteur des coefficients `S_st`.
        - Srs : np.ndarray, shape (dim, dim)
            Matrice des coefficients `S_rs`.
        - Qr : np.ndarray, shape (dim,)
            Vecteur des coefficients `Q_r`.
        - Qt : int
            Scalaire `Q_t`.
        - Qs : int
            Scalaire `Q_s`.
        - t : int
            Scalaire associé au temps ou à la distribution `t`.
        - s : int
            Scalaire associé au temps ou à la distribution `s`.
        - r : np.ndarray, shape (dim,)
            Vecteur des coefficients `r`.
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

def test_feasible_solution(mu, A, c) :
    if np.all(A @ mu >= c) :
        return True
    else :
        return False

def optimisation1(r, s, t, Qr, Qs, Qt, Srs, Sst, df, neg_D_funct, 
                  contraintes_fonctions : tuple, rows : int, columns : tuple, methode, distribution):
    """
    Effectue l'optimisation du problème dual pour une ligne du DataFrame.

    Cette fonction construit les contraintes linéaires associées à une distribution
    donnée, initialise un point de départ faisable `mu_0`, puis lance une
    optimisation numérique via `scipy.optimize.minimize` pour trouver les 
    multiplicateurs de Lagrange optimaux. Le DataFrame est mis à jour avec 
    les valeurs optimales et les indicateurs de validité.

    Parameters
    ----------
    r : np.ndarray, shape (n_dims,)
        Vecteur d'indices r.
    s : int
        Indice entier s.
    t : int
        Indice entier t.
    Qr : np.ndarray, shape (n_dims,)
        Vecteur Qr.
    Qs : float
        Scalaire Qs.
    Qt : float
        Scalaire Qt.
    Srs : np.ndarray, shape (n_dims, n_dims)
        Matrice des coefficients S_rs.
    Sst : np.ndarray, shape (n_dims,)
        Vecteur des coefficients S_st.
    df : pandas.DataFrame
        DataFrame contenant les données du problème, qui sera mis à jour.
    neg_D_funct : Callable
        Fonction objectif à minimiser (opposée de la fonction duale D).
    contraintes_fonctions : tuple of Callables
        Ensemble de fonctions générant les contraintes linéaires (A, c).
    rows : int
        Index de la ligne du DataFrame en cours de traitement.
    columns : tuple[str, ...]
        Noms des colonnes associées aux multiplicateurs mu optimaux.
    methode : str
        Méthode d’optimisation utilisée par `scipy.optimize.minimize`
        (ex. "trust-constr", "SLSQP", etc.).
    distribution : str
        Nom de la distribution ("bern", "geom", "gauss", etc.).

    Returns
    -------
    tuple
        - df : pandas.DataFrame
            DataFrame mis à jour avec :
            - colonnes `mu_opt` contenant les multiplicateurs optimaux,
            - colonne `dual_mu_opt` avec la valeur du dual,
            - colonne `pruning` indiquant si la solution est valide (1) ou non (0).
        - index_row_to_drop : int or None
            Index de la ligne à supprimer si aucune solution faisable n’a pu être trouvée,
            sinon None.
        - A : np.ndarray
            Matrice concaténée des contraintes linéaires.
        - c : np.ndarray
            Vecteur concaténé des seconds membres des contraintes.
        - mu_optimal : np.ndarray
            Solution optimale pour les multiplicateurs de Lagrange.
    """
    A_list = []
    c_list = []
    for get_A_and_c in contraintes_fonctions:
        A, c = get_A_and_c(r, s, t, Qr, Qs, Qt, Srs, Sst)
        A_list.append(A)
        c_list.append(c)

    contraintes_list = []
    for i in range(len(A_list)):
        A = A_list[i]
        c = c_list[i]
        contrainte = LinearConstraint(A, lb=c, ub=np.inf, keep_feasible=True)
        contraintes_list.append(contrainte)

    A = np.vstack(A_list)
    c = np.hstack(c_list)

    d = len(columns)
    mu_0 = np.full(d, 2e-2)   # vecteur de taille d rempli de 2*1e-2

    if (distribution == 'bern') or (distribution == 'geom'):
        is_feasible = test_feasible_solution(mu_0, A, c)
        while (is_feasible == False) and np.all(mu_0 > 1e-12):
            mu_0 = mu_0 / 2
            is_feasible = test_feasible_solution(mu_0, A, c)
    else:
        is_feasible = test_feasible_solution(mu_0, A, c)
        while (is_feasible == False) and np.all(mu_0 > 1e-6):
            mu_0 = mu_0 / 2
            is_feasible = test_feasible_solution(mu_0, A, c)      

    if mu_0 is None:
        index_row_to_drop = rows
    else:
        solution = minimize(
            neg_D_funct,
            mu_0,
            args=(Sst, Srs, Qr, Qs, Qt, r, s, t),
            #method="trust-constr",
            method = methode,
            jac=None,
            hess=None,
            hessp=None,
            bounds=None,
            constraints=contraintes_list,
            tol=None,
            callback=None,
            options={'maxiter' : 10000}
        )

        mu_optimal = solution.x
        neg_D_mu_opt = solution.fun
        D_mu_opt = -neg_D_mu_opt

        for col, val in zip(columns, mu_optimal):
            df.loc[rows, col] = val
        df.loc[rows, f"dual_mu_opt"] = D_mu_opt

        if D_mu_opt < 0:
                df.loc[rows, f"pruning"] = 0
        else:
            df.loc[rows, f"pruning"] = 1

        df["pruning"] = df["pruning"].astype(int)
        index_row_to_drop = None
    return df, index_row_to_drop, A, c, mu_optimal



def count(df, dim) -> tuple :
    """
    Compte différentes configurations des valeurs optimales mu_opt et de la valeur duale,
    et renvoie un résumé textuel.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes :
        - "mu_opt1", "mu_opt2" : valeurs optimales de mu (si dim=2),
        - "dual_mu_opt" : valeur de la fonction duale en solution optimale.
    dim : int
        Dimension du problème (nombre de variables mu_opt).

    Returns
    -------
    tuple of str
        Ensemble de chaînes de caractères résumant :
        - le nombre de valeurs duales positives (cas pruning),
        - le nombre de valeurs duales négatives (cas non pruning),
        - si dim==2, le nombre de lignes dans chaque configuration de mu_opt1 et mu_opt2 :
          - mu_opt1 > 0 et mu_opt2 > 0
          - mu_opt1 = 0 et mu_opt2 > 0
          - mu_opt1 > 0 et mu_opt2 = 0
          - mu_opt1 = 0 et mu_opt2 = 0
    """
    nb_dual_positives = (df["dual_mu_opt"] > 0).sum()
    nb_dual_negatives = (df["dual_mu_opt"] < 0).sum()

    count = [
        f"nombre de valeurs du dual positives (cas pruning) : {nb_dual_positives}",
        f"nombre de valeurs du dual négatives (cas non pruning) : {nb_dual_negatives}",
    ]


    if dim == 2 :
        nb_mu_opt1_pos_and_mu_opt2_pos = ((df["mu_opt1"] > 0) & (df["mu_opt2"] > 0)).sum()
        nb_mu_opt1_nul_and_mu_opt2_pos = ((df["mu_opt1"] == 0) & (df["mu_opt2"] > 0)).sum()
        nb_mu_opt1_pos_and_mu_opt2_nul = ((df["mu_opt1"] > 0) & (df["mu_opt2"] == 0)).sum()
        nb_mu_opt1_nul_and_mu_opt2_nul = ((df["mu_opt1"] == 0) & (df["mu_opt2"] == 0)).sum()

        count +=[
            f"nombre de mu_opt1 > 0 et mu_opt2 > 0 (>,>) : {nb_mu_opt1_pos_and_mu_opt2_pos}",
            f"nombre de mu_opt1 = 0 et mu_opt2 > 0 (0,>) : {nb_mu_opt1_nul_and_mu_opt2_pos}",
            f"nombre de mu_opt1 > 0 et mu_opt2 = 0 (>,0) : {nb_mu_opt1_pos_and_mu_opt2_nul}",
            f"nombre de mu_opt1 = 0 et mu_opt2 = 0 (0,0) : {nb_mu_opt1_nul_and_mu_opt2_nul}",
        ]

    return count

def opti_one_row(df, rows, dim, neg_D_funct, contraintes_fonctions, columns, methode, distribution) :
    """
    Effectue l’optimisation duale pour une ligne spécifique du DataFrame.

    Cette fonction extrait les vecteurs et matrices nécessaires à partir du
    DataFrame, puis appelle `optimisation1` pour résoudre le problème dual
    avec contraintes linéaires et mettre à jour le DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les coefficients d’entrée et les colonnes de résultats.
    rows : int
        Index de la ligne du DataFrame à optimiser.
    dim : int
        Dimension du problème (nombre de variables mu).
    neg_D_funct : Callable
        Fonction objectif à minimiser (opposée de la fonction duale D).
    contraintes_fonctions : tuple of functions
        Fonctions générant les contraintes linéaires (A, c) pour la distribution.
    columns : list of str
        Noms des colonnes où stocker les valeurs optimales de mu.
    methode : str
        Méthode d’optimisation utilisée par `scipy.optimize.minimize`
        (par ex. `"trust-constr"`, `"SLSQP"`).
    distribution : str
        Nom de la distribution considérée (`"bern"`, `"geom"`, `"gauss"`, etc.).

    Returns
    -------
    tuple
        - df : pandas.DataFrame
            DataFrame mis à jour avec :
            - colonnes `mu_opt` contenant les multiplicateurs optimaux,
            - colonne `dual_mu_opt` avec la valeur du dual,
            - colonne `pruning` indiquant si la solution est valide (1) ou non (0).
        - index_row_to_drop : int | None
            Index de la ligne à supprimer si aucune solution faisable n’a été trouvée,
            sinon None.
        - A : numpy.ndarray
            Matrice concaténée des contraintes linéaires.
        - c : numpy.ndarray
            Vecteur concaténé des seconds membres des contraintes.
        - mu_optimal : numpy.ndarray
            Solution optimale des multiplicateurs de Lagrange.
    """
    Sst, Srs, Qr, Qt, Qs, t, s, r = vecteur(df, rows, dim)

    df, index_row_to_drop, A, c, mu_optimal = optimisation1(r, s, t, Qr, Qs, Qt, Srs, Sst, df, neg_D_funct, contraintes_fonctions, rows, columns, methode, distribution)
    
    return df, index_row_to_drop, A, c, mu_optimal



