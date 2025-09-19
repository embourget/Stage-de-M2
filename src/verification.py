import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
import pandas as pd

from constraints import *
from decision_constraints import *
from dual import *
from decision import *
from optim import *

def plot_feasible_region(A, c, x_range=(-10, 10), y_range=(-10, 10), resolution=400):
    """
    Affiche la région faisable définie par les contraintes linéaires A μ ≥ c en 2D.

    Parameters
    ----------
    A : ndarray, shape (m, 2)
        Matrice des contraintes linéaires.
    c : ndarray, shape (m,)
        Second membre des contraintes.
    x_range : tuple of float
        Intervalle des valeurs pour μ1.
    y_range : tuple of float
        Intervalle des valeurs pour μ2.
    resolution : int
        Résolution de la grille pour le shading.
    
    Returns
    -------
    None
        Affiche la région faisable et les lignes représentant A μ = c.
        Grid resolution for shading.
    """
    
    # Create grid
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    
    # Check constraints: A mu >= c
    is_feasible = np.all((A @ points) >= c[:, None], axis=0)
    Z = is_feasible.reshape(X.shape)
    
    # Plot feasible region
    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, Z, levels=[0.5, 1], colors=["lightblue"], alpha=0.6)
    
    # Plot boundary lines (A mu = c)
    for i in range(A.shape[0]):
        if A[i, 1] != 0:  # avoid vertical division by zero
            y_line = (c[i] - A[i, 0] * x) / A[i, 1]
            plt.plot(x, y_line, "k--")
        else:
            # vertical line case
            x_line = np.full_like(y, c[i] / A[i, 0])
            plt.plot(x_line, y, "k--")
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\mu_2$")
    plt.title("Feasible Region for A μ ≥ c")
    plt.grid(True, linestyle=":")
    plt.show()



def D_max_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, A=None, c=None, resolution=200, x_range=(0,1), y_range=(0,1)):
    """
    Calcule la valeur maximale de D(μ) sur une grille 2D dans la région faisable.

    Parameters
    ----------
    D_funct : callable
        Fonction duale D(μ, Sst, Srs, Qr, Qs, Qt, r, s, t).
    Sst, Srs, Qr, Qs, Qt, r, s, t : divers
        Paramètres nécessaires à D_funct.
    A, c : ndarray or None
        Contraintes linéaires A μ ≥ c. Si None, aucun filtrage de faisabilité.
    resolution : int
        Nombre de points par axe pour la grille.
    x_range, y_range : tuple of float
        Bornes pour μ1 et μ2.

    Returns
    -------
    tuple
        - mu_max : tuple of float
            Coordonnées (μ1, μ2) qui maximisent D.
        - D_max : float
            Valeur maximale de D(μ).
    """
    mu1 = np.linspace(*x_range, resolution)
    mu2 = np.linspace(*y_range, resolution)
    M1, M2 = np.meshgrid(mu1, mu2)

    Z = np.full_like(M1, -np.inf, dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            mu = np.array([M1[i, j], M2[i, j]])
            
            if A is not None and c is not None:
                if not np.all(A @ mu >= c - 1e-9):  # tolérance numérique
                    continue  # point non faisable
            
            val = D_funct(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)
            if not np.isnan(val):
                Z[i, j] = val

    if np.all(Z == -np.inf):
        print("⚠️ Aucune zone faisable détectée")
        mu_max, D_max = None, None
    else:
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        mu_max = (M1[max_idx], M2[max_idx])
        D_max = Z[max_idx]
    return mu_max, D_max


def plot_D_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, 
                   mu_vals, dual_val, A=None, c=None, 
                   resolution=200, x_range=(0,1), y_range=(0,1)):
    """
    Affiche une heatmap de D(μ) dans la région faisable et marque les points d'intérêt.

    Parameters
    ----------
    D_funct : callable
        Fonction duale D(μ, Sst, Srs, Qr, Qs, Qt, r, s, t).
    Sst, Srs, Qr, Qs, Qt, r, s, t : divers
        Paramètres nécessaires à D_funct.
    mu_vals : array-like or None
        Solution candidate (μ1, μ2) à afficher sur la heatmap.
    dual_val : float
        Valeur de D en mu_vals.
    A, c : ndarray or None
        Contraintes linéaires A μ ≥ c. Si None, aucun filtrage.
    resolution : int
        Nombre de points par axe pour la grille.
    x_range, y_range : tuple of float
        Bornes pour μ1 et μ2.

    Returns
    -------
    tuple
        - mu_max : tuple of float
            Coordonnées du maximum de D trouvé sur la grille.
        - D_max : float
            Valeur maximale de D(μ) sur la grille.
    """

    mu1 = np.linspace(*x_range, resolution)
    mu2 = np.linspace(*y_range, resolution)
    M1, M2 = np.meshgrid(mu1, mu2)
    
    Z = np.full_like(M1, -np.inf, dtype=float)  # -inf par défaut (hors faisable)
    
    for i in range(resolution):
        for j in range(resolution):
            mu = np.array([M1[i, j], M2[i, j]])
            
            # Vérifier faisabilité si contraintes données
            if A is not None and c is not None:
                if not np.all(A @ mu >= c - 1e-9):  # tolérance numérique
                    continue  # point non faisable
            
            val = D_funct(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)
            if not np.isnan(val):
                Z[i, j] = val
    
    plt.figure(figsize=(6,5))
    plt.imshow(Z, extent=[*x_range, *y_range], origin='lower', 
               aspect='auto', cmap='viridis')
    plt.colorbar(label='D(mu)')
    plt.xlabel('mu_1')
    plt.ylabel('mu_2')
    plt.title('Heatmap de D(mu) sur le domaine de faisabilité')

    # Maximum dans la région faisable
    if np.all(Z == -np.inf):
        print("⚠️ Aucune zone faisable détectée")
        mu_max, D_max = None, None
    else:
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        mu_max = (M1[max_idx], M2[max_idx])
        D_max = Z[max_idx]
        plt.scatter(mu_max[0], mu_max[1], color='red', s=50, marker='x', 
                    label=f"max D={D_max:.3f}")

    # Ajouter la solution d'optimisation si dispo
    if mu_vals is not None:
        plt.scatter(mu_vals[0], mu_vals[1], color='orange', s=40, marker='o', 
                    label=f"dual_val={dual_val:.3f}")
    
    plt.legend()
    plt.show()
    return mu_max, D_max



def feasible_region(rows, fpath, distribution, dim) :
    """
    Calcule et affiche les variables intermédiaires et la région faisable
    pour une ligne donnée d'un DataFrame.

    Parameters
    ----------
    rows : int
        Indice de la ligne du DataFrame à analyser.
    fpath : str ou Path
        Chemin vers le fichier CSV contenant les données.
    distribution : str
        Nom de la distribution utilisée.
    dim : int
        Dimension du problème (nombre de variables μ).

    Returns
    -------
    None
        Affiche :
        - la matrice et vecteurs calculés (Sst, Srs, Qr, Qt, Qs, t, s, r),
        - les contraintes A, c,
        - la région faisable (si dim == 2).
    """
    columns = []
    df = get_csv(fpath)
    columns, df = ajout_colonnes(df, dim)
    Sst, Srs, Qr, Qt, Qs, t, s, r = vecteur(df, rows, dim) 
    (D_funct, neg_D_funct), contraintes_fonctions = get_D(distribution)

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

    names = ["Sst", "Srs", "Qr", "Qt", "Qs", "t", "s", "r", "A", "c"]
    values = [Sst, Srs, Qr, Qt, Qs, t, s, r, A, c]

    if dim == 2:
        plot_feasible_region(A, c, x_range=(-1, 3), y_range=(-1, 3))
        #plot_D_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, mu_vals, dual_val)
    
    for name, val in zip(names, values):
        if isinstance(val, np.ndarray):
            print(f"{name} =\n{np.array2string(val, precision=3, suppress_small=True)}\n")
        else:
            print(f"{name} = {val}\n")
    
    return


def verif(rows, fpath, distribution, dim, methode) :
    """
    Vérifie et analyse une ligne d'un DataFrame en résolvant le problème dual.

    Cette fonction :
    - extrait les vecteurs et matrices nécessaires,
    - effectue l’optimisation duale,
    - affiche les variables intermédiaires,
    - trace la région faisable et la heatmap de D(μ) si dim == 2.

    Parameters
    ----------
    rows : int
        Indice de la ligne du DataFrame à traiter.
    fpath : str ou Path
        Chemin vers le fichier CSV contenant les données.
    distribution : str
        Nom de la distribution utilisée.
    dim : int
        Dimension du problème (nombre de variables μ).
    methode : str
        Méthode d'optimisation pour `scipy.optimize.minimize`.

    Returns
    -------
    None
        Affiche :
        - variables intermédiaires calculées,
        - région faisable et heatmap de D(μ),
        - points maximaux et solution duale.
    """
    columns = []
    df = get_csv(fpath)
    columns, df = ajout_colonnes(df, dim)
    Sst, Srs, Qr, Qt, Qs, t, s, r = vecteur(df, rows, dim) 
    (D_funct, neg_D_funct), contraintes_fonctions = get_D(distribution)

    df, index_to_drop, A, c, mu_opt = opti_one_row(df, rows, dim, neg_D_funct, contraintes_fonctions, columns, methode, distribution)
    #df.loc[3, ["mu_opt1", "mu_opt2"]] = [2, 3]
    cols = [f"mu_opt{j}" for j in range(1, dim+1)]
    mu_vals = df.loc[rows, cols].to_numpy()
    dual_val = df.loc[rows, "dual_mu_opt"]
    
    names = ["Sst", "Srs", "Qr", "Qt", "Qs", "t", "s", "r", "A", "c", "mu_vals", "dual_val", "mu_opt"]
    values = [Sst, Srs, Qr, Qt, Qs, t, s, r, A, c, mu_vals, dual_val, mu_opt]

    if dim == 2:
        plot_feasible_region(A, c, x_range=(-1, 3), y_range=(-1, 3))
        #plot_D_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, mu_vals, dual_val)
        mu_max, D_max = plot_D_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, mu_vals, dual_val, A, c)
        names += ["mu_max","D_max"]
        values += [mu_max,D_max ]
    
    for name, val in zip(names, values):
        if isinstance(val, np.ndarray):
            print(f"{name} =\n{np.array2string(val, precision=3, suppress_small=True)}\n")
        else:
            print(f"{name} = {val}\n")
    return 

