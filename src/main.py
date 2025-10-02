import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
import pandas as pd

from pathlib import Path

import pandas as pd
from scipy.optimize import (Bounds, LinearConstraint, NonlinearConstraint,
                            minimize)
from scipy.optimize import linprog

from constraints import *
from dual import *
from optim import *
from verification import *




def main(fpath: Path, distribution: str, dim: int, methode: str):
    """
    Exécute le pipeline complet d’optimisation duale sur toutes les lignes
    d’un fichier CSV et retourne les résultats filtrés avec analyse des distances.

    Cette fonction effectue les opérations suivantes :
    - Lecture du fichier CSV et ajout des colonnes mu_opt et indicateurs.
    - Optimisation duale pour chaque ligne avec `opti_one_row`.
    - Calcul du maximum de la fonction duale D sur une grille 2D (si dim==2)
      et distance euclidienne entre solution optimale et maximum sur grille.
    - Collecte des lignes échouées et filtrage des lignes non faisables.
    - Calcul de statistiques résumant les cas de dual positif/négatif et
      des combinaisons de mu_opt.
    - Affichage de KDE plots pour mu_opt et mu_max_heatmap si la distribution
      n’est pas exponentielle.

    Parameters
    ----------
    fpath : pathlib.Path
        Chemin vers le fichier CSV contenant les données.
    distribution : str
        Nom de la distribution utilisée. Doit être parmi :
        {"poisson", "gauss", "exp", "geom", "bern", "negbin"}.
    dim : int
        Dimension du problème (nombre de variables mu).
    methode : str
        Méthode d’optimisation passée à `scipy.optimize.minimize`.

    Returns
    -------
    tuple
        - df : pandas.DataFrame
            DataFrame mis à jour avec les résultats des optimisations
            (mu_opt, dual_mu_opt, pruning), les valeurs max sur heatmap et
            la distance euclidienne entre mu_opt et mu_max_heatmap.
        - compte : list of str
            Statistiques textuelles détaillant :
            - cas positifs/négatifs du dual,
            - combinaisons de mu_opt,
            - lignes avec distance > 0.01,
            - lignes et nombres d’échecs d’optimisation.
    """
    df = get_csv(fpath)
    columns, df = ajout_colonnes(df, dim)

    # Colonnes pour heatmap
    df = df.assign(
        mu_max1_heatmap=np.nan,
        mu_max2_heatmap=np.nan,
        D_max_heatmap=np.nan,
        distance_euclidienne=np.nan
    )

    rows_to_drop = []
    (D_funct, neg_D_funct), contraintes_fonctions = get_D(distribution)
    failed_rows = []

    for rows in df.index.copy():
        Sst, Srs, Qr, Qt, Qs, t, s, r = vecteur(df, rows, dim)
        try:
            df, index_row_to_drop, A, c, mu_optimal = opti_one_row(
                df, rows, dim, neg_D_funct, contraintes_fonctions, columns, methode, distribution
            )
            
            mu_max, D_max = D_max_heatmap(D_funct, Sst, Srs, Qr, Qs, Qt, r, s, t, A, c)
            
            if mu_max is not None and mu_optimal is not None:
                df.loc[rows, "distance_euclidienne"] = np.linalg.norm(np.array(mu_max)-np.array(mu_optimal))
                df.loc[rows, "mu_max1_heatmap"] = mu_max[0]
                df.loc[rows, "mu_max2_heatmap"] = mu_max[1]
                df.loc[rows, "D_max_heatmap"] = D_max
            # else:
            #     df.loc[rows, "distance_euclidienne"] = np.nan
            #     df.loc[rows, "mu_max1_heatmap"] = np.nan
            #     df.loc[rows, "mu_max2_heatmap"] = np.nan
            #     df.loc[rows, "D_max_heatmap"] = np.nan

            if index_row_to_drop is not None:
                rows_to_drop.append(index_row_to_drop)

        except Exception as e:
            failed_rows.append(rows)
            df.loc[rows, ["distance_euclidienne", "mu_max1_heatmap", "mu_max2_heatmap", "D_max_heatmap"]] = np.nan
            #print(f"Ligne {rows} échouée : {e}")

    df = df.drop(index=rows_to_drop).reset_index(drop=True)

    compte = count(df, dim)
    somme = (df["distance_euclidienne"] > 0.01).sum()
    indices = df.index[df["distance_euclidienne"] > 0.01].tolist()
    a = len(failed_rows)
    compte += [
        f"nombre de lignes avec distance > 0.01 : {somme}",
        f"indices des lignes avec distance > 0.01 : {indices}",
        f"indices des lignes échec optimisation: {failed_rows}",
        f"nombres échecs optimisation: {a}",
    ]
    
    if distribution != 'exp' :
        for x_col, y_col, title in [
            ("mu_opt1", "mu_opt2", "Répartition des mu_opt"),
            ("mu_max1_heatmap", "mu_max2_heatmap", "Répartition des mu_max_heatmap")
        ]:
            valid_df = df[[x_col, y_col]].dropna()
            if not valid_df.empty:
                plt.figure(figsize=(6,5))
                kde = sns.kdeplot(
                    data=valid_df,
                    x=x_col,
                    y=y_col,
                    fill=True,
                    cmap="viridis",
                    thresh=0
                )
                if kde.collections:
                    plt.colorbar(kde.collections[0], label="Densité estimée")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title)
                plt.show()

                plt.figure(figsize=(6,5))
                sns.kdeplot(valid_df[x_col], label=x_col, fill=True)
                sns.kdeplot(valid_df[y_col], label=y_col, fill=True)
                plt.xlabel(f"{x_col} et {y_col}")
                plt.ylabel("Densité")
                plt.title(f"Densités des {x_col} et {y_col}")
                plt.legend()
                plt.show()
   

    return df, compte



def main_opti(fpath: Path, distribution: str, dim: int, methode: str):
    """
    Exécute le pipeline complet d’optimisation duale sur toutes les lignes
    d’un fichier CSV et retourne les résultats filtrés sans calcul des heatmaps.

    Cette fonction effectue les opérations suivantes :
    - Lecture du fichier CSV et ajout des colonnes mu_opt et indicateurs.
    - Optimisation duale pour chaque ligne avec `opti_one_row`.
    - Collecte des lignes échouées et suppression des lignes non faisables.
    - Calcul de statistiques résumant les cas de dual positif/négatif et
      des combinaisons de mu_opt.

    Parameters
    ----------
    fpath : pathlib.Path
        Chemin vers le fichier CSV contenant les données.
    distribution : str
        Nom de la distribution utilisée.
    dim : int
        Dimension du problème (nombre de variables mu).
    methode : str
        Méthode d’optimisation passée à `scipy.optimize.minimize`.

    Returns
    -------
    tuple
        - df : pandas.DataFrame
            DataFrame mis à jour avec les résultats des optimisations
            (mu_opt, dual_mu_opt, pruning), après suppression des lignes
            non faisables.
        - compte : list of str
            Statistiques textuelles détaillant les lignes échouées et nombres
            d’échecs d’optimisation.
    """
    df = get_csv(fpath)
    columns, df = ajout_colonnes(df, dim)


    rows_to_drop = []
    (D_funct, neg_D_funct), contraintes_fonctions = get_D(distribution)
    failed_rows = []

    for rows in df.index.copy():
        Sst, Srs, Qr, Qt, Qs, t, s, r = vecteur(df, rows, dim)
        try:
            df, index_row_to_drop, A, c, mu_optimal = opti_one_row(
                df, rows, dim, neg_D_funct, contraintes_fonctions, columns, methode, distribution
            )

            if index_row_to_drop is not None:
                rows_to_drop.append(index_row_to_drop)

        except Exception as e:
            failed_rows.append(rows)
            rows_to_drop.append(rows)

    df = df.drop(index=rows_to_drop).reset_index(drop=True)

    compte = count(df, dim)
    a = len(failed_rows)
    compte += [
        f"indices des lignes échec optimisation: {failed_rows}",
        f"nombres échecs optimisation: {a}",
    ]
   

    return df, compte

