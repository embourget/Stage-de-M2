import numpy as np


def decision_get_A1_and_c1(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
) -> tuple:
    """
    Génère les contraintes linéaires pour sigma_i(x) > 0.

    Parameters
    ----------
    r : np.ndarray
        Vecteur des indices, shape (n_dims,).
    s : int
        Indice entier s.
    t : int
        Indice entier t (non utilisé ici, gardé pour compatibilité).
    Qr : np.ndarray
        Matrice ou vecteur Qr, shape (n_dims, n_dims) ou (n_dims,).
    Qs : float
        Scalaire Qs (non utilisé ici).
    Qt : float
        Scalaire Qt (non utilisé ici).
    Srs : np.ndarray
        Matrice Srs, shape (n_dims, n_dims).
    Sst : np.ndarray
        Vecteur Sst, shape (n_dims,).

    Returns
    -------
    A : np.ndarray
        Matrice des coefficients des contraintes linéaires.
    c : np.ndarray
        Borne supérieure de la contrainte.
    """
    A = Sst - Srs
    c = - Sst
    return A,c 


def decision_get_A2_and_c2(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
) -> tuple:
    """
    Génère les contraintes linéaires pour sigma_i(x) - 1 > 0.

    Parameters
    ----------
    (idem decision_get_A1_and_c1)

    Returns
    -------
    A : np.ndarray
        Matrice des coefficients.
    c : np.ndarray
        Borne supérieure de la contrainte.
    """
    A = Sst - Srs
    c = 1 - Sst
    return A,c 



def decision_get_A3_and_c3(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
) -> tuple:
    """
    Génère les contraintes linéaires pour x > 0 (positivité des variables).

    Parameters
    ----------
    (idem decision_get_A1_and_c1)

    Returns
    -------
    A : np.ndarray
        Matrice identité de taille (n_dims, n_dims).
    c : np.ndarray
        Vecteur nul, shape (n_dims,).
    """
    n = Qr.shape[0]  # je veux le nb de lignes
    A = np.eye(n)
    c = np.zeros(n)
    return A,c  


def decision_get_A4_and_c4(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
) -> tuple:
    """
    Génère les contraintes linéaires pour 1 - sigma_i(x) > 0.

    Parameters
    ----------
    (idem decision_get_A1_and_c1)

    Returns
    -------
    A : np.ndarray
        Matrice des coefficients des contraintes linéaires.
    c : np.ndarray
        Borne supérieure de la contrainte.
    """
    A = - (Sst - Srs)
    c = - 1 + Sst
    return A,c  


def decision_get_A5_and_c5(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
) -> tuple:
    """
    Génère les contraintes linéaires pour sigma_i(x) + 1 > 0.

    Parameters
    ----------
    (idem decision_get_A1_and_c1)

    Returns
    -------
    A : np.ndarray
        Matrice des coefficients des contraintes linéaires.
    c : np.ndarray
        Borne supérieure de la contrainte.
    """
    A = Sst - Srs
    c = - 1 - Sst
    return A,c 