import numpy as np


def get_A1_and_c1(
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
    Génère les contraintes linéaires pour s_i(mu) > 0.

    Parameters
    ----------
    r : np.ndarray
        Vecteur des indices, shape (n_dims,).
    s : int
        Indice entier s.
    t : int
        Indice entier t (non utilisé ici, gardé pour compatibilité).
    Qr : np.ndarray
        Vecteur Qr, shape (n_dims,).
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
    n_dims = Qr.shape[0]
    A = np.empty((n_dims, n_dims))
    for i in range(n_dims):
        factor = -1 if s > r[i] else 1
        A[:, i] = factor * Srs[:, i]
    c = -Sst
    return A, c


def get_A2_and_c2(
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
    Génère les contraintes linéaires pour s_i(mu) - l(mu) > 0.
    """
    n_dims = Qr.shape[0]
    A = np.empty((n_dims, n_dims))
    for i in range(n_dims):
        factor = -1 if s > r[i] else 1
        A[:, i] = factor * (Srs[:, i] - 1)
    c = -Sst + 1
    return A, c


def get_A3_and_c3(
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
    Génère les contraintes linéaires pour s_i(mu) + l(mu) > 0.
    """
    n_dims = Qr.shape[0]
    A = np.empty((n_dims, n_dims))
    for i in range(n_dims):
        factor = -1 if s > r[i] else 1
        A[:, i] = factor * (Srs[:, i] + 1)
    c = -Sst - 1
    return A, c


def get_A4_and_c4(
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
    Génère les contraintes linéaires pour - s_i(mu) + l(mu) > 0.
    """
    n_dims = Qr.shape[0]
    A = np.empty((n_dims, n_dims))
    for i in range(n_dims):
        factor = -1 if s > r[i] else 1
        A[:, i] = factor * (-Srs[:, i] + 1)
    c = Sst - 1
    return A, c


def get_A5_and_c5(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
):
    """
    Génère les contraintes linéaires pour l(mu)>0.
    """
    n_dims = Qr.shape[0]
    A = np.empty((1, n_dims))
    for i in range(n_dims):
        factor = -1 if s > r[i] else 1
        A[0, i] = factor
    c = np.array([-1])
    return A, c


def get_A6_and_c6(
    r: np.ndarray,
    s: int,
    t: int,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    Srs: np.ndarray,
    Sst: np.ndarray,
):
    """
    Génère les contraintes linéaires pour mu > 0.

    Parameters
    ----------
    r, s, t, Qr, Qs, Qt, Srs, Sst : divers
        Arguments conservés pour cohérence d'interface mais non utilisés ici.

    Returns
    -------
    A : np.ndarray
        Matrice identité de taille (n_dims, n_dims).
    c : np.ndarray
        Vecteur nul de taille (n_dims,).
    """
    n = Qr.shape[0]  # je veux le nb de lignes
    A = np.eye(n)
    c = np.zeros(n)
    return A, c
