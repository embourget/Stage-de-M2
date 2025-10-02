import numpy as np
from scipy.special import xlogy

def Decision_gauss(
    x: np.ndarray,
    Sst: np.ndarray,
    Srs: np.ndarray,
    Qr: np.ndarray,
    Qs: float,
    Qt: float,
    r: np.ndarray,
    s: int,
    t: int,
    ):
    """
    Calcule la fonction de décision associée au modèle gaussien.

    Parameters
    ----------
    x : np.ndarray
        Vecteur des multiplicateurs de Lagrange, shape (n_dims,).
    Sst : np.ndarray
        Vecteur Sst, shape (n_dims,).
    Srs : np.ndarray
        Matrice Srs, shape (n_dims, n_dims).
    Qr : np.ndarray
        Vecteur Qr, shape (n_dims,).
    Qs : float
        Scalaire Qs.
    Qt : float
        Scalaire Qt.
    r : np.ndarray
        Vecteur des indices, shape (n_dims,).
    s : int
        Indice entier s.
    t : int
        Indice entier t.

    Returns
    -------
    float
        Valeur de la fonction de décision pour la loi normale.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = sigma ** 2
    B = -0.5 * np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_gauss(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi normale.

    Returns
    -------
    float
        -Decision_gauss(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_gauss(x, Sst, Srs, Qr, Qs, Qt, r, s, t)


def Decision_poisson(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction de décision associée au modèle de Poisson.

    Returns
    -------
    float
        Valeur due la fonction de décision pour la loi de Poisson.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = - sigma + xlogy(sigma, sigma)
    B = - np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_poisson(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi de Poisson.

    Returns
    -------
    float
        -Decision_poisson(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_poisson(x, Sst, Srs, Qr, Qs, Qt, r, s, t)


def Decision_exp(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction de décision associée au modèle exponentiel.

    Returns
    -------
    float
        Valeur due la fonction de décision pour la loi exponentielle.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = xlogy(1, sigma) + 1
    B = - np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_exp(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi exponentielle.

    Returns
    -------
    float
        -Decision_exp(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_exp(x, Sst, Srs, Qr, Qs, Qt, r, s, t)


def Decision_geom(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction de décision associée au modèle géométrique.

    Returns
    -------
    float
        Valeur due la fonction de décision pour la loi géométrique.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = xlogy(sigma - 1, sigma - 1) - xlogy(sigma, sigma)
    B = - np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_geom(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi géométrique.

    Returns
    -------
    float
        -Decision_geom(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_geom(x, Sst, Srs, Qr, Qs, Qt, r, s, t)


def Decision_bern(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction de décision associée au modèle de Bernoulli.

    Returns
    -------
    float
        Valeur due la fonction de décision pour la loi de Bernoulli.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = xlogy(1-sigma, 1-sigma) + xlogy(sigma, sigma)
    B = - np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_bern(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi de Bernoulli.

    Returns
    -------
    float
        -Decision_bern(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_bern(x, Sst, Srs, Qr, Qs, Qt, r, s, t)


def Decision_negbin(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction de décision associée au modèle binomial négatif.

    Returns
    -------
    float
        Valeur due la fonction de décision pour la loi binomiale négative.
    """
    DeltaSrst = Sst - Srs
    sigma = Sst + np.sum(DeltaSrst * x, axis = 1)
    A = xlogy(sigma, sigma) - xlogy(1+sigma, 1+sigma)
    B = - np.sum(A)
    r = r[r != s]
    Qrs = (Qs - Qr) / (r -s)
    Qst = (Qt - Qs) /  (t-s)
    DeltaQrst = Qst - Qrs
    l = -np.sum(DeltaQrst * x)
    Phi = l - Qst
    return B + Phi

def neg_Decision_negbin(x, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur de la fonction de décision pour la loi binomiale négative.

    Returns
    -------
    float
        -Decision_negbin(x, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -Decision_negbin(x, Sst, Srs, Qr, Qs, Qt, r, s, t)