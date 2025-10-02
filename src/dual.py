import numpy as np
from scipy.special import xlogy


def D_poisson(
    mu: np.ndarray,
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
    Calcule la fonction duale associée au modèle de Poisson.

    Parameters
    ----------
    mu : np.ndarray
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
        Valeur du dual pour la loi de Poisson.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[
        r != s
    ]  # [r != s] renvoie un vecteur (True, False, True, True....) et r[r != s] garde uniquement les composantes du vecteur r où r != s = True => ici on supprime les lignes où r=s
    A = xlogy(S, S) - xlogy(S, l) - S
    B = -np.sum(A)
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_poisson(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi de Poisson

    Returns
    -------
    float
        -D_poisson(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_poisson(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)


def D_gauss(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction duale associée au modèle de Gauss.

    Returns
    -------
    float
        Valeur du dual pour la loi gaussienne.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[
        r != s
    ]  # [r != s] renvoie un vecteur (True, False, True, True....) et r[r != s] garde uniquement les composantes du vecteur r où r != s = True => ici on supprime les lignes où r=s
    A = S**2
    B = -0.5 * np.sum(A) / l
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_gauss(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi normale.

    Returns
    -------
    float
        -D_gauss(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_gauss(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)


def D_exp(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction duale associée au modèle exponentiel.

    Returns
    -------
    float
        Valeur du dual pour la loi exponentielle.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[r != s]
    A = xlogy(S, S) - xlogy(l, S) - l
    B = -np.sum(A)
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_exp(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi exponentielle.

    Returns
    -------
    float
        -D_exp(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_exp(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)


def D_geom(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction duale associée au modèle géométrique.

    Returns
    -------
    float
        Valeur du dual pour la loi géométrique.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[r != s]
    a = S - l
    A = xlogy(a, a) - xlogy(S, S) + xlogy(l, l)
    B = -np.sum(A)
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_geom(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi géométrique.

    Returns
    -------
    float
        -D_geom(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_geom(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)


def D_bern(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction duale associée au modèle de Bernoulli.

    Returns
    -------
    float
        Valeur du dual pour la loi de Bernoulli.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[r != s]
    a = l - S
    A = xlogy(a, a) - xlogy(S, S) + xlogy(l, l)
    B = -np.sum(A)
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_bern(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi de Bernoulli.

    Returns
    -------
    float
        -D_bern(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_bern(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)


def D_negbin(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule la fonction duale associée au modèle binomial négatif.

    Returns
    -------
    float
        Valeur du dual pour la loi binomiale négative.
    """
    sign = np.where(r < s, -1, 1)
    sum_mu = np.sum(mu * sign)
    l = 1 + sum_mu
    S = Sst + np.sum((sign * mu) * Srs, axis=1)
    r = r[r != s]
    a = S + l
    A = -xlogy(a, a) + xlogy(S, S) + xlogy(l, l)
    B = -np.sum(A)
    Qrs = (Qs - Qr) / (s - r)
    L = -np.sum(sign * mu * Qrs) - (Qt - Qs) / (t - s)
    return B + L


def neg_D_negbin(mu, Sst, Srs, Qr, Qs, Qt, r, s, t):
    """
    Calcule l'opposé de la valeur du dual pour la loi binomiale négative.

    Returns
    -------
    float
        -D_negbin(mu, Sst, Srs, Qr, Qs, Qt, r, s, t).
    """
    return -D_negbin(mu, Sst, Srs, Qr, Qs, Qt, r, s, t)
