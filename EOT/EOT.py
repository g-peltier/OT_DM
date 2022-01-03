from numba import jit
import numpy as np
from scipy.optimize import linprog

def hello():
    return 'hello'

def new_hello():
    return 'hello2'

## Acceleration of the functions in EOT_sinkhorn
@jit(nopython=True)
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

@jit(nopython=True)
def compute_EOT_dual(lam, alpha, beta, denom, reg, a, b):
    n = np.shape(a)[0]
    m = np.shape(b)[0]
    res = reg * (np.dot(a, np.log(alpha)) + np.dot(b, np.log(beta)))
    res_trans = reg * (np.log(denom))
    return res - res_trans + reg * np.log(n * m)

@jit(nopython=True) #reg est le epsilon de l'algorithme
def EOT_PSinkhorn(C, reg, a, b, max_iter=100000, tau=1e-10, stable=0):
    acc = np.zeros(max_iter+1) # donne la valeur de la foction obj a chaque iteration
    
    N, n, m = C.shape
    
    K = np.exp(-(C / reg))
    
    L = np.max(np.abs(C)) ** 2 / reg
    lam = (1 / N) * np.ones(N) #lambda de l'article
    alpha, beta = np.ones(n), np.ones(m) # remplace f (alpha) et g (beta)

    K_trans = np.zeros((N, n, m))
    K_lam_trans = np.zeros((N, n, m))
    
    denom = 1
    stop = 1
    k = 1

    while stop > tau and k < max_iter:

        for j in range(N):
            K_trans[j, :, :] = K[j, :, :] ** lam[j] # les K^lambda
            K_lam_trans[j, :, :] = K_trans[j, :, :].copy() * C[j, :, :]# les K^lambda*C

        K_update = np.sum(K_trans, axis=0) #le nouveau K

        # Update alpha ie f
        alpha_trans = np.dot(K_update, beta) + stable # produit K.g
        denom = np.dot(alpha, alpha_trans) # c'est ck 
        alpha = denom * (a / alpha_trans) #nouveau fk

        # Update beta ie g
        beta_trans = np.dot(K_update.T, alpha) + stable
        denom = np.dot(beta, beta_trans) # dk
        beta = denom * (b / beta_trans)

        # Update lam
        lam_trans = np.zeros((N, n))
        for i, k_lam_trans in enumerate(K_lam_trans):
            lam_trans[i] = np.dot(k_lam_trans, beta)

        lam_trans = lam_trans * alpha
        grad_lam = np.sum(lam_trans, axis=1) / (denom + stable)
        lam_trans = lam + (1 / L) * grad_lam
        lam = projection_simplex_sort(lam_trans)
            
        # Update the total cost
        acc[k] = compute_EOT_dual(lam, alpha, beta, denom, reg, a, b)
        stop = np.abs(acc[k] - acc[k-1])
        k = k + 1
    
    # on renvoie la dernière valeur de acc qui est la valeur de la fonction objectif finale
    # lam : lambda, alpha :f, beta : g , denom -ck-ck' : produit f.K.g,  
    return acc[-1], acc[1:k-1], lam, alpha, beta, denom, K_lam_trans, K_trans

## Linear Program: primal formulation of EOT
def LP_solver_Primal(M, a, b):

    if len(np.shape(M)) != 3:
        M = np.expand_dims(M, axis=0)

    N, n, m = np.shape(M)

    M_flat = []
    for k in range(N):
        M_flat.append(M[k, :, :].flatten("C"))

    A_ub = np.zeros((N, N * n * m + 1))
    A_ub[:, 0] = -1
    for k in range(N):
        A_ub[k, 1 + k * n * m : 1 + (k + 1) * n * m] = M_flat[k]

    b_ub = np.zeros(N)

    A_eq = np.zeros((n + m, N * n * m + 1))
    for i in range(n):
        for k in range(N):
            A_eq[i, 1 + k * m * n + m * i : 1 + k * m * n + m * (i + 1)] = 1

    for j in range(m):
        for k in range(N):
            ind_j = [1 + k * m * n + i * m + j for i in range(n)]
            A_eq[n + j, ind_j] = 1

    A_eq = A_eq[:-1, :]

    b_eq = np.zeros(n + m - 1)
    b_eq[:n] = a
    b_eq[n:] = b[:-1]

    c = np.zeros(N * n * m + 1)
    c[0] = 1

    bounds = [(None, None)] + [(0, None)] * (N * n * m)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res

