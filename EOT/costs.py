import numpy as np

## Cost functions

def hello_world():
    print('hello world')

def alpha_L1_Distance(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin), 2)
    C = C ** (alpha)
    return C


def Square_Euclidean_Distance(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    return C


def alpha_Euclidean_Distance(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = (np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))) ** (alpha)
    return C


def alpha_Euclidean_cost_equivalent(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = (np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))) ** (alpha)
    C_den = 1 + C
    C = C / C_den
    return C


def Lp_to_the_p_cost(X, Y, p=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin) ** p, 2)
    return C


def Trivial_cost(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))
    res = C == 0
    res = 1 - res.astype(float)

    return res


def Angle_cost(X, Y, stable=1e-6):
    C = np.dot(X, Y.T)
    norm_X = np.sqrt(np.sum(X ** 2, 1)) + stable
    norm_Y = np.sqrt(np.sum(Y ** 2, 1)) + stable
    C = C / norm_X
    C = C / norm_Y

    return C


def Teleport_cost_alpha(X, Y, stable=1e-6, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C_inv = np.sqrt(np.sum(np.abs(X_col - Y_lin) ** 2, 2)) ** alpha + stable
    C = 1 / C_inv

    return C


def Cos_cost(X, Y):
    C = np.dot(X, Y.T)
    C = np.cos(C) + 1

    return C


def cost_fixed_line(n, m, cost_function):
    X = np.arange(n).reshape(n, 1)
    Y = np.arange(m).reshape(m, 1)

    C = cost_function(X / n, Y / n)

    return C


def cost_matrix_sequential(alpha, w, x, y):
    res = alpha_Euclidean_Distance(x, y)

    xw = x.dot(w)
    yw = y.dot(w)

    wxy = np.tile(xw, (len(yw), 1)).T - np.tile(yw, (len(xw), 1))

    return res - alpha * wxy


def N_cost_matrices(alpha, x, y, N, seed_init=49):
    n = len(x)
    m = len(y)
    C = np.zeros((N, n, m))
    seed = seed_init
    np.random.seed(seed)
    for i in range(N):
        rho = np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        w = [rho * np.cos(theta), rho * np.sin(theta)]
        C[i, :, :] = cost_matrix_sequential(alpha, w, x, y)
        seed = seed + 1
        np.random.seed(seed)

    return C