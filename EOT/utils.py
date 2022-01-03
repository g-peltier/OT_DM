import numpy as np
import EOT.EOT as eot
import matplotlib.pyplot as plt

COLORS = ['tab:green', 'tab:purple', 'tab:orange']

## Generate samples`
def simul_two_Gaussians(num_samples):

    mean_X = np.array([3, 3])
    cov_X = np.array([[1, 0], [0, 1]])
    X = np.random.multivariate_normal(mean_X, cov_X, num_samples)

    mean_Y = np.array([4, 4])
    cov_Y = np.array([[1, -0.2], [-0.2, 1]])
    Y = np.random.multivariate_normal(mean_Y, cov_Y, num_samples)

    return X, Y

def price_of_players(C,P):
    mult = np.multiply(C.astype(float),P.astype(float))
    prices = [np.sum(mult[i]) for i in range(len(C))]
    return prices

def print_ot(X, Y, lst_coupl_matrix=None, agent=None):
    fig, ax = plt.subplots()
    
    lst_coupl_matrix = [lst_coupl_matrix[agent]] if agent is not None else lst_coupl_matrix 
    colors = [COLORS[agent]] if agent is not None else COLORS
    
    if lst_coupl_matrix is not None:
        for p, coupl_matrix in enumerate(lst_coupl_matrix):
            for i in range(len(coupl_matrix)):
                for j in range(len(coupl_matrix[i])):
                    if coupl_matrix[i][j]:
                        ax.plot([X[i][0], Y[j][0]], [X[i][1], Y[j][1]],
                                colors[p], lw=coupl_matrix[i][j]*30, zorder=1)

    ax.scatter(X[:, 0], X[:, 1], 100, 'b', zorder=2)
    ax.scatter(Y[:, 0], Y[:, 1], 100, 'r', marker='s', zorder=2)
    ax.axis('off')
   
    plt.show()
    
def print_ot_and_cost(X, Y, lst_coupl_matrix=None, cost_matrix = None, agent=None):
    fig, ax = plt.subplots()
    
    prices = price_of_players(cost_matrix,np.array(lst_coupl_matrix))
    #print("prices",prices)
    
    lst_coupl_matrix = [lst_coupl_matrix[agent]] if agent is not None else lst_coupl_matrix 
    colors = [COLORS[agent]] if agent is not None else COLORS
    
    if lst_coupl_matrix is not None:
        for p, coupl_matrix in enumerate(lst_coupl_matrix):
            ax.plot(0,0,colors[p],label = "Cost of agent "+str(p)+": "+ '%.2f' % prices[p])
            for i in range(len(coupl_matrix)):
                for j in range(len(coupl_matrix[i])):
                    if coupl_matrix[i][j]:
                        ax.plot([X[i][0], Y[j][0]], [X[i][1], Y[j][1]],
                                colors[p], lw=coupl_matrix[i][j]*30, zorder=1)

    ax.scatter(X[:, 0], X[:, 1], 100, 'b', zorder=2)
    ax.scatter(Y[:, 0], Y[:, 1], 100, 'r', marker='s', zorder=2)
    ax.axis('off')
   
    plt.legend()
    plt.show()
    
def plot_accuracy_algorithm(C, a, b) :

    reg_m = 5 * 1e-2
    res_m, acc_m, lam_m, alpha_m, beta_m, denom_m, KC_m, K_trans_m = eot.EOT_PSinkhorn(C, reg_m, a, b)

    reg_mm = 5* 1e-1
    res_mm, acc_mm, lam_mm, alpha_mm, beta_mm, denom_mm, KC_mm, K_trans_mm = eot.EOT_PSinkhorn(C, reg_mm, a, b)

    reg = 5 * 1e-3
    res, acc, lam, alpha, beta, denom, KC, K_trans = eot.EOT_PSinkhorn(C, reg, a, b)

    res_lp = eot.LP_solver_Primal(C, a, b)

    plt.plot([res_lp["fun"]]*len(acc), c = "red", label = "LP solver")
    plt.plot(acc,c="blue",label = "eps = 0.005")
    plt.plot(acc_m,c="turquoise",label = "eps = 0.05")
    plt.plot(acc_mm,c="green",label = "eps = 0.5")
    plt.legend()
    plt.show()
    return()