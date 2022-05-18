import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sts


# Function Definition
def bias(T_true, T_est):
    return np.mean(T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))


def MAE(T_true, T_est):
    return np.abs(T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.nanstd(np.array(x)) / np.sqrt(B)


# def coverage(T_true, T_est):
#     quan = T_est
#
#
# np.quantile(X, [0.025, 0.975])
#
# coverage_95 < - function(ITE_EST, ITE)
# {
#     quan = apply(ITE_EST, 2, function(x)
# quantile(x, c(0.025, 0.975)))
# cove = sum((quan[1,] < ITE) & (ITE < quan[2,])) / length(ITE)
#
# return (cove)
# }


# Train-Test Split
def tr_te_split(X, split):
    train = np.array(X[split])
    test = np.array(X[~split])

    return train, test


def gen_1d_data(N, P):
    # Set seed
    np.random.seed(123)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))

    # Generate A
    und_lin = -0.2 + 0.8*X[:, 0]
    pscore = sts.norm.cdf(und_lin)
    A = sts.binom.rvs(1, pscore)

    # Generate Y
    Y_0 = 2 - 0.1*X[:, 0]**2
    Y_1 = 4 + 0.2*np.exp(X[:, 0]) + 0.2*np.sin(X[:, 0])
    ITE = Y_1 - Y_0

    # Generate C
    C_0 = 2 - 0.1*X[:, 0]**2
    C_1 = 3 + 0.2*np.exp(X[:, 0])
    ICE = C_1 - C_0

    sigma_Y = np.std(ITE)/4
    sigma_C = np.std(ICE)/4
    Y = Y_0 + ITE*A + sts.norm.rvs(0, sigma_Y, N)
    C = C_0 + ICE*A + sts.norm.rvs(0, sigma_C, N)

    return X, A, Y, C, ITE, ICE


def backdoor_dgp(N=1000, P=10, rng=1):

    # Set seed
    np.random.seed(rng)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))
    X_ones = np.c_[np.ones(N), X]

    # Generate A
    coef1 = np.zeros(P+1)
    coef2 = np.zeros(P+1)
    coef3 = np.zeros(P+1)
    coef4 = np.zeros(P+1)

    coef4[range(4)] = np.array([2, 1.5, 0.5, 0.2])
    coef3[range(4)] = np.array([1.5, 0.6, 0.3, 0.3])
    coef2[range(2, 5)] = np.array([1, 0.8, 0.2])
    coef1[range(4)] = np.array([-1, -0.8, -0.1, -0.1])

    und_lin = np.c_[np.exp(X_ones @ coef1), np.exp(X_ones @ coef2),
                    np.exp(X_ones @ coef3), np.exp(X_ones @ coef4)]

    for i in range(N):
        und_lin[i, :] /= und_lin[i, :].sum(axis=0)

    np.random.multinomial(1, und_lin[0, :])

    A = np.array([np.random.multinomial(1, und_lin[i, :]) for i in range(N)])

    # # Generate Y
    # Y_true = np.zeros((N, 4))
    # Y_true[:, 0] = 3 + 0.2 * X[:, 0] + 0.1 * np.exp(X[:, 3])
    # Y_true[:, 1] = 2 + 0.5 * X[:, 0] - 0.2 * X[:, 1] ** 2
    # Y_true[:, 2] = 0.5 + 0.2 * X[:, 0] + 0.3 * np.exp(X[:, 3])
    # Y_true[:, 3] = 3 + 0.1 * X[:, 0] - 0.1 * X[:, 2] ** 2
    #
    # # Generate C
    # C_true = np.zeros((N, 4))
    # C_true[:, 0] = 1 + 0.1 * X[:, 0] + 0.1 * np.exp(X[:, 3])
    # C_true[:, 1] = 1 + 0.8 * X[:, 0] - 0.3 * X[:, 1] ** 2
    # C_true[:, 2] = 0.5 + 0.1 * X[:, 0] + 0.2 * np.exp(X[:, 3])
    # C_true[:, 3] = 2 + 0.3 * X[:, 1]

        ##################### ALTERNATIVE DGP TO TRY IN CASE
    # Generate Y
    Y_true = np.zeros((N, 4))
    Y_true[:, 0] = 3 + 0.4 * X[:, 0]*X[:, 1] - 0.3 * X[:, 2] ** 2 + 0.2 * np.exp(X[:, 3]) + 0.6 * np.sin(X[:, 4])
    Y_true[:, 1] = -1 + Y_true[:, 0] + 0.1 * X[:, 5]
    Y_true[:, 2] = 1 + Y_true[:, 0] + 0.3 * X[:, 5]
    Y_true[:, 3] = 0.5 + Y_true[:, 0] + 0.5 * X[:, 6]

    # Generate C
    C_true = np.zeros((N, 4))
    C_true[:, 0] = 1 + 0.2 * X[:, 0]*X[:, 1] - 0.2 * X[:, 2] ** 2 + 0.1 * np.exp(X[:, 3])
    C_true[:, 1] = -2 + C_true[:, 0] + 0.2 * X[:, 5]
    C_true[:, 2] = 2 + C_true[:, 0] + 0.4 * X[:, 5]
    C_true[:, 3] = 1 + C_true[:, 0] + 0.5 * X[:, 6]

    sigma_Y = 0.5
    sigma_C = 0.5
    Y_obs = (Y_true + sts.norm.rvs(0, sigma_Y, (N, 4))) * A
    C_obs = (C_true + sts.norm.rvs(0, sigma_C, (N, 4))) * A

    Y = Y_obs[Y_obs != 0]
    C = C_obs[C_obs != 0]

    A_ = np.zeros(N)

    for i in range(4):
        A_[A[:, i] == 1] = i

    return X, A_, Y, C, Y_true, C_true


def frontdoor_dgp(N=500, P=5, rng=1):
    # Set seed
    np.random.seed(rng)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))
    U = np.random.uniform(-3., 3, N)
    X_ones = np.c_[np.ones(N), X, U]

    # Generate A
    coef1 = np.array([-1.2, 0.4, 0.4, 0, 0, 0, 0.3])
    und_lin = X_ones @ coef1
    pscore = sts.logistic.cdf(und_lin)

    A = sts.binom.rvs(1, pscore)

    # Generate Z
    X_Z = np.c_[np.ones(N), X, A]
    coef_Z = np.array([-1.2, 0.4, 0, 0, 0, 0, 0.7])
    und_lin_Z = X_Z @ coef_Z

    pscore_Z = sts.logistic.cdf(und_lin_Z)
    Z = sts.binom.rvs(1, pscore_Z)

    tilt = np.mean(sts.logistic.cdf(und_lin_Z)) - np.mean(sts.logistic.cdf(X_Z @ np.array([-1.2, 0.4, 0, 0, 0, 0, 0])))

    # Generate Y
    tau_Y = 0.2 + 0.4*X[:, 0]**2 - 0.2*X[:, 1]
    Y_true = 3 + 0.3*np.exp(X[:, 1]) + 0.4*U + tau_Y*Z

    # Generate C
    tau_C = 0.2 + 0.3*X[:, 0]**2 - 0.1*X[:, 1]
    C_true = 1 + 0.2*np.exp(X[:, 1]) + 0.4*U + tau_C*Z

    sigma_Y = 0.25
    sigma_C = 0.25
    Y = (Y_true + sts.norm.rvs(0, sigma_Y, N))
    C = (C_true + sts.norm.rvs(0, sigma_C, N))

    CATE_Y = tau_Y * tilt
    CATE_C = tau_C * tilt

    return X, A, Z, Y, C, CATE_Y, CATE_C, tilt


def instrument_dgp(N=500, P=5, rng=1):
    # Set seed
    np.random.seed(rng)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))
    U = np.random.uniform(-3., 3, N)

    # Generate Z
    X_Z = np.c_[np.ones(N), X]
    coef_Z = np.array([-1, 0.5, 0, 0, 0, 0])
    und_lin_Z = X_Z @ coef_Z

    pscore_Z = sts.logistic.cdf(und_lin_Z)
    Z = sts.binom.rvs(1, pscore_Z)

    # Generate A
    X_ones = np.c_[np.ones(N), X, U, Z]
    coef1 = np.array([-1, 0.4, 0, 0, 0, 0, 0.3, 1])
    und_lin = X_ones @ coef1
    pscore = sts.logistic.cdf(und_lin)

    A = sts.binom.rvs(1, pscore)

    tilt = np.mean(sts.logistic.cdf(und_lin)) - np.mean(sts.logistic.cdf(X_ones @ np.array([-1, 0.4, 0, 0, 0, 0, 0.3, 0])))

    ATE_Y = ATE_C = 1.0

    # Generate Y
    Y_true = 3 + 0.2*np.exp(X[:, 1]) - 0.5*X[:, 0] ** 2 + 0.4*U + ATE_Y*A

    # Generate C
    C_true = 3 + 0.1*np.exp(X[:, 1]) - 0.4 * X[:, 0] ** 2 + 0.4*U + ATE_C*A

    sigma_Y = 0.25
    sigma_C = 0.25
    Y = (Y_true + sts.norm.rvs(0, sigma_Y, N))
    C = (C_true + sts.norm.rvs(0, sigma_C, N))

    # ATE_Y = (np.mean(Y_true[Z == 1]) - np.mean(Y_true[Z == 0])) / tilt
    # ATE_C = (np.mean(C_true[Z == 1]) - np.mean(C_true[Z == 0])) / tilt

    return X, A, Z, Y, C, ATE_Y, ATE_C, tilt


def simulated_study_2(N=500, P=5, rng=1):
    # Set seed
    np.random.seed(rng)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))
    X_ones = np.c_[np.ones(N), X]

    # Generate A
    coef1 = np.array([-0.1, 0.7, 0, 0, 0, 0])
    coef2 = np.array([-0.3, 0, 0.1, 0.2, 0, 0])

    und_lin = np.c_[np.exp(X_ones @ coef1), np.exp(X_ones @ coef2)]

    for i in range(N):
        und_lin[i, :] /= und_lin[i, :].sum(axis=0)

    np.random.multinomial(1, und_lin[0, :])

    A = np.array([np.random.multinomial(1, und_lin[i, :]) for i in range(N)])

    # Generate Y
    Y_true = np.zeros((N, 2))
    Y_true[:, 0] = 2 - 0.1 * X[:, 0] ** 2
    Y_true[:, 1] = 0.5 + 0.2 * np.exp(X[:, 2])

    # Generate C
    C_true = np.zeros((N, 2))
    C_true[:, 0] = 2 - 0.2 * X[:, 0] ** 2
    C_true[:, 1] = 0.5 + 0.3 * np.exp(X[:, 2])

    sigma_Y = 0.25
    sigma_C = 0.25
    Y_obs = (Y_true + sts.norm.rvs(0, sigma_Y, (N, 2))) * A
    C_obs = (C_true + sts.norm.rvs(0, sigma_C, (N, 2))) * A

    Y = Y_obs[Y_obs != 0]
    C = C_obs[C_obs != 0]

    A_ = np.zeros(N)

    for i in range(2):
        A_[A[:, i] == 1] = i

    return X, A_, Y, C, Y_true, C_true

def plot_confound(X, A, Y, C):

  fig, axs = plt.subplots(1, 2, figsize=(14, 6))

  axs[0].scatter(X[:, 0][A==1], Y[A==1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
  axs[0].scatter(X[:, 0][A==0], Y[A==0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
  axs[0].set_xlabel('$X_1$'); axs[0].set_ylabel('Outcome $Y$')
  axs[0].legend(facecolor='white', fontsize=12)

  axs[1].scatter(X[:, 0][A==1], C[A==1], facecolor=(0.1, 0, 0.8), alpha=0.5, label='Treated')
  axs[1].scatter(X[:, 0][A==0], C[A==0], facecolor=(0.8, 0.1, 0), alpha=0.5, label='Control')
  axs[1].set_xlabel('$X_1$'); axs[1].set_ylabel('Cost $C$')
  axs[1].legend(facecolor='white', fontsize=12)
  plt.show()

  return axs


