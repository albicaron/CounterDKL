import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sts
import torch

import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, fetch_openml
import numpy as np

# Function Definition
def bias(T_true, T_est):
    return np.mean(T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))


def RMSE(T_true, T_est):
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))**2))


def MAE(T_true, T_est):
    return np.mean(np.abs(T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


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
    torch.manual_seed(rng)
    torch.cuda.manual_seed(rng)
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


def get_data(dataset: str = None, scale: bool = True) -> tuple:
    """Get data (features and labels) used in experiments.

    Parameters
    ----------

    dataset : str, default: None
        It should be one of: 'ecoli', 'glass', 'letter-recognition',
        'lymphography', 'yeast', 'digits', 'breast-cancer', 'wine', or
        'mnist'.

    scale : bool, default: True
        Standardize features by zero mean and unit variance.

    Returns
    -------

    tuple, length=2
        tuple containing features-target split of inputs.
    References
    ----------
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science.

    Examples
    --------
    >>> X, y = get_data(dataset='ecoli')
    >>> X[0,:]
    array([0.49, 0.29, 0.48, 0.5 , 0.56, 0.24, 0.35])

    """

    if dataset not in ['australian', 'ecoli', 'glass', 'letter-recognition', 'heart', 'yeast',
                       'digits', 'breast-cancer', 'wine', 'mnist', 'cmc', 'tae']:
        raise ValueError("Invalid dataset provided.")

    if dataset in dataset in ['ecoli', 'glass', 'cmc', 'tae',
                              'letter-recognition', 'yeast']:
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
        f = path + dataset + "/" + dataset + ".data"
    elif dataset == 'australian':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/'
        f = path + dataset + "/" + dataset + ".dat"
    elif dataset == 'heart':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/'
        f = path + dataset + "/" + dataset + ".dat"

    if dataset in ['australian', 'heart', 'ecoli', 'yeast']:
        df = pd.read_table(f, delim_whitespace=True, header=None)
    elif dataset in ['tae', 'cmc', 'glass', 'letter-recognition', 'lymphography']:
        df = pd.read_csv(f, header=None)
    elif dataset == 'digits':
        df = load_digits()
        X = df.data
        y = df.target
    elif dataset == 'breast-cancer':
        df = load_breast_cancer()
        X = df.data
        y = df.target
    elif dataset == 'wine':
        df = load_wine()
        X = df.data
        y = df.target

    if dataset == 'ecoli':
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 1:8].values

    elif dataset == 'glass':
        y = df.iloc[:, -1].values
        X = df.iloc[:, 1:(df.shape[1] - 1)].values

    elif dataset == 'australian':
        y = df.iloc[:, -1].values
        X = df.iloc[:, 0:(df.shape[1] - 1)].values

    elif dataset == 'cmc':
        y = df.iloc[:, -1].values
        X = df.iloc[:, 0:(df.shape[1] - 1)].values

    elif dataset == 'tae':
        y = df.iloc[:, -1].values
        X = df.iloc[:, 0:(df.shape[1] - 1)].values

    elif dataset == 'heart':
        y = df.iloc[:, -1].values
        X = df.iloc[:, 0:(df.shape[1] - 1)].values

    elif dataset in ['letter-recognition', 'lymphography']:
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, 0])
        X = df.iloc[:, 1:(df.shape[1])].values

    elif dataset == 'yeast':
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 1:9].values

    elif dataset == 'mnist':
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype('int64')

    if scale == True:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

    return X, y


def gen_Y(X, A):
    beta = [np.random.choice([0.4, 0.2, 0], X.shape[1], replace=True, p=[0.6, 0.25, 0.15]) for _ in
            range(len(np.unique(A)))]

    Y_true = np.array([np.exp(X @ beta[i]) for i in np.unique(A)]).transpose()

    A_onehot = np.zeros((A.size, A.max() + 1))
    A_onehot[np.arange(A.size), A] = 1

    Y = np.sum((Y_true + sts.norm.rvs(0, 0.5, (A.shape[0], len(np.unique(A))))) * A_onehot, axis=1)

    return Y_true, Y


def get_pol_val(Y_true, A):
    gen_pol = np.random.choice(np.unique(A), A.shape[0])
    random_pol = np.zeros((gen_pol.size, gen_pol.max() + 1))
    random_pol[np.arange(gen_pol.size), gen_pol] = 1

    pol_val = np.mean(Y_true * random_pol)

    return pol_val, random_pol
