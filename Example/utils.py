import numpy as np
from scipy import stats as sts
import matplotlib.pyplot as plt

# Function Definition
def RMSE(T_true, T_est):
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1)))**2))


def generate_data(N, rng, similar=True):
    # Set seed
    np.random.seed(rng)

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, N)

    # Generate A
    und_lin = 0.2 + 1*X
    pscore = sts.norm.cdf(und_lin)
    A = sts.binom.rvs(1, pscore)

    if similar:

        # Generate Y
        Y_0 = 2 + 0.3*np.exp(X)
        Y_1 = 3 + Y_0
        ITE = Y_1 - Y_0

    elif not similar:
        # Generate Y
        Y_0 = 2 + 0.3*X
        Y_1 = 5 + 0.3*np.exp(X)
        ITE = Y_1 - Y_0

    sigma_Y = 0.75
    Y = Y_0 + ITE*A + sts.norm.rvs(0, sigma_Y, N)

    return X, A, Y, ITE


def plot_confound(X, A, Y):

  fig, axs = plt.subplots(1, 2, figsize=(14, 6))

  axs[0].scatter(X[A==1], Y[A==1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
  axs[0].scatter(X[A==0], Y[A==0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
  axs[0].set_xlabel('$X$'); axs[0].set_ylabel('Outcome $Y$')
  axs[0].legend(facecolor='white', fontsize=12)

  plt.show()
