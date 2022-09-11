# Importing packages
from Example.utils import *
import matplotlib.pyplot as plt
from copy import deepcopy

import GPy
import os

# Options
plt.rcParams["figure.figsize"] = (12, 6)

# Case 1: Similar surfaces
N = 300
X, A, Y, ITE = generate_data(N, 234, similar=True)

##### Single GP
kernel = GPy.kern.RBF(1)

X_r = X.reshape(N, 1)
Y_r = Y.reshape(N, 1)

m = {'m_Y_1': GPy.models.GPRegression(X_r[A == 1], Y_r[A == 1],
                                      deepcopy(kernel), normalizer=True),
     'm_Y_0': GPy.models.GPRegression(X_r[A == 0], Y_r[A == 0],
                                      deepcopy(kernel), normalizer=True)}

# Log-lik Optimization
for j in np.unique(A):
    m['m_Y_%i' % j].optimize('bfgs')


##### Multitask GP
# base_kernel = GPy.kern.RBF(input_dim=1)
# kern = GPy.util.multioutput.ICM(1, 2, base_kernel, W_rank=2, W=None, kappa=None,name='ICM')

K0 = GPy.kern.RBF(1)
K1 = GPy.kern.RBF(1)
kern = GPy.util.multioutput.LCM(input_dim=1, num_outputs=2, kernels_list=[K0, K1])

X_r = np.c_[X, A]
Y_r = Y.reshape(N, 1)

mod = GPy.models.GPRegression(X_r, Y_r, deepcopy(kern), normalizer=True)

# Log-lik Optimization
mod.optimize('bfgs')

#
# Case 2: different surfaces
X_dif, A_dif, Y_dif, ITE_dif = generate_data(N, 234, similar=False)

##### Single GP
kernel = GPy.kern.RBF(1)

X_dif_r = X_dif.reshape(N, 1)
Y_dif_r = Y_dif.reshape(N, 1)

m_dif = {'m_Y_dif_1': GPy.models.GPRegression(X_dif_r[A_dif == 1], Y_dif_r[A_dif == 1],
                                          deepcopy(kernel), normalizer=True),
         'm_Y_dif_0': GPy.models.GPRegression(X_dif_r[A_dif == 0], Y_dif_r[A_dif == 0],
                                          deepcopy(kernel), normalizer=True)}

# Log-lik Optimization
for j in np.unique(A_dif):
    m_dif['m_Y_dif_%i' % j].optimize('bfgs')


##### Multitask GP
# base_kernel = GPy.kern.RBF(input_dim=1)
# kern = GPy.util.multioutput.ICM(1, 2, base_kernel, W_rank=2, W=None, kappa=None,name='ICM')

K0 = GPy.kern.RBF(1)
K1 = GPy.kern.RBF(1)
kern = GPy.util.multioutput.LCM(input_dim=1, num_outputs=2, kernels_list=[K0, K1])

X_dif_r = np.c_[X_dif, A_dif]
Y_dif_r = Y_dif.reshape(N, 1)

mod_dif = GPy.models.GPRegression(X_dif_r, Y_dif_r, deepcopy(kern), normalizer=True)

# Log-lik Optimization
mod_dif.optimize('bfgs')


# Plot together
pts = np.linspace(-3., 3., 100).reshape(100, 1)
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
fig.tight_layout()

axs[0, 0].scatter(X[A==1], Y[A==1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
axs[0, 0].scatter(X[A==0], Y[A==0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
axs[0, 0].plot(pts, np.array(m['m_Y_1'].predict(pts))[0, :, :], c='black', label="Fit")
axs[0, 0].fill_between(pts[:, 0],
                    (np.array(m['m_Y_1'].predict(pts))[0, :, :] - 2*np.sqrt(np.array(m['m_Y_1'].predict(pts))[1, :, :]))[:,0],
                    (np.array(m['m_Y_1'].predict(pts))[0, :, :] + 2*np.sqrt(np.array(m['m_Y_1'].predict(pts))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[0, 0].plot(pts, 5 + 0.3*np.exp(pts), c=(0.2, 0.3, 1), linestyle='--')
axs[0, 0].plot(pts, np.array(m['m_Y_0'].predict(pts))[0, :, :], c='black')
axs[0, 0].fill_between(pts[:, 0],
                    (np.array(m['m_Y_0'].predict(pts))[0, :, :] - 2*np.sqrt(np.array(m['m_Y_0'].predict(pts))[1, :, :]))[:,0],
                    (np.array(m['m_Y_0'].predict(pts))[0, :, :] + 2*np.sqrt(np.array(m['m_Y_0'].predict(pts))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[0, 0].plot(pts, 2 + 0.3*np.exp(pts), c=(1, 0.2, 0.1), linestyle='--')
axs[0, 0].set_xlabel('$X$'); axs[0, 0].set_ylabel('Outcome $Y$')
axs[0, 0].legend(facecolor='white', fontsize=12, loc='upper left')
axs[0, 0].grid()
axs[0, 0].set_title('GP')

axs[0, 1].scatter(X[A == 1], Y[A == 1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
axs[0, 1].scatter(X[A == 0], Y[A == 0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
axs[0, 1].plot(pts, np.array(mod.predict(np.c_[pts, np.ones(100)]))[0, :, :], c='black', label="Fit")
axs[0, 1].fill_between(pts[:, 0],
                    (np.array(mod.predict(np.c_[pts, np.ones(100)]))[0, :, :] -
                     2*np.sqrt(np.array(mod.predict(np.c_[pts, np.ones(100)]))[1, :, :]))[:,0],
                       (np.array(mod.predict(np.c_[pts, np.ones(100)]))[0, :, :] +
                        2 * np.sqrt(np.array(mod.predict(np.c_[pts, np.ones(100)]))[1, :, :]))[:, 0],
                       alpha=0.3, color='grey')
axs[0, 1].plot(pts, np.array(mod.predict(np.c_[pts, np.zeros(100)]))[0, :, :], c='black')
axs[0, 1].fill_between(pts[:, 0],
                    (np.array(mod.predict(np.c_[pts, np.zeros(100)]))[0, :, :] - 2*np.sqrt(np.array(mod.predict(np.c_[pts, np.zeros(100)]))[1, :, :]))[:,0],
                    (np.array(mod.predict(np.c_[pts, np.zeros(100)]))[0, :, :] + 2*np.sqrt(np.array(mod.predict(np.c_[pts, np.zeros(100)]))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[0, 1].plot(pts, 5 + 0.3*np.exp(pts), c=(0.2, 0.3, 1), linestyle='--')
axs[0, 1].plot(pts, 2 + 0.3*np.exp(pts), c=(1, 0.2, 0.1), linestyle='--')
axs[0, 1].set_xlabel('$X$'); axs[0, 1].set_ylabel('Outcome $Y$')
axs[0, 1].legend(facecolor='white', fontsize=12, loc='upper left')
axs[0, 1].grid()
axs[0, 1].set_title('Multitask GP')


axs[1, 0].scatter(X_dif[A_dif==1], Y_dif[A_dif==1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
axs[1, 0].scatter(X_dif[A_dif==0], Y_dif[A_dif==0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
axs[1, 0].plot(pts, np.array(m_dif['m_Y_dif_1'].predict(pts))[0, :, :], c='black', label="Fit")
axs[1, 0].fill_between(pts[:, 0],
                    (np.array(m_dif['m_Y_dif_1'].predict(pts))[0, :, :] - 2*np.sqrt(np.array(m_dif['m_Y_dif_1'].predict(pts))[1, :, :]))[:,0],
                    (np.array(m_dif['m_Y_dif_1'].predict(pts))[0, :, :] + 2*np.sqrt(np.array(m_dif['m_Y_dif_1'].predict(pts))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[1, 0].plot(pts, 5 + 0.3*np.exp(pts), c=(0.2, 0.3, 1), linestyle='--')
axs[1, 0].plot(pts, np.array(m_dif['m_Y_dif_0'].predict(pts))[0, :, :], c='black')
axs[1, 0].fill_between(pts[:, 0],
                    (np.array(m_dif['m_Y_dif_0'].predict(pts))[0, :, :] - 2*np.sqrt(np.array(m_dif['m_Y_dif_0'].predict(pts))[1, :, :]))[:,0],
                    (np.array(m_dif['m_Y_dif_0'].predict(pts))[0, :, :] + 2*np.sqrt(np.array(m_dif['m_Y_dif_0'].predict(pts))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[1, 0].plot(pts, 2 + 0.3*pts, c=(1, 0.2, 0.1), linestyle='--')
axs[1, 0].set_xlabel('$X$'); axs[1, 0].set_ylabel('Outcome $Y$')
axs[1, 0].legend(facecolor='white', fontsize=12, loc='upper left')
axs[1, 0].grid()


axs[1, 1].scatter(X_dif[A_dif == 1], Y_dif[A_dif == 1], facecolor=(0.2, 0.3, 1), alpha=0.5, label='Treated')
axs[1, 1].scatter(X_dif[A_dif == 0], Y_dif[A_dif == 0], facecolor=(1, 0.2, 0.1), alpha=0.5, label='Control')
axs[1, 1].plot(pts, np.array(mod_dif.predict(np.c_[pts, np.ones(100)]))[0, :, :], c='black', label="Fit")
axs[1, 1].fill_between(pts[:, 0],
                    (np.array(mod_dif.predict(np.c_[pts, np.ones(100)]))[0, :, :] -
                     2*np.sqrt(np.array(mod_dif.predict(np.c_[pts, np.ones(100)]))[1, :, :]))[:,0],
                       (np.array(mod_dif.predict(np.c_[pts, np.ones(100)]))[0, :, :] +
                        2 * np.sqrt(np.array(mod_dif.predict(np.c_[pts, np.ones(100)]))[1, :, :]))[:, 0],
                       alpha=0.3, color='grey')
axs[1, 1].plot(pts, np.array(mod_dif.predict(np.c_[pts, np.zeros(100)]))[0, :, :], c='black')
axs[1, 1].fill_between(pts[:, 0],
                    (np.array(mod_dif.predict(np.c_[pts, np.zeros(100)]))[0, :, :] - 2*np.sqrt(np.array(mod_dif.predict(np.c_[pts, np.zeros(100)]))[1, :, :]))[:,0],
                    (np.array(mod_dif.predict(np.c_[pts, np.zeros(100)]))[0, :, :] + 2*np.sqrt(np.array(mod_dif.predict(np.c_[pts, np.zeros(100)]))[1, :, :]))[:,0],
                    alpha=0.3, color='grey')
axs[1, 1].plot(pts, 5 + 0.3*np.exp(pts), c=(0.2, 0.3, 1), linestyle='--')
axs[1, 1].plot(pts, 2 + 0.3*pts, c=(1, 0.2, 0.1), linestyle='--')
axs[1, 1].set_xlabel('$X$'); axs[1, 1].set_ylabel('Outcome $Y$')
axs[1, 1].legend(facecolor='white', fontsize=12, loc='upper left')
axs[1, 1].grid()

fig.savefig('./Example/Example_Overlap.pdf', dpi=400)
