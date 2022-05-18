import GPy
import numpy as np
from copy import deepcopy

class SimpleGPreg:
    """
    Single-output Gaussian Process regression
    """
    def __init__(self, input_dim, kern='RBF', sparse=False):
        self.input_dim = input_dim
        self.sparse = sparse

        if kern == 'RBF':
            self.kern = GPy.kern.RBF(input_dim=self.input_dim, variance=1., lengthscale=1., ARD=True)
        elif kern == 'Matern':
            self.kern = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)

    def fit(self, X, A, Y):
        """
        Optimize model's parameters

        :X: Matrix of Covariates
        :A: Matrix of Actions (one-hot encoding form)
        :Y: Matrix of Outcomes
        """
        try:
            self.n_out = Y.shape[1]
        except IndexError:
            self.n_out = 1

        self.mod = dict()
        self.n_causes = np.unique(A)

        if self.sparse is False:
            for j in range(self.n_out):
                for i in self.n_causes:
                    m = {'m_Y%s_A%s' % (j, i): GPy.models.GPRegression(X.reshape(X.shape[0], self.input_dim)[A == i],
                                                                       Y[:, j].reshape(X.shape[0], 1)[A == i],
                                                                       deepcopy(self.kern))}
                    self.mod.update(m)

        elif self.sparse is True:
            for j in range(self.n_out):
                for i in self.n_causes:
                    m = {'m_Y%s_A%s' % (j, i): GPy.models.SparseGPRegression(X.reshape(X.shape[0], self.input_dim)[A == i],
                                                                             Y[:, j].reshape(X.shape[0], 1)[A == i],
                                                                             deepcopy(self.kern))}
                    self.mod.update(m)

        # Log-likelihood maximization:
        for j in range(self.n_out):
            for i in self.n_causes:
                self.mod['m_Y%s_A%s' % (j, i)].optimize('lbfgs', max_iters=1000)

    def predict(self, X):
        """
        Predict out of sample

        :X: test set matrix of covariates
        """
        counter_Y = list()
        for j in range(self.n_out):
            for i in self.n_causes:

                pred = np.array(self.mod['m_Y%s_A%s' % (j, i)].predict(X.reshape(X.shape[0], self.input_dim)))[0, :, :]
                counter_Y.append(pred)

        return np.array(counter_Y)


class CounterGP:
    """
    Multi-task Gaussian Process
    """
    def __init__(self, input_dim, kern='RBF', kern_type='LCM', num_actions=2, sparse=False):
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.sparse = sparse

        if kern_type is 'LCM':
            if kern is 'RBF':
                self.K0 = GPy.kern.RBF(self.input_dim, ARD=True)
                self.K1 = GPy.kern.RBF(self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.LCM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernels_list=[self.K0, self.K1])
            elif kern is 'Matern':
                self.K0 = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.K1 = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.LCM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernels_list=[self.K0, self.K1])

        elif kern_type is 'ICM':
            if kern is 'RBF':
                self.base_kernel = GPy.kern.RBF(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.ICM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernel=self.base_kernel, W_rank=1)
            elif kern is 'Matern':
                self.base_kernel = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.ICM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernel=self.base_kernel, W_rank=1)

    def fit(self, X, A, Y):
        """
        Optimize model's parameters

        :X: Matrix of Covariates
        :A: Matrix of Actions (one-hot encoding form)
        :Y: Matrix of Outcomes
        """
        try:
            self.n_out = Y.shape[1]
        except IndexError:
            self.n_out = 1

        self.n_causes = np.unique(A)

        if self.sparse is False:
            self.mod = dict()
            for j in range(self.n_out):
                X_list = list()
                Y_list = list()
                for i in self.n_causes:
                    X_list.append(X.reshape(X.shape[0], self.input_dim)[A == i, :])
                    Y_list.append(Y.reshape(X.shape[0], self.n_out)[A == i, j].reshape(-1, 1))

                m = {'m_Y%s' % j: GPy.models.GPCoregionalizedRegression(X_list=X_list,
                                                                        Y_list=Y_list,
                                                                        kernel=deepcopy(self.kern))}
                self.mod.update(m)

        elif self.sparse is True:
            self.mod = dict()
            for j in range(self.n_out):
                X_list = list()
                Y_list = list()
                for i in self.n_causes:
                    X_list.append(X.reshape(X.shape[0], self.input_dim)[A == i, :])
                    Y_list.append(Y.reshape(X.shape[0], self.n_out)[A == i, j].reshape(-1, 1))

                m = {'m_Y%s' % j: GPy.models.SparseGPCoregionalizedRegression(X_list=X_list,
                                                                              Y_list=Y_list,
                                                                              kernel=deepcopy(self.kern))}
                self.mod.update(m)

        # Log-likelihood maximization:
        for j in range(self.n_out):
            self.mod['m_Y%s' % j].optimize('lbfgs', max_iters=1000)

    def predict(self, X):
        """
        Predict out of sample

        :X: test set matrix of covariates
        """
        counter_Y = list()

        for j in range(self.n_out):
            for i in self.n_causes:
                X_augm = np.hstack([X, i*np.ones((X.shape[0], 1))])
                noise_dict = {'output_index': (i*np.ones((X.shape[0], 1))).astype(int)}

                pred = np.array(self.mod['m_Y%s' % j].predict(X_augm, Y_metadata=noise_dict)[0])
                counter_Y.append(pred)

        return np.array(counter_Y)


class CounterMOGP:
    """
    Multioutput - Multi-task Gaussian Process
    """
    def __init__(self, input_dim, kern='RBF', num_out=2, num_actions=2, sparse=False):
        self.num_out = num_out
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.sparse = sparse

        # kern_1 = GPy.kern.Bias(input_dim=self.input_dim)
        if kern is 'RBF':
            kern_2 = GPy.kern.RBF(input_dim=self.input_dim, ARD=True)
        elif kern is 'Matern':
            kern_2 = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)

        self.kern = ((kern_2)
                     * GPy.kern.Coregionalize(input_dim=1, output_dim=self.num_actions,
                                              active_dims=self.input_dim, rank=1)
                     * GPy.kern.Coregionalize(input_dim=1, output_dim=self.num_out,
                                              active_dims=self.input_dim + 1, rank=1)
                     )

    def fit(self, X, A, Y):
        """
        Optimize model's parameters

        :X: Matrix of Covariates
        :A: Matrix of Actions (one-hot encoding form)
        :Y: Matrix of Outcomes
        """
        try:
            self.n_out = Y.shape[1]
        except IndexError:
            self.n_out = 1

        self.n_causes = np.unique(A)

        X_augm = np.c_[X, A]
        Y_stack = Y.flatten('F')
        X_stack = np.tile(X_augm, (self.n_out, 1))
        input = np.c_[X_stack, np.tile(range(self.n_out), (X.shape[0], 1)).flatten('F')]

        # Log-likelihood maximization:
        if self.sparse is False:
            self.mod = GPy.models.GPRegression(input, Y_stack.reshape(-1, 1), self.kern)
            self.mod.optimize('lbfgs', max_iters=2000)

        elif self.sparse is True:
            self.mod = GPy.models.GPRegression(input, Y_stack.reshape(-1, 1), self.kern)
            self.mod.optimize('lbfgs', max_iters=2000)

    def predict(self, X):
        """
        Predict out of sample

        :X: test set matrix of covariates
        """
        counter_Y = list()

        for j in range(self.n_out):
            for i in self.n_causes:
                X_augm = np.hstack([X, i*np.ones((X.shape[0], 1)), j*np.ones((X.shape[0], 1))])

                pred = np.array(self.mod.predict(X_augm))[0, :, :]
                counter_Y.append(pred)

        return np.array(counter_Y)



class BlockCounterGP:
    """
    Blocked Multi-task Gaussian Process
    """
    def __init__(self, input_dim, kern='RBF', kern_type='LCM', num_actions=2, block_list=None, sparse=False):
        self.block_list = block_list  # Only same length blocks supported for now
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.sparse = sparse

        if kern_type is 'LCM':
            if kern is 'RBF':
                self.K0 = GPy.kern.RBF(self.input_dim, ARD=True)
                self.K1 = GPy.kern.RBF(self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.LCM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernels_list=[self.K0, self.K1])
            elif kern is 'Matern':
                self.K0 = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.K1 = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.LCM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernels_list=[self.K0, self.K1])

        elif kern_type is 'ICM':
            if kern is 'RBF':
                self.base_kernel = GPy.kern.RBF(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.ICM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernel=self.base_kernel, W_rank=1)
            elif kern is 'Matern':
                self.base_kernel = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
                self.kern = GPy.util.multioutput.ICM(input_dim=self.input_dim, num_outputs=self.num_actions,
                                                     kernel=self.base_kernel, W_rank=1)

    def fit(self, X, A, Y):
        """
        Optimize model's parameters

        :X: Matrix of Covariates
        :A: Matrix of Actions (one-hot encoding form)
        :Y: Matrix of Outcomes
        """
        try:
            self.n_out = Y.shape[1]
        except IndexError:
            self.n_out = 1

        self.n_causes = np.unique(A)

        if self.sparse is False:
            self.mod = dict()
            for v in range(self.n_out):
                for j in self.block_list:
                    X_list = list()
                    Y_list = list()
                    for i in j:
                        X_list.append(X.reshape(X.shape[0], self.input_dim)[A == i, :])
                        Y_list.append(Y.reshape(X.shape[0], self.n_out)[A == i, v].reshape(-1, 1))

                    m = {'m_Y%s_%s' % (v + 1, ''.join(map(str, j))): GPy.models.GPCoregionalizedRegression(X_list=X_list,
                                                                                                           Y_list=Y_list,
                                                                                                           kernel=deepcopy(self.kern))}
                    self.mod.update(m)

        elif self.sparse is True:
            self.mod = dict()
            for v in range(self.n_out):
                for j in self.block_list:
                    X_list = list()
                    Y_list = list()
                    for i in j:
                        X_list.append(X.reshape(X.shape[0], self.input_dim)[A == i, :])
                        Y_list.append(Y.reshape(X.shape[0], self.n_out)[A == i, v].reshape(-1, 1))

                    m = {
                        'm_Y%s_%s' % (v + 1, ''.join(map(str, j))): GPy.models.SparseGPCoregionalizedRegression(X_list=X_list,
                                                                                                                Y_list=Y_list,
                                                                                                                kernel=deepcopy(self.kern))}
                    self.mod.update(m)

        # Log-likelihood maximization:
        for k in self.mod:
            self.mod[k].optimize('lbfgs', max_iters=1000)

    def predict(self, X):
        """
        Predict out of sample

        :X: test set matrix of covariates
        """
        counter_Y = list()

        for k in self.mod:
            for v in range(len(self.block_list[0])):        # Only same length blocks supported for now
                X_augm = np.hstack([X, v * np.ones((X.shape[0], 1))])
                noise_dict = {'output_index': (v * np.ones((X.shape[0], 1))).astype(int)}

                pred = np.array(self.mod[k].predict(X_augm, Y_metadata=noise_dict)[0])
                counter_Y.append(pred)

        return np.array(counter_Y)


class BlockCounterMOGP:
    """
    Blocked Multi-task Gaussian Process
    """
    def __init__(self, input_dim, kern='RBF', num_out=2, num_actions=2, block_list=None, sparse=False):
        self.block_list = block_list  # Only same length blocks supported for now
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.sparse = sparse
        self.kern = kern
        self.num_out = num_out

        self.mod = dict()
        for i in range(len(self.block_list)):
            m = {'m_Y%s' % (i + 1): CounterMOGP(input_dim=self.input_dim, kern=self.kern, num_out=self.num_out,
                                                num_actions=self.num_actions, sparse=self.sparse)}
            self.mod.update(m)

    def fit(self, X, A, Y):
        """
        Optimize model's parameters

        :X: Matrix of Covariates
        :A: Matrix of Actions (one-hot encoding form)
        :Y: Matrix of Outcomes
        """
        X_list = list()
        A_list = list()
        Y_list = list()
        for k in self.block_list:
            cond = np.isin(A, k)
            X_list.append(X[cond])
            Y_list.append(Y[cond])

            temp = np.zeros(A[cond].shape)
            for i, j in zip(k, range(len(k))):
                temp[A[cond] == i] = j

            A_list.append(temp)

        # Log-likelihood maximization:
        for i, k in zip(self.mod, range(len(self.block_list))):
            self.mod[i].fit(X=X_list[k], A=A_list[k], Y=Y_list[k])

    def predict(self, X):
        """
        Predict out of sample

        :X: test set matrix of covariates
        """
        counter_Y = list()
        for k in self.mod:
            counter_Y.append(self.mod[k].predict(X))

        return np.array(counter_Y)
