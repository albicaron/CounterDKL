import torch
import gpytorch
from copy import deepcopy


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, hidden_layers):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, hidden_layers[0]))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(hidden_layers[0], hidden_layers[1]))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_causes, input_dim):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=len(n_causes), rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class MultioutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_causes, n_out, input_dim):
        super(MultioutputGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

        # We learn two IndexKernel for 2 layered tasks
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=len(n_causes), rank=1)
        self.task_covar_module_Y = gpytorch.kernels.IndexKernel(num_tasks=n_out, rank=1)

    def forward(self, x, i, k):

        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        covar_k = self.task_covar_module_Y(k)
        # Multiply the two together to get the covariance we want
        cov = covar_x.mul(covar_k)
        covar = cov.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class DKLRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hidden_layers, feature_extractor):
        super(DKLRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=hidden_layers[1])),
            num_dims=hidden_layers[1], grid_size=100
        )
        self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskDKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hidden_layers, feature_extractor, n_causes):
        super(MultitaskDKLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=hidden_layers[1])),
            num_dims=hidden_layers[1], grid_size=100
        )
        self.feature_extractor = feature_extractor

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=len(n_causes), rank=1)

    def forward(self, x, i):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)

        # Get input-input covariance
        covar_x = self.covar_module(projected_x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class MultioutputDKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hidden_layers, feature_extractor, n_causes, n_out):
        super(MultioutputDKLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=hidden_layers[1])),
            num_dims=hidden_layers[1], grid_size=100
        )
        self.feature_extractor = feature_extractor

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        # We learn two IndexKernel for 2 layered tasks
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=len(n_causes), rank=1)
        self.task_covar_module_Y = gpytorch.kernels.IndexKernel(num_tasks=n_out, rank=1)

    def forward(self, x, i, k):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)

        # Get input-input covariance
        covar_x = self.covar_module(projected_x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        covar_k = self.task_covar_module_Y(k)
        # Multiply the two together to get the covariance we want
        cov = covar_x.mul(covar_k)
        covar = cov.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    #


class CounterGP:
    """
    Single-output GP object
    """

    def __init__(self, train_x, train_a, train_y, input_dim, GPtype='single', GPU=True):
        """
        Initialize GP model

        :input_dim: features X dimension
        :GPtype: type of learning paradigm ('single', 'multitask', 'multiout')
        """

        self.train_x = torch.from_numpy(train_x).float()
        self.train_a = torch.from_numpy(train_a).float()
        self.train_y = torch.from_numpy(train_y).float()
        self.GPU = GPU

        if torch.cuda.is_available() and self.GPU:
            self.train_x = self.train_x.cuda()
            self.train_a = self.train_a.cuda()
            self.train_y = self.train_y.cuda()

        self.data_dim = input_dim
        self.GPtype = GPtype
        self.n_causes = torch.unique(self.train_a)

        try:
            self.n_out = train_y.shape[1]
        except IndexError:
            self.n_out = 1


        if self.GPtype is 'single':

            # initialize likelihood and model
            self.likelihood = []
            self.models = dict()

            self.full_train_x = dict()
            self.full_train_y = dict()

            count = 0

            for j in range(self.n_out):
                for i in self.n_causes:
                    self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())

                    m = {'m_Y%s_A%s' % (j, i): GPRegressionModel(self.train_x[self.train_a == i],
                                                                 self.train_y[:, j][self.train_a == i],
                                                                 self.likelihood[count], self.data_dim)}

                    self.full_train_x.update({'m_Y%s_A%s' % (j, i): self.train_x[self.train_a == i]})
                    self.full_train_y.update({'m_Y%s_A%s' % (j, i): self.train_y[:, j][self.train_a == i]})

                    self.models.update(m)

                    count += 1

        if self.GPtype is 'multitask':

            # initialize likelihood and model
            self.models = dict()
            self.likelihood = []

            self.full_train_x = dict()
            self.full_train_i = dict()
            self.full_train_y = dict()

            for j in range(self.n_out):
                list_train_x = []
                list_train_i = []
                list_train_y = []
                for i in self.n_causes:
                    aux_a = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=i)

                    list_train_x.append(self.train_x[self.train_a == i])
                    list_train_i.append(aux_a)

                    try:
                        list_train_y.append(self.train_y[:, j][self.train_a == i])
                    except IndexError:
                        list_train_y.append(self.train_y[self.train_a == i])

                cat_train_x = torch.cat(list_train_x)
                cat_train_i = torch.cat(list_train_i)
                cat_train_y = torch.cat(list_train_y)

                if torch.cuda.is_available() and self.GPU:
                    cat_train_x = cat_train_x.cuda()
                    cat_train_i = cat_train_i.cuda()
                    cat_train_y = cat_train_y.cuda()

                self.full_train_x.update({'m_Y%s' % j: cat_train_x})
                self.full_train_i.update({'m_Y%s' % j: cat_train_i})
                self.full_train_y.update({'m_Y%s' % j: cat_train_y})

                self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())

                m = {'m_Y%s' % j: MultitaskGPModel((cat_train_x, cat_train_i),
                                                   cat_train_y, self.likelihood[j], self.n_causes,
                                                   self.data_dim)}

                self.models.update(m)
                #

        if self.GPtype is 'multioutput':

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

            # self.models = dict()

            list_train_x = []
            list_train_i = []
            list_train_k = []
            list_train_y = []

            for j in range(self.n_out):
                for i in self.n_causes:
                    aux_a = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=i)
                    aux_k = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=j)

                    list_train_x.append(self.train_x[self.train_a == i])
                    list_train_i.append(aux_a)
                    list_train_k.append(aux_k)

                    try:
                        list_train_y.append(self.train_y[:, j][self.train_a == i])
                    except IndexError:
                        list_train_y.append(self.train_y[self.train_a == i])

            cat_train_x = torch.cat(list_train_x)
            cat_train_i = torch.cat(list_train_i)
            cat_train_k = torch.cat(list_train_k)
            cat_train_y = torch.cat(list_train_y)

            if torch.cuda.is_available() and self.GPU:
                cat_train_x = cat_train_x.cuda()
                cat_train_i = cat_train_i.cuda()
                cat_train_k = cat_train_k.cuda()
                cat_train_y = cat_train_y.cuda()

            self.full_train_x = cat_train_x
            self.full_train_i = cat_train_i
            self.full_train_k = cat_train_k
            self.full_train_y = cat_train_y

            self.models = MultioutputGPModel((cat_train_x, cat_train_i,
                                             cat_train_k), cat_train_y,
                                             self.likelihood, self.n_causes, self.n_out, self.data_dim)

            # self.models.update(m)
            #

    def train(self, learn_rate=0.01, training_iter=50):
        """
        Optimize model's parameters with ADAM solver

        :learn_rate: learning rate
        :training_iter: number of training iterations
        """

        if self.GPtype is 'single':

            if torch.cuda.is_available() and self.GPU:
                for i, j in zip(self.models, range(len(self.likelihood))):
                    self.models[i] = self.models[i].cuda()
                    self.likelihood[j] = self.likelihood[j].cuda()

            for i, j, k, likel in zip(self.models, self.full_train_x, self.full_train_y, self.likelihood):

                self.models[i].train()
                likel.train()

                optimizer = torch.optim.Adam(self.models[i].parameters(), lr=learn_rate)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likel, self.models[i])

                for n in range(training_iter):
                    optimizer.zero_grad()
                    output = self.models[i](self.full_train_x[j])
                    loss = -mll(output, self.full_train_y[k])
                    loss.backward()
                    # print('Iter %d/%s - Loss: %.3f' % (n + 1, training_iter, loss.item()))
                    optimizer.step()

        elif self.GPtype is 'multitask':

            if torch.cuda.is_available() and self.GPU:
                for i, j in zip(self.models, range(len(self.likelihood))):
                    self.models[i] = self.models[i].cuda()
                    self.likelihood[j] = self.likelihood[j].cuda()

            for i, j, k, m, likel in zip(self.models, self.full_train_x, self.full_train_i,
                                         self.full_train_y, self.likelihood):

                self.models[i].train()
                likel.train()

                optimizer = torch.optim.Adam(self.models[i].parameters(), lr=learn_rate)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likel, self.models[i])

                for n in range(training_iter):
                    optimizer.zero_grad()
                    output = self.models[i](self.full_train_x[j], self.full_train_i[k])
                    loss = -mll(output, self.full_train_y[m])
                    loss.backward()
                    # print('Iter %d/%s - Loss: %.3f' % (n + 1, training_iter, loss.item()))
                    optimizer.step()

        elif self.GPtype is 'multioutput':

            if torch.cuda.is_available() and self.GPU:
                self.models = self.models.cuda()
                self.likelihood = self.likelihood.cuda()

            self.models.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.models.parameters(), lr=learn_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.models)

            for j in range(training_iter):
                optimizer.zero_grad()
                output = self.models(self.full_train_x, self.full_train_i, self.full_train_k)
                loss = -mll(output, self.full_train_y)
                loss.backward()
                # print('Iter %d/%s - Loss: %.3f' % (j + 1, training_iter, loss.item()))
                optimizer.step()

        #
        #

    def predict(self, x):
        """
        Predict out of sample

        :test_x: test set matrix of covariates
        """
        observed_pred = []
        test_x = torch.from_numpy(x).float()

        if torch.cuda.is_available() and self.GPU:
            test_x = test_x.cuda()

        if self.GPtype is 'single':

            for i, likel in zip(self.models, self.likelihood):
                self.models[i].eval()
                likel.eval()

                with torch.no_grad():
                    pred = likel(self.models[i](test_x))
                    pred = pred.mean.detach()

                if torch.cuda.is_available() and self.GPU:
                    pred = pred.cpu()

                observed_pred.append(pred.numpy())

        elif self.GPtype is 'multitask':

            list_test_i = []
            for i in self.n_causes:
                task_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=i)

                if torch.cuda.is_available() and self.GPU:
                    task_index = task_index.cuda()

                list_test_i.append(task_index)

            for i, likel in zip(self.models, self.likelihood):
                self.models[i].eval()
                likel.eval()

                for j in list_test_i:
                    with torch.no_grad():
                        pred = likel(self.models[i](test_x, j))
                        pred = pred.mean

                    if torch.cuda.is_available() and self.GPU:
                        pred = pred.cpu()

                    observed_pred.append(pred.numpy())

        elif self.GPtype is 'multioutput':

            self.models.eval()
            self.likelihood.eval()

            for j in range(self.n_out):
                for i in self.n_causes:
                    task_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=i)
                    out_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=j)

                    if torch.cuda.is_available() and self.GPU:
                        task_index = task_index.cuda()
                        out_index = out_index.cuda()

                    with torch.no_grad():
                        pred = self.likelihood(self.models(test_x, task_index, out_index))
                        pred = pred.mean

                    if torch.cuda.is_available() and self.GPU:
                        pred = pred.cpu()

                    observed_pred.append(pred.numpy())

        return observed_pred
    #


class CounterDKL:
    """
    Counterfactual Deep Kernel Learning object
    """

    def __init__(self, train_x, train_a, train_y, input_dim,
                 GPtype='single', hidden_layers=[50, 2], GPU=True):
        """
        Initialize model

        :input_dim: features X dimension
        :GPtype: type of learning paradigm ('single', 'multitask', 'multiout')
        :hidden_layers: hidden layers. List specifying how many nodes in each. Default [50, 2]
        """

        self.train_x = torch.from_numpy(train_x).float()
        self.train_a = torch.from_numpy(train_a).float()
        self.train_y = torch.from_numpy(train_y).float()
        self.GPU = GPU

        if torch.cuda.is_available() and self.GPU:
            self.train_x = self.train_x.cuda()
            self.train_a = self.train_a.cuda()
            self.train_y = self.train_y.cuda()

        self.data_dim = input_dim
        self.GPtype = GPtype
        self.hidden_layers = hidden_layers
        self.n_causes = torch.unique(self.train_a)

        try:
            self.n_out = train_y.shape[1]
        except IndexError:
            self.n_out = 1

        feature_extractor = LargeFeatureExtractor(self.data_dim, self.hidden_layers)

        if self.GPtype is 'single':

            # initialize likelihood and model
            self.likelihood = []
            self.models = dict()

            self.full_train_x = dict()
            self.full_train_y = dict()

            count = 0

            for j in range(self.n_out):
                for i in self.n_causes:
                    self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())

                    m = {'m_Y%s_A%s' % (j, i): DKLRegressionModel(self.train_x[self.train_a == i],
                                                                  self.train_y[:, j][self.train_a == i],
                                                                  self.likelihood[count],
                                                                  self.hidden_layers, deepcopy(feature_extractor))}

                    self.full_train_x.update({'m_Y%s_A%s' % (j, i): self.train_x[self.train_a == i]})
                    self.full_train_y.update({'m_Y%s_A%s' % (j, i): self.train_y[:, j][self.train_a == i]})

                    self.models.update(m)

                    count += 1

        if self.GPtype is 'multitask':

            # initialize likelihood and model
            self.models = dict()
            self.likelihood = []

            self.full_train_x = dict()
            self.full_train_i = dict()
            self.full_train_y = dict()

            for j in range(self.n_out):
                list_train_x = []
                list_train_i = []
                list_train_y = []
                for i in self.n_causes:
                    aux_a = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=i)

                    list_train_x.append(self.train_x[self.train_a == i])
                    list_train_i.append(aux_a)

                    try:
                        list_train_y.append(self.train_y[:, j][self.train_a == i])
                    except IndexError:
                        list_train_y.append(self.train_y[self.train_a == i])

                cat_train_x = torch.cat(list_train_x)
                cat_train_i = torch.cat(list_train_i)
                cat_train_y = torch.cat(list_train_y)

                if torch.cuda.is_available() and self.GPU:
                    cat_train_x = cat_train_x.cuda()
                    cat_train_i = cat_train_i.cuda()
                    cat_train_y = cat_train_y.cuda()

                self.full_train_x.update({'m_Y%s' % j: cat_train_x})
                self.full_train_i.update({'m_Y%s' % j: cat_train_i})
                self.full_train_y.update({'m_Y%s' % j: cat_train_y})

                self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())

                m = {'m_Y%s' % j: MultitaskDKLModel((cat_train_x, cat_train_i),
                                                    cat_train_y, self.likelihood[j],
                                                    self.hidden_layers, deepcopy(feature_extractor), self.n_causes)}

                self.models.update(m)
                #

        if self.GPtype is 'multioutput':

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

            # self.models = dict()

            list_train_x = []
            list_train_i = []
            list_train_k = []
            list_train_y = []

            for j in range(self.n_out):
                for i in self.n_causes:
                    aux_a = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=i)
                    aux_k = torch.full((self.train_x[self.train_a == i].shape[0], 1),
                                       dtype=torch.long, fill_value=j)

                    list_train_x.append(self.train_x[self.train_a == i])
                    list_train_i.append(aux_a)
                    list_train_k.append(aux_k)

                    try:
                        list_train_y.append(self.train_y[:, j][self.train_a == i])
                    except IndexError:
                        list_train_y.append(self.train_y[self.train_a == i])

            cat_train_x = torch.cat(list_train_x)
            cat_train_i = torch.cat(list_train_i)
            cat_train_k = torch.cat(list_train_k)
            cat_train_y = torch.cat(list_train_y)

            if torch.cuda.is_available() and self.GPU:
                cat_train_x = cat_train_x.cuda()
                cat_train_i = cat_train_i.cuda()
                cat_train_k = cat_train_k.cuda()
                cat_train_y = cat_train_y.cuda()

            self.full_train_x = cat_train_x
            self.full_train_i = cat_train_i
            self.full_train_k = cat_train_k
            self.full_train_y = cat_train_y

            self.models = MultioutputDKLModel((cat_train_x, cat_train_i,
                                               cat_train_k), cat_train_y,
                                              self.likelihood, self.hidden_layers, deepcopy(feature_extractor),
                                              self.n_causes, self.n_out)

            # self.models.update(m)
            #

    def train(self, learn_rate=0.01, training_iter=50):
        """
        Optimize model's parameters with ADAM solver

        :learn_rate: learning rate
        :training_iter: number of training iterations
        """

        if self.GPtype is 'single':

            if torch.cuda.is_available() and self.GPU:
                for i, j in zip(self.models, range(len(self.likelihood))):
                    self.models[i] = self.models[i].cuda()
                    self.likelihood[j] = self.likelihood[j].cuda()

            for i, j, k, likel in zip(self.models, self.full_train_x, self.full_train_y, self.likelihood):

                self.models[i].train()
                likel.train()

                optimizer = torch.optim.Adam([
                    {'params': self.models[i].feature_extractor.parameters()},
                    {'params': self.models[i].covar_module.parameters()},
                    {'params': self.models[i].mean_module.parameters()},
                    {'params': self.models[i].likelihood.parameters()},
                ], lr=learn_rate)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likel, self.models[i])

                for n in range(training_iter):
                    optimizer.zero_grad()
                    output = self.models[i](self.full_train_x[j])
                    loss = -mll(output, self.full_train_y[k])
                    loss.backward()
                    # print('Iter %d/%s - Loss: %.3f' % (n + 1, training_iter, loss.item()))
                    optimizer.step()

        elif self.GPtype is 'multitask':

            if torch.cuda.is_available() and self.GPU:
                for i, j in zip(self.models, range(len(self.likelihood))):
                    self.models[i] = self.models[i].cuda()
                    self.likelihood[j] = self.likelihood[j].cuda()

            for i, j, k, m, likel in zip(self.models, self.full_train_x, self.full_train_i,
                                         self.full_train_y, self.likelihood):

                self.models[i].train()
                likel.train()

                optimizer = torch.optim.Adam([
                    {'params': self.models[i].feature_extractor.parameters()},
                    {'params': self.models[i].covar_module.parameters()},
                    {'params': self.models[i].task_covar_module.parameters()},
                    {'params': self.models[i].mean_module.parameters()},
                    {'params': self.models[i].likelihood.parameters()},
                ], lr=learn_rate)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likel, self.models[i])

                for n in range(training_iter):
                    optimizer.zero_grad()
                    output = self.models[i](self.full_train_x[j], self.full_train_i[k])
                    loss = -mll(output, self.full_train_y[m])
                    loss.backward()
                    # print('Iter %d/%s - Loss: %.3f' % (n + 1, training_iter, loss.item()))
                    optimizer.step()

        elif self.GPtype is 'multioutput':

            if torch.cuda.is_available() and self.GPU:
                self.models = self.models.cuda()
                self.likelihood = self.likelihood.cuda()

            self.models.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam([
                {'params': self.models.feature_extractor.parameters()},
                {'params': self.models.covar_module.parameters()},
                {'params': self.models.task_covar_module.parameters()},
                {'params': self.models.task_covar_module_Y.parameters()},
                {'params': self.models.mean_module.parameters()},
                {'params': self.models.likelihood.parameters()},
            ], lr=learn_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.models)

            for j in range(training_iter):
                optimizer.zero_grad()
                output = self.models(self.full_train_x, self.full_train_i, self.full_train_k)
                loss = -mll(output, self.full_train_y)
                loss.backward()
                # print('Iter %d/%s - Loss: %.3f' % (j + 1, training_iter, loss.item()))
                optimizer.step()

        #
        #

    def predict(self, x):
        """
        Predict out of sample

        :test_x: test set matrix of covariates
        """
        observed_pred = []
        test_x = torch.from_numpy(x).float()

        if torch.cuda.is_available() and self.GPU:
            test_x = test_x.cuda()

        if self.GPtype is 'single':

            for i, likel in zip(self.models, self.likelihood):
                self.models[i].eval()
                likel.eval()

                with torch.no_grad():
                    pred = likel(self.models[i](test_x))
                    pred = pred.mean

                if torch.cuda.is_available() and self.GPU:
                    pred = pred.cpu()

                observed_pred.append(pred.numpy())

        elif self.GPtype is 'multitask':

            list_test_i = []
            for i in self.n_causes:
                task_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=i)

                if torch.cuda.is_available() and self.GPU:
                    task_index = task_index.cuda()

                list_test_i.append(task_index)

            for i, likel in zip(self.models, self.likelihood):
                self.models[i].eval()
                likel.eval()

                for j in list_test_i:
                    with torch.no_grad():
                        pred = likel(self.models[i](test_x, j))
                        pred = pred.mean

                    if torch.cuda.is_available() and self.GPU:
                        pred = pred.cpu()

                    observed_pred.append(pred.numpy())


        elif self.GPtype is 'multioutput':

            self.models.eval()
            self.likelihood.eval()

            for j in range(self.n_out):
                for i in self.n_causes:
                    task_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=i)
                    out_index = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=j)

                    if torch.cuda.is_available() and self.GPU:
                        task_index = task_index.cuda()
                        out_index = out_index.cuda()

                    with torch.no_grad():
                        pred = self.likelihood(self.models(test_x, task_index, out_index))
                        pred = pred.mean

                    if torch.cuda.is_available() and self.GPU:
                        pred = pred.cpu()

                    observed_pred.append(pred.numpy())

        return observed_pred
