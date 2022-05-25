from Code.Simulation.utils import *
from Code.models.ScalableCounterfactual_DKL import *

datasets = ['australian', 'tae', 'heart', 'cmc']
models = ['GP', 'CounterGP', 'DKL', 'CounterDKL']

bias_mod = {}
for df in datasets:
    bias_mod.update({df: {}})
    for mod in models:
        bias_mod[df].update({mod: []})

B = 10

for df in datasets:

    X, A = get_data(df, scale=True)

    for b in range(2, B):

        print('\n\nDataset: %s, Iteration: %s' % (df, b+1))

        # Generate Data
        torch.manual_seed(b*2 + 20)
        torch.cuda.manual_seed(b*2 + 20)
        np.random.seed(b*2 + 20)

        if np.min(np.unique(A)) is not 0:
            A = A - np.min(np.unique(A))

        if df is 'glass':
            A[A > 3] = A[A > 3] - 1
        elif df is 'abalone':
            A[A > 27] = A[A > 27] - 1

        Y_true, Y = gen_Y(X, A)
        pol_val, random_pol = get_pol_val(Y_true, A)

        N, P = X.shape

        # Simple GP Reg
        model = CounterGP(train_x=X, train_a=A, train_y=Y.reshape(-1, 1),
                          input_dim=P, GPtype='single', GPU=True)
        model.train(training_iter=100, learn_rate=0.01)
        counter = np.array(model.predict(X)).transpose()

        GP_polval = np.mean(counter * random_pol)

        bias_mod[df]['GP'].append(bias(pol_val, GP_polval))


        # Counterfactual GP
        model_GP = CounterGP(train_x=X, train_a=A, train_y=Y.reshape(-1, 1),
                             input_dim=P, GPtype='multitask', GPU=True)
        model_GP.train(training_iter=100, learn_rate=0.01)
        counter_GP = np.array(model_GP.predict(X)).transpose()

        MGP_polval = np.mean(counter_GP * random_pol)

        bias_mod[df]['CounterGP'].append(bias(pol_val, MGP_polval))


        #### Counterfactual DKL
        model_DKL = CounterDKL(train_x=X, train_a=A, train_y=Y.reshape(-1, 1),
                               input_dim=P, GPtype='single', hidden_layers=[5, 5, 2], GPU=True)
        model_DKL.train(training_iter=50, learn_rate=0.01)
        counter_DKL = np.array(model_DKL.predict(X)).transpose()

        DKL_polval = np.mean(counter_DKL * random_pol)

        bias_mod[df]['DKL'].append(bias(pol_val, DKL_polval))


        #### Counterfactual Multitask DKL
        model_MDKL = CounterDKL(train_x=X, train_a=A, train_y=Y.reshape(-1, 1),
                                input_dim=P, GPtype='multitask', hidden_layers=[5, 5, 2], GPU=True)
        model_MDKL.train(training_iter=50, learn_rate=0.01)
        counter_MDKL = np.array(model_MDKL.predict(X)).transpose()

        MDKL_polval = np.mean(counter_MDKL * random_pol)

        bias_mod[df]['CounterDKL'].append(bias(pol_val, MDKL_polval))


for name, elem in bias_mod.items():
    print('\n\n%s\n' % name)
    for mod, bias in elem.items():
        print('%s - bias: %f, SE: %f' % (mod, np.mean(bias), MC_se(bias, B)))
