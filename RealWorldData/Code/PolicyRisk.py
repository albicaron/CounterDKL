import os
import timeit
import pandas as pd
import numpy as np
from RealWorldData.Code.models.ClassifierModels import *
import gpytorch, torch
from RealWorldData.Code.utils import *
from RealWorldData.Code.models.Autoencoder import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

col_names = ['treat', 'age', 'education', 'black', 'hispanic',
             'married', 'nodegree', 're75', 're78']

exp_trt = pd.read_csv('./RealWorldData/Data/nsw_treated.txt', sep='  ', header=None)
exp_ctr = pd.read_csv('./RealWorldData/Data/nsw_control.txt', sep='  ', header=None)

exp_trt.columns = col_names
exp_ctr.columns = col_names

exp = pd.concat([exp_trt, exp_ctr])
exp.insert(0, 'data_id', ['Lalonde']*exp.shape[0])

psid = pd.read_stata('./RealWorldData/Data/psid_controls.dta')
psid = psid.drop(columns=['re74'])

assert np.all(exp.columns == psid.columns)

#####
psid['treat'].value_counts()

bind = pd.concat([exp, psid])
bind['empl'] = np.where(bind['re78'] > 0, 1.0, 0.0)

# Compute unbiased ATT and pol risk
ATT = np.mean(bind.loc[bind['treat'] == 1, 'empl']) - np.mean(bind.loc[(bind['treat'] == 0) & (bind['data_id'] == 'Lalonde'), 'empl'])

# Define X
X = np.array(bind.drop(columns=['data_id', 'treat', 're78', 'empl']).values)
sc = StandardScaler()
X[:, [0, 1, 6]] = sc.fit_transform(X[:, [0, 1, 6]])
Z = np.array(bind['treat'].values)
Y = np.array(bind['empl'].values)
Exp = np.array(bind['data_id'] == 'Lalonde')

# Results Storage
err_train = {'GP': [], 'CounterGP': [], 'PCAGP': [], 'PCAcounterGP': [],
             'AutoGP': [], 'AutoCountGP': [], 'DKL': [], 'CounterDKL': []}
err_test = {'GP': [], 'CounterGP': [], 'PCAGP': [], 'PCAcounterGP': [],
            'AutoGP': [], 'AutoCountGP': [], 'DKL': [], 'CounterDKL': []}

polrisk_train = {'GP': [], 'CounterGP': [], 'PCAGP': [], 'PCAcounterGP': [],
                 'AutoGP': [], 'AutoCountGP': [], 'DKL': [], 'CounterDKL': []}
polrisk_test = {'GP': [], 'CounterGP': [], 'PCAGP': [], 'PCAcounterGP': [],
                'AutoGP': [], 'AutoCountGP': [], 'DKL': [], 'CounterDKL': []}

runtime = {'GP': [], 'CounterGP': [], 'PCAGP': [], 'PCAcounterGP': [],
           'AutoGP': [], 'AutoCountGP': [], 'DKL': [], 'CounterDKL': []}

B = 10

for i in range(B):

    print('\n\n******* Validation %s\n\n' % (i+1))

    torch.manual_seed(i*13 + 51)
    torch.cuda.manual_seed(i*13 + 51)
    np.random.seed(i*13 + 51)

    # Train-Test Split (70-30%)
    split = np.random.choice(np.array([True, False]), X.shape[0], replace=True, p=np.array([0.7, 0.3]))

    X_train, X_test = tr_te_split(X, split)
    Z_train, Z_test = tr_te_split(Z, split)
    Y_train, Y_test = tr_te_split(Y, split)


    # GP
    start = timeit.default_timer()

    myGP = CounterGP(train_x=X_train, train_a=Z_train, train_y=Y_train,
                     input_dim=X.shape[1], GPtype='single', GPU=False)
    myGP.train(learn_rate=0.1, training_iter=100)
    pred_GP = myGP.predict(X)
    ATT_train_GP = np.mean(pred_GP[1][1][(split & (Z == 1))]) - np.mean(pred_GP[0][1][(split & (Z == 1))])
    ATT_test_GP = np.mean(pred_GP[1][1][(~split & (Z == 1))]) - np.mean(pred_GP[0][1][(~split & (Z == 1))])

    alloc = (pred_GP[1][1] - pred_GP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)])*np.mean(alloc_exp_train)
                          + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)])*(1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)])*np.mean(alloc_exp_test)
                         + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)])*(1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    # Storing
    err_train['GP'].append(MAE(ATT, ATT_train_GP))
    err_test['GP'].append(MAE(ATT, ATT_test_GP))
    polrisk_train['GP'].append(pol_risk_train)
    polrisk_test['GP'].append(pol_risk_test)
    runtime['GP'].append(stop - start)



    # CounterGP
    start = timeit.default_timer()

    multiGP = CounterGP(train_x=X_train, train_a=Z_train, train_y=Y_train,
                        input_dim=X.shape[1], GPtype='multitask', GPU=False)
    multiGP.train(learn_rate=0.1, training_iter=100)
    pred_multiGP = multiGP.predict(X)
    ATT_train_multiGP = np.mean(pred_multiGP[1][1][(split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(split & (Z == 1))])
    ATT_test_multiGP = np.mean(pred_multiGP[1][1][(~split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(~split & (Z == 1))])

    alloc = (pred_multiGP[1][1] - pred_multiGP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['CounterGP'].append(MAE(ATT, ATT_train_multiGP))
    err_test['CounterGP'].append(MAE(ATT, ATT_test_multiGP))
    polrisk_train['CounterGP'].append(pol_risk_train)
    polrisk_test['CounterGP'].append(pol_risk_test)
    runtime['CounterGP'].append(stop - start)


    # PCA + GP
    start = timeit.default_timer()

    myPCA = PCA(n_components=2)
    low_x_train = myPCA.fit_transform(X_train)
    low_x = myPCA.fit_transform(X)

    myGP = CounterGP(train_x=low_x_train, train_a=Z_train, train_y=Y_train,
                     input_dim=X.shape[1], GPtype='single', GPU=False)
    myGP.train(learn_rate=0.1, training_iter=100)
    pred_GP = myGP.predict(low_x)
    ATT_train_GP = np.mean(pred_GP[1][1][(split & (Z == 1))]) - np.mean(pred_GP[0][1][(split & (Z == 1))])
    ATT_test_GP = np.mean(pred_GP[1][1][(~split & (Z == 1))]) - np.mean(pred_GP[0][1][(~split & (Z == 1))])

    alloc = (pred_GP[1][1] - pred_GP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['PCAGP'].append(MAE(ATT, ATT_train_GP))
    err_test['PCAGP'].append(MAE(ATT, ATT_test_GP))
    polrisk_train['PCAGP'].append(pol_risk_train)
    polrisk_test['PCAGP'].append(pol_risk_test)
    runtime['PCAGP'].append(stop - start)


    # PCA + CounterGP
    start = timeit.default_timer()

    myPCA = PCA(n_components=2)
    low_x_train = myPCA.fit_transform(X_train)
    low_x = myPCA.fit_transform(X)

    multiGP = CounterGP(train_x=low_x_train, train_a=Z_train, train_y=Y_train,
                        input_dim=X.shape[1], GPtype='multitask', GPU=False)
    multiGP.train(learn_rate=0.1, training_iter=100)
    pred_multiGP = multiGP.predict(low_x)
    ATT_train_multiGP = np.mean(pred_multiGP[1][1][(split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(split & (Z == 1))])
    ATT_test_multiGP = np.mean(pred_multiGP[1][1][(~split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(~split & (Z == 1))])

    alloc = (pred_multiGP[1][1] - pred_multiGP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['PCAcounterGP'].append(MAE(ATT, ATT_train_multiGP))
    err_test['PCAcounterGP'].append(MAE(ATT, ATT_test_multiGP))
    polrisk_train['PCAcounterGP'].append(pol_risk_train)
    polrisk_test['PCAcounterGP'].append(pol_risk_test)
    runtime['PCAcounterGP'].append(stop - start)


    # Autoencoder + GP
    start = timeit.default_timer()

    AE = Autoencoder()
    AE.fit(X_train, X_test)

    low_x_train = AE.encod_pred(X_train)
    low_x = AE.encod_pred(X)

    myGP = CounterGP(train_x=low_x_train, train_a=Z_train, train_y=Y_train,
                     input_dim=X.shape[1], GPtype='single', GPU=False)
    myGP.train(learn_rate=0.1, training_iter=100)
    pred_GP = myGP.predict(low_x)
    ATT_train_GP = np.mean(pred_GP[1][1][(split & (Z == 1))]) - np.mean(pred_GP[0][1][(split & (Z == 1))])
    ATT_test_GP = np.mean(pred_GP[1][1][(~split & (Z == 1))]) - np.mean(pred_GP[0][1][(~split & (Z == 1))])

    alloc = (pred_GP[1][1] - pred_GP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['AutoGP'].append(MAE(ATT, ATT_train_GP))
    err_test['AutoGP'].append(MAE(ATT, ATT_test_GP))
    polrisk_train['AutoGP'].append(pol_risk_train)
    polrisk_test['AutoGP'].append(pol_risk_test)
    runtime['AutoGP'].append(stop - start)


    # Autoencoder + CounterGP
    start = timeit.default_timer()

    AE = Autoencoder()
    AE.fit(X_train, X_test)

    low_x_train = AE.encod_pred(X_train)
    low_x = AE.encod_pred(X)

    multiGP = CounterGP(train_x=low_x_train, train_a=Z_train, train_y=Y_train,
                        input_dim=X.shape[1], GPtype='multitask', GPU=False)
    multiGP.train(learn_rate=0.1, training_iter=100)
    pred_multiGP = multiGP.predict(low_x)
    ATT_train_multiGP = np.mean(pred_multiGP[1][1][(split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(split & (Z == 1))])
    ATT_test_multiGP = np.mean(pred_multiGP[1][1][(~split & (Z == 1))]) - np.mean(pred_multiGP[0][1][(~split & (Z == 1))])

    alloc = (pred_multiGP[1][1] - pred_multiGP[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['AutoCountGP'].append(MAE(ATT, ATT_train_multiGP))
    err_test['AutoCountGP'].append(MAE(ATT, ATT_test_multiGP))
    polrisk_train['AutoCountGP'].append(pol_risk_train)
    polrisk_test['AutoCountGP'].append(pol_risk_test)
    runtime['AutoCountGP'].append(stop - start)

    # DKL
    start = timeit.default_timer()

    myDKL = CounterDKL(X, Z, Y, X.shape[1], GPtype='single', hidden_layers=[10, 5, 2], GPU=False)
    myDKL.train(learn_rate=0.01, training_iter=50)
    pred_DKL = myDKL.predict(X)
    ATT_train_DKL = np.mean(pred_DKL[1][1][(split & (Z == 1))]) - np.mean(pred_DKL[0][1][(split & (Z == 1))])
    ATT_test_DKL = np.mean(pred_DKL[1][1][(~split & (Z == 1))]) - np.mean(pred_DKL[0][1][(~split & (Z == 1))])

    alloc = (pred_DKL[1][1] - pred_DKL[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['DKL'].append(MAE(ATT, ATT_train_DKL))
    err_test['DKL'].append(MAE(ATT, ATT_test_DKL))
    polrisk_train['DKL'].append(pol_risk_train)
    polrisk_test['DKL'].append(pol_risk_test)
    runtime['DKL'].append(stop - start)


    # CounterDKL
    start = timeit.default_timer()

    multiDKL = CounterDKL(X, Z, Y, X.shape[1], GPtype='multitask', hidden_layers=[10, 5, 2], GPU=False)
    multiDKL.train(learn_rate=0.01, training_iter=50)
    pred_multiDKL = multiDKL.predict(X)
    ATT_train_multiDKL = np.mean(pred_multiDKL[1][1][(split & (Z == 1))]) - np.mean(pred_multiDKL[0][1][(split & (Z == 1))])
    ATT_test_multiDKL = np.mean(pred_multiDKL[1][1][(~split & (Z == 1))]) - np.mean(pred_multiDKL[0][1][(~split & (Z == 1))])

    alloc = (pred_multiDKL[1][1] - pred_multiDKL[0][1] > 0).astype(int)
    alloc_exp_train = alloc[split & (Exp == 1)]
    alloc_exp_test = alloc[~split & (Exp == 1)]

    pol_risk_train = 1 - (
                np.mean(Y[Exp & split][(alloc_exp_train == 1) & (Z[Exp & split] == 1)]) * np.mean(alloc_exp_train)
                + np.mean(Y[Exp & split][(alloc_exp_train == 0) & (Z[Exp & split] == 0)]) * (
                            1 - np.mean(alloc_exp_train)))
    pol_risk_test = 1 - (
                np.mean(Y[Exp & ~split][(alloc_exp_test == 1) & (Z[Exp & ~split] == 1)]) * np.mean(alloc_exp_test)
                + np.mean(Y[Exp & ~split][(alloc_exp_test == 0) & (Z[Exp & ~split] == 0)]) * (
                            1 - np.mean(alloc_exp_test)))

    stop = timeit.default_timer()

    err_train['CounterDKL'].append(MAE(ATT, ATT_train_multiDKL))
    err_test['CounterDKL'].append(MAE(ATT, ATT_test_multiDKL))
    polrisk_train['CounterDKL'].append(pol_risk_train)
    polrisk_test['CounterDKL'].append(pol_risk_test)
    runtime['CounterDKL'].append(stop - start)

#
for i in err_train.keys():
    print('\n%s' % i)
    print('\nTrain MAE: %s  SE: %s' % (np.round(np.mean(err_train[i]), 4), np.round(MC_se(err_train[i], B=B), 4)))
    print('Test MAE: %s  SE: %s' % (np.round(np.mean(err_test[i]), 4), np.round(MC_se(err_test[i], B=B), 4)))
    print('Train Pol Risk: %s  SE: %s' % (np.round(np.nanmean(polrisk_train[i]), 4), np.round(MC_se(polrisk_train[i], B=B), 4)))
    print('Test Pol Risk: %s  SE: %s' % (np.round(np.nanmean(polrisk_test[i]), 4), np.round(MC_se(polrisk_test[i], B=B), 4)))
    print('Runtime (s): %s  SE: %s' % (np.round(np.mean(runtime[i]), 5), np.round(MC_se(runtime[i], B=B), 5)))

with open('./RealWorldData/Results/Results_PolRisk.txt', 'w') as f:

    for i in err_train.keys():
        f.write('%s' % i)
        f.write('\nTrain MAE: %s  SE: %s' % (np.round(np.mean(err_train[i]), 4), np.round(MC_se(err_train[i], B=B), 4)))
        f.write('\nTest MAE: %s  SE: %s' % (np.round(np.mean(err_test[i]), 4), np.round(MC_se(err_test[i], B=B), 4)))
        f.write('\nTrain Pol Risk: %s  SE: %s' % (np.round(np.nanmean(polrisk_train[i]), 4), np.round(MC_se(polrisk_train[i], B=B), 4)))
        f.write('\nTest Pol Risk: %s  SE: %s' % (np.round(np.nanmean(polrisk_test[i]), 4), np.round(MC_se(polrisk_test[i], B=B), 4)))
        f.write('\nRuntime (s): %s  SE: %s\n\n' % (np.round(np.mean(runtime[i]), 5), np.round(MC_se(runtime[i], B=B), 5)))
