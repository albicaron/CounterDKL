# Importing
import gc, os, csv
from Code.Simulation.utils import *
from Code.models.ScalableCounterfactual_DKL import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Options
N = [500, 1000, 1500, 2000, 2500, 3000]
P = 10
B = 100

models = {'GP': [], 'CounterGP': [], 'MOGP': [],
          'DKL': [], 'CounterDKL': [], 'MODKL': []}

Counter_RMSE = {'500': deepcopy(models), '1000': deepcopy(models), '1500': deepcopy(models),
                '2000': deepcopy(models), '2500': deepcopy(models), '3000': deepcopy(models)}
OPE_RMSE = deepcopy(Counter_RMSE)
OPL_PolAcc = deepcopy(Counter_RMSE)

for j in range(len(N)):

    print("\n********** Iteration N:", N[j])

    for i in range(B):

        print("\n***** Iteration B:", i+1)
        # Generate Data
        X, A, Y, C, Y_true, C_true = backdoor_dgp(N=N[j], P=P, rng=i*13)

        np.unique(A, return_counts=True)

        # Train-Test Split (80-20%)
        split = np.random.choice(np.array([True, False]), N[j], replace=True, p=np.array([0.8, 0.2]))

        X_train, X_test = tr_te_split(X, split)
        A_train, A_test = tr_te_split(A, split)
        Y_train, Y_test = tr_te_split(Y, split)
        C_train, C_test = tr_te_split(C, split)

        Y_true_train, Y_true_test = tr_te_split(Y_true, split)
        C_true_train, C_true_test = tr_te_split(C_true, split)

        ###### CounterFactual Inference
        # Simple GP Reg
        model = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                          input_dim=P, GPtype='single')
        model.train(training_iter=100, learn_rate=0.01)
        counter = model.predict(X)

        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter[i+4][~split]) for i in range(4)])

        Counter_RMSE[str(N[j])]['GP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        gc.collect()
        torch.cuda.empty_cache()

        # Counterfactual GP
        ## Van der Schaar's equivalent
        model_GP = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                             input_dim=P, GPtype='multitask')
        model_GP.train(training_iter=100, learn_rate=0.01)
        counter_GP = model_GP.predict(X)

        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_GP[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_GP[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(N[j])]['CounterGP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        gc.collect()
        torch.cuda.empty_cache()

        # MO-Counterfactual GP

        # Because of parameter proliferation, in high-dimensions with larger N, MOGP optimization
        # often fail to converge or returns NaNs in training. We have to take the latter into account
        model_MOGP = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                               input_dim=P, GPtype='multioutput')
        try:
            model_MOGP.train(training_iter=100, learn_rate=0.01)
            counter_MOGP = model_MOGP.predict(X)

            err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MOGP[i][~split]) for i in range(4)])
            err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MOGP[i + 4][~split]) for i in range(4)])

            Counter_RMSE[str(N[j])]['MOGP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        except RuntimeError:
            pass

        gc.collect()
        torch.cuda.empty_cache()

        #### Counterfactual DKL
        model_DKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                               input_dim=P, GPtype='single', hidden_layers=[50, 50, 2])
        model_DKL.train(training_iter=50, learn_rate=0.01)
        counter_DKL = model_DKL.predict(X)

        gc.collect()
        torch.cuda.empty_cache()

        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_DKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_DKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(N[j])]['DKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        #### Counterfactual Multitask DKL
        model_MDKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                                input_dim=P, GPtype='multitask', hidden_layers=[50, 50, 2])
        model_MDKL.train(training_iter=50, learn_rate=0.01)
        counter_MDKL = model_MDKL.predict(X)

        gc.collect()
        torch.cuda.empty_cache()

        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MDKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MDKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(N[j])]['CounterDKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))


        #### Counterfactual Multitask DKL
        model_MODKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                                 input_dim=P, GPtype='multioutput', hidden_layers=[50, 50, 2])
        model_MODKL.train(training_iter=50, learn_rate=0.01)
        counter_MODKL = model_MODKL.predict(X)

        gc.collect()
        torch.cuda.empty_cache()

        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MODKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MODKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(N[j])]['MODKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        ###### OPE
        mask = np.random.choice([0, 1, 2, 3], N[j], replace=True)
        pi_b = np.zeros((mask.size, mask.max() + 1))
        pi_b[np.arange(mask.size), mask] = 1

        pol_val_Y = np.multiply(Y_true, pi_b).sum() / N[j]
        pol_val_C = np.multiply(C_true, pi_b).sum() / N[j]

        GP_Y = np.multiply(np.array(counter)[range(4), :].T, pi_b).sum() / N[j]
        GP_C = np.multiply(np.array(counter)[range(4, 8), :].T, pi_b).sum() / N[j]

        OPE_RMSE[str(N[j])]['GP'].append(np.mean(np.c_[RMSE(pol_val_Y, GP_Y), RMSE(pol_val_C, GP_C)]))

        CounterGP_Y = np.multiply(np.array(counter_GP)[range(4), :].T, pi_b).sum() / N[j]
        CounterGP_C = np.multiply(np.array(counter_GP)[range(4, 8), :].T, pi_b).sum() / N[j]

        OPE_RMSE[str(N[j])]['CounterGP'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterGP_Y), RMSE(pol_val_C, CounterGP_C)]))

        try:
            CounterMOGP_Y = np.multiply(np.array(counter_MOGP)[range(4), :].T, pi_b).sum() / N[j]
            CounterMOGP_C = np.multiply(np.array(counter_MOGP)[range(4, 8), :].T, pi_b).sum() / N[j]

            OPE_RMSE[str(N[j])]['MOGP'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMOGP_Y), RMSE(pol_val_C, CounterMOGP_C)]))

        except ValueError:
            pass

        CounterDKL_Y = np.multiply(np.array(counter_DKL)[range(4), :].T, pi_b).sum() / N[j]
        CounterDKL_C = np.multiply(np.array(counter_DKL)[range(4, 8), :].T, pi_b).sum() / N[j]

        OPE_RMSE[str(N[j])]['DKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterDKL_Y), RMSE(pol_val_C, CounterDKL_C)]))

        CounterMDKL_Y = np.multiply(np.array(counter_MDKL)[range(4), :].T, pi_b).sum() / N[j]
        CounterMDKL_C = np.multiply(np.array(counter_MDKL)[range(4, 8), :].T, pi_b).sum() / N[j]

        OPE_RMSE[str(N[j])]['CounterDKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMDKL_Y), RMSE(pol_val_C, CounterMDKL_C)]))

        CounterMODKL_Y = np.multiply(np.array(counter_MODKL)[range(4), :].T, pi_b).sum() / N[j]
        CounterMODKL_C = np.multiply(np.array(counter_MODKL)[range(4, 8), :].T, pi_b).sum() / N[j]

        OPE_RMSE[str(N[j])]['MODKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMODKL_Y), RMSE(pol_val_C, CounterMODKL_C)]))


        ###### OPL
        pi_opt_Y = Y_true.argmax(axis=1)
        pi_opt_C = Y_true.argmax(axis=1)

        GP_Y_opt = np.array(counter)[range(4), :].T.argmax(axis=1)
        GP_C_opt = np.array(counter)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['GP'].append(np.mean(np.c_[np.sum(pi_opt_Y == GP_Y_opt)/N[j], np.sum(pi_opt_C == GP_C_opt)/N[j]]))

        CounterGP_Y_opt = np.array(counter_GP)[range(4), :].T.argmax(axis=1)
        CounterGP_C_opt = np.array(counter_GP)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['CounterGP'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterGP_Y_opt)/N[j],
                                                     np.sum(pi_opt_C == CounterGP_C_opt)/N[j]]))

        CounterMOGP_Y_opt = np.array(counter_MOGP)[range(4), :].T.argmax(axis=1)
        CounterMOGP_C_opt = np.array(counter_MOGP)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['MOGP'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMOGP_Y_opt) / N[j],
                                                np.sum(pi_opt_C == CounterMOGP_C_opt) / N[j]]))

        CounterDKL_Y_opt = np.array(counter_DKL)[range(4), :].T.argmax(axis=1)
        CounterDKL_C_opt = np.array(counter_DKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['DKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterDKL_Y_opt) / N[j],
                                               np.sum(pi_opt_C == CounterDKL_C_opt) / N[j]]))

        CounterMDKL_Y_opt = np.array(counter_MDKL)[range(4), :].T.argmax(axis=1)
        CounterMDKL_C_opt = np.array(counter_MDKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['CounterDKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMDKL_Y_opt) / N[j],
                                                      np.sum(pi_opt_C == CounterMDKL_C_opt) / N[j]]))

        CounterMODKL_Y_opt = np.array(counter_MODKL)[range(4), :].T.argmax(axis=1)
        CounterMODKL_C_opt = np.array(counter_MODKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(N[j])]['MODKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMODKL_Y_opt) / N[j],
                                              np.sum(pi_opt_C == CounterMODKL_C_opt) / N[j]]))

for k in N:
    print("\n\n************  N:", k, " ************\n")
    a_file = open("./Results/Counter_RMSE_Results_N%s.csv" % k, "w", newline='')
    writer = csv.writer(a_file)
    for i, j in zip(Counter_RMSE[str(k)].keys(), Counter_RMSE[str(k)].values()):
        print(i, ': ', np.around(np.mean(np.array(j)), 2), '  SE: ', np.around(MC_se(np.array(j), B), 2))
        writer.writerow([i, np.mean(np.array(j))])
        writer.writerow(['SE', MC_se(np.array(j), B)])
    a_file.close()

    a_file = open("./Results/OPE_RMSE_Results_N%s.csv" % k, "w", newline='')
    writer = csv.writer(a_file)
    print('\n')
    for i, j in zip(OPE_RMSE[str(k)].keys(), OPE_RMSE[str(k)].values()):
        print(i, ': ', np.around(np.mean(np.array(j)), 2), '  SE: ', np.around(MC_se(np.array(j), B), 2))
        writer.writerow([i, np.mean(np.array(j))])
        writer.writerow(['SE', MC_se(np.array(j), B)])
    a_file.close()

    a_file = open("./Results/OPL_PolAcc_Results_N%s.csv" % k, "w", newline='')
    writer = csv.writer(a_file)
    print('\n')
    for i, j in zip(OPL_PolAcc[str(k)].keys(), OPL_PolAcc[str(k)].values()):
        print(i, ': ', np.around(np.mean(np.array(j)), 2), '  SE: ', np.around(MC_se(np.array(j), B), 2))
        writer.writerow([i, np.mean(np.array(j))])
        writer.writerow(['SE', MC_se(np.array(j), B)])
    a_file.close()


import matplotlib.pyplot as plt

RMSE_avgs, OPE_avgs, OPL_avgs = deepcopy(models), deepcopy(models), deepcopy(models)
RMSE_se, OPE_se, OPL_se = deepcopy(models), deepcopy(models), deepcopy(models)

for item in RMSE_avgs.keys():
    for j in N:
        RMSE_avgs[item].append(np.mean(Counter_RMSE[str(j)][item]))
        OPE_avgs[item].append(np.mean(OPE_RMSE[str(j)][item]))
        OPL_avgs[item].append(np.mean(OPL_PolAcc[str(j)][item]))

        RMSE_se[item].append(MC_se(Counter_RMSE[str(j)][item], B))
        OPE_se[item].append(MC_se(OPE_RMSE[str(j)][item], B))
        OPL_se[item].append(MC_se(OPL_PolAcc[str(j)][item], B))


colors = ['#FF9999', '#FF3333', '#990000',
          '#CCE5FF', '#3399FF', '#3333FF']
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.grid()
for i, color in zip(RMSE_avgs.keys(), colors):
    plt.plot(N, RMSE_avgs[i][0], c=color, label=i)
    plt.fill_between(N, RMSE_avgs[i][0] + RMSE_se[i][0], RMSE_avgs[i][0] - RMSE_se[i][0], color='grey', alpha=0.2)
    plt.xlabel('N')
    plt.ylabel('RMSE')

plt.subplot(1, 3, 2)
plt.grid()
for i, color in zip(RMSE_avgs.keys(), colors):
    plt.plot(N, OPE_avgs[i][0], c=color, label=i)
    plt.fill_between(N, OPE_avgs[i][0] + OPE_se[i][0], OPE_avgs[i][0] - OPE_se[i][0], color='grey', alpha=0.2)
    plt.xlabel('N')
    plt.ylabel('RMSE')

plt.subplot(1, 3, 3)
plt.grid()
for i, color in zip(RMSE_avgs.keys(), colors):
    plt.plot(N, OPL_avgs[i][0], c=color, label=i)
    plt.fill_between(N, OPL_avgs[i][0] + OPL_se[i][0], OPL_avgs[i][0] - OPL_se[i][0], color='grey', alpha=0.2)
    plt.xlabel('N')
    plt.ylabel('Accuracy')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('./Results/Results.pdf')
