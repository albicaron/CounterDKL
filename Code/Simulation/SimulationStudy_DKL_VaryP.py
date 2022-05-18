# Importing
import gc, os, csv
from Code.Simulation.utils import *
from Code.models.ScalableCounterfactual_DKL import *
from multiprocessing import Queue
from threading import Thread


# Function
def train_parall(model, q):
    try:
        model.train()
        pred = model.predict(X)

        q.put(pred)

    except RuntimeError:
        pass


# Options
N = 1500
P = [10, 15, 20, 25]
B = 100

models = {'GP': [], 'CounterGP': [], 'MOGP': [],
          'DKL': [], 'CounterDKL': [], 'MODKL': []}

Counter_RMSE = {'10': deepcopy(models), '15': deepcopy(models),
                '20': deepcopy(models), '25': deepcopy(models)}
OPE_RMSE = deepcopy(Counter_RMSE)
OPL_PolAcc = deepcopy(Counter_RMSE)

for j in range(len(P)):

    print("\n********** Iteration P:", P[j])

    for i in range(B):

        print("\n***** Iteration B:", i+1)
        # Generate Data
        X, A, Y, C, Y_true, C_true = backdoor_dgp(N=N, P=P[j], rng=i*13)

        np.unique(A, return_counts=True)

        # Train-Test Split (80-20%)
        split = np.random.choice(np.array([True, False]), N, replace=True, p=np.array([0.8, 0.2]))

        X_train, X_test = tr_te_split(X, split)
        A_train, A_test = tr_te_split(A, split)
        Y_train, Y_test = tr_te_split(Y, split)
        C_train, C_test = tr_te_split(C, split)

        Y_true_train, Y_true_test = tr_te_split(Y_true, split)
        C_true_train, C_true_test = tr_te_split(C_true, split)

        ###### CounterFactual Inference
        # GP Reg
        model = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                          input_dim=P[j], GPtype='single', GPU=True)

        model_GP = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                             input_dim=P[j], GPtype='multitask', GPU=True)

        model_MOGP = CounterGP(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                               input_dim=P[j], GPtype='multioutput')

        # DKL Reg
        model_DKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                               input_dim=P[j], GPtype='single', hidden_layers=[50, 50, 2])

        model_MDKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                                input_dim=P[j], GPtype='multitask', hidden_layers=[50, 50, 2])

        model_MODKL = CounterDKL(train_x=X_train, train_a=A_train, train_y=np.c_[Y_train, C_train],
                                 input_dim=P[j], GPtype='multioutput', hidden_layers=[50, 50, 2])

        if __name__ == '__main__':
            mods = [model, model_GP, model_MOGP,
                    model_DKL, model_MDKL, model_MODKL]

            ques = list()
            threads = list()

            for q in range(len(mods)):
                ques.append(Queue())
                threads.append(Thread(target=train_parall, args=(mods[q], ques[q], )))

            for thread in threads:
                thread.start()

            counter, counter_GP, counter_MOGP, counter_DKL, counter_MDKL, counter_MODKL = [q.get() for q in ques]

        gc.collect()
        torch.cuda.empty_cache()

        # Single GP
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter[i+4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['GP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        # Counterfactual GP
        ## Van der Schaar's equivalent
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_GP[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_GP[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['CounterGP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        # MO-Counterfactual GP

        # Because of parameter proliferation, in high-dimensions with larger N, MOGP optimization
        # often fail to converge or returns NaNs in training. We have to take the latter into account
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MOGP[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MOGP[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['MOGP'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        #### Counterfactual DKL
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_DKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_DKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['DKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        #### Counterfactual Multitask DKL
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MDKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MDKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['CounterDKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        #### Counterfactual Multitask DKL
        err_quad_Y = np.array([RMSE(Y_true_test[:, i], counter_MODKL[i][~split]) for i in range(4)])
        err_quad_C = np.array([RMSE(C_true_test[:, i], counter_MODKL[i + 4][~split]) for i in range(4)])

        Counter_RMSE[str(P[j])]['MODKL'].append(np.mean([np.mean(err_quad_Y), np.mean(err_quad_C)]))

        ###### OPE
        mask = np.random.choice([0, 1, 2, 3], N, replace=True)
        pi_b = np.zeros((mask.size, mask.max() + 1))
        pi_b[np.arange(mask.size), mask] = 1

        pol_val_Y = np.multiply(Y_true, pi_b).sum() / N
        pol_val_C = np.multiply(C_true, pi_b).sum() / N

        GP_Y = np.multiply(np.array(counter)[range(4), :].T, pi_b).sum() / N
        GP_C = np.multiply(np.array(counter)[range(4, 8), :].T, pi_b).sum() / N

        OPE_RMSE[str(P[j])]['GP'].append(np.mean(np.c_[RMSE(pol_val_Y, GP_Y), RMSE(pol_val_C, GP_C)]))

        CounterGP_Y = np.multiply(np.array(counter_GP)[range(4), :].T, pi_b).sum() / N
        CounterGP_C = np.multiply(np.array(counter_GP)[range(4, 8), :].T, pi_b).sum() / N

        OPE_RMSE[str(P[j])]['CounterGP'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterGP_Y), RMSE(pol_val_C, CounterGP_C)]))

        try:
            CounterMOGP_Y = np.multiply(np.array(counter_MOGP)[range(4), :].T, pi_b).sum() / P[j]
            CounterMOGP_C = np.multiply(np.array(counter_MOGP)[range(4, 8), :].T, pi_b).sum() / P[j]

            OPE_RMSE[str(P[j])]['MOGP'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMOGP_Y), RMSE(pol_val_C, CounterMOGP_C)]))

        except ValueError:
            pass

        CounterDKL_Y = np.multiply(np.array(counter_DKL)[range(4), :].T, pi_b).sum() / N
        CounterDKL_C = np.multiply(np.array(counter_DKL)[range(4, 8), :].T, pi_b).sum() / N

        OPE_RMSE[str(P[j])]['DKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterDKL_Y), RMSE(pol_val_C, CounterDKL_C)]))

        CounterMDKL_Y = np.multiply(np.array(counter_MDKL)[range(4), :].T, pi_b).sum() / N
        CounterMDKL_C = np.multiply(np.array(counter_MDKL)[range(4, 8), :].T, pi_b).sum() / N

        OPE_RMSE[str(P[j])]['CounterDKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMDKL_Y), RMSE(pol_val_C, CounterMDKL_C)]))

        CounterMODKL_Y = np.multiply(np.array(counter_MODKL)[range(4), :].T, pi_b).sum() / N
        CounterMODKL_C = np.multiply(np.array(counter_MODKL)[range(4, 8), :].T, pi_b).sum() / N

        OPE_RMSE[str(P[j])]['MODKL'].append(np.mean(np.c_[RMSE(pol_val_Y, CounterMODKL_Y), RMSE(pol_val_C, CounterMODKL_C)]))


        ###### OPL
        pi_opt_Y = Y_true.argmax(axis=1)
        pi_opt_C = Y_true.argmax(axis=1)

        GP_Y_opt = np.array(counter)[range(4), :].T.argmax(axis=1)
        GP_C_opt = np.array(counter)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['GP'].append(np.mean(np.c_[np.sum(pi_opt_Y == GP_Y_opt)/N, np.sum(pi_opt_C == GP_C_opt)/N]))

        CounterGP_Y_opt = np.array(counter_GP)[range(4), :].T.argmax(axis=1)
        CounterGP_C_opt = np.array(counter_GP)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['CounterGP'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterGP_Y_opt)/N,
                                                     np.sum(pi_opt_C == CounterGP_C_opt)/N]))

        CounterMOGP_Y_opt = np.array(counter_MOGP)[range(4), :].T.argmax(axis=1)
        CounterMOGP_C_opt = np.array(counter_MOGP)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['MOGP'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMOGP_Y_opt) / N,
                                                np.sum(pi_opt_C == CounterMOGP_C_opt) / N]))

        CounterDKL_Y_opt = np.array(counter_DKL)[range(4), :].T.argmax(axis=1)
        CounterDKL_C_opt = np.array(counter_DKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['DKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterDKL_Y_opt) / N,
                                               np.sum(pi_opt_C == CounterDKL_C_opt) / N]))

        CounterMDKL_Y_opt = np.array(counter_MDKL)[range(4), :].T.argmax(axis=1)
        CounterMDKL_C_opt = np.array(counter_MDKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['CounterDKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMDKL_Y_opt) / N,
                                                      np.sum(pi_opt_C == CounterMDKL_C_opt) / N]))

        CounterMODKL_Y_opt = np.array(counter_MODKL)[range(4), :].T.argmax(axis=1)
        CounterMODKL_C_opt = np.array(counter_MODKL)[range(4, 8), :].T.argmax(axis=1)

        OPL_PolAcc[str(P[j])]['MODKL'].append(np.mean(np.c_[np.sum(pi_opt_Y == CounterMODKL_Y_opt) / N,
                                              np.sum(pi_opt_C == CounterMODKL_C_opt) / N]))

for k in P:
    print("\n\n************  P:", k, " ************\n")
    a_file = open("%s/Results/VaryP_Counter_RMSE_Results_P%s_N%s.csv" % (os.path.dirname(os.getcwd()), k, N), "w", newline='')
    writer = csv.writer(a_file)
    for i, j in zip(Counter_RMSE[str(k)].keys(), Counter_RMSE[str(k)].values()):
        print(i, ': ', np.around(np.mean(np.array(j)), 2), '  SE: ', np.around(MC_se(np.array(j), B), 2))
        writer.writerow([i, np.mean(np.array(j))])
        writer.writerow(['SE', MC_se(np.array(j), B)])
    a_file.close()

    a_file = open("%s/Results/VaryP_OPE_RMSE_Results_P%s_N%s.csv" % (os.path.dirname(os.getcwd()), k, N), "w", newline='')
    writer = csv.writer(a_file)
    print('\n')
    for i, j in zip(OPE_RMSE[str(k)].keys(), OPE_RMSE[str(k)].values()):
        print(i, ': ', np.around(np.mean(np.array(j)), 2), '  SE: ', np.around(MC_se(np.array(j), B), 2))
        writer.writerow([i, np.mean(np.array(j))])
        writer.writerow(['SE', MC_se(np.array(j), B)])
    a_file.close()

    a_file = open("%s/Results/VaryP_OPL_PolAcc_Results_P%s_N%s.csv" % (os.path.dirname(os.getcwd()), k, N), "w", newline='')
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
    for j in P:
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
    plt.plot(P, np.array(RMSE_avgs[i]), c=color, label=i)
    plt.fill_between(P, np.array(RMSE_avgs[i]) + np.array(RMSE_se[i]),
                     np.array(RMSE_avgs[i]) - np.array(RMSE_se[i]), color='grey', alpha=0.2)
    plt.xlabel('P')
    plt.ylabel('RMSE')

plt.subplot(1, 3, 2)
plt.grid()
for i, color in zip(RMSE_avgs.keys(), colors):
    plt.plot(P, np.array(OPE_avgs[i]), c=color, label=i)
    plt.fill_between(P, np.array(OPE_avgs[i]) + np.array(OPE_se[i]),
                     np.array(OPE_avgs[i]) - np.array(OPE_se[i]), color='grey', alpha=0.2)
    plt.xlabel('P')
    plt.ylabel('RMSE')

plt.subplot(1, 3, 3)
plt.grid()
for i, color in zip(RMSE_avgs.keys(), colors):
    plt.plot(P, np.array(OPL_avgs[i]), c=color, label=i)
    plt.fill_between(P, np.array(OPL_avgs[i]) + np.array(OPL_se[i]),
                     np.array(OPL_avgs[i]) - np.array(OPL_se[i]), color='grey', alpha=0.2)
    plt.xlabel('P')
    plt.ylabel('Accuracy')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('%s/Results/VaryP_Results_N%s.pdf' % (os.path.dirname(os.getcwd()), N))
