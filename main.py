import numpy as np
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
import glob
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1
from knoranew import KNORA_U
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate
# datasets = glob.glob("datasets/*.dat")

# metrics = {
#     'precision': precision,
#     "recall": recall,
#     'specificity': specificity,
#     'f1': f1_score,
#     'g-mean': geometric_mean_score_1,
# }

# print(datasets)
# cls_dict = {"positive": 1, "negative": 0}
# n_datasets = len(datasets)
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

# scores = np.zeros((3, n_datasets, n_splits * n_repeats, len(metrics)))

# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt(dataset, delimiter=",", comments="@",  dtype='unicode')
#     X = dataset[:, :-1].astype(float)
#     y = dataset[:, -1]
#     y = np.char.strip(y)
#     y = [cls_dict[el] for el in y]
#     y = np.array(y)
#     ir = max(np.count_nonzero(y == 0)/np.count_nonzero(y == 1), (np.count_nonzero(y == 1)/np.count_nonzero(y == 0)))
#     # print(ir)
#     # print(y)
#     # print(X)
    
#     for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#         pool_classifiers = RandomForestClassifier(n_estimators=15, random_state=1234)
#         pool_classifiers.fit(X[train], y[train])
#         clfs = {
#             'KNORAU': KNORAU(pool_classifiers, random_state=1234),
#             'KNORAE': KNORAE(pool_classifiers, random_state=1234),
#             'KNORA_U': KNORA_U(pool_classifiers, ir=ir),
#         }
#         for clf_id, clf in enumerate(clfs):
#             clfs[clf].fit(X[train], y[train])

#             y_pred = clfs[clf].predict(X[test])
#             # print(y_pred)
#             for metric_id, metric in enumerate(metrics):
#                 # print(metric)
#                 scores[clf_id, data_id, fold_id, metric_id] = metrics[metric](
#                     y[test], y_pred)

# np.save('results', scores)


# STATISTICS

clfs = {
            'KNORAU': KNORAU,
            'KNORAE': KNORAE,
            'KNORA_IMB': KNORA_U,
        }
scores = np.load('results.npy')
scores = np.mean(scores, axis=2).T
print(scores.shape)
for i in range(scores.shape[0]):
    mean_scores = scores[i,:,:]

    # Ranks
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    # Mean ranks
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks, "\n")


    # w-statistic and p-value
    alfa = 0.05
    w_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)


    headers = list(clfs.keys())
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("w-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    # Advantage
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    # Statistical significance
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(
        np.concatenate((names_column, significance), axis=1), headers
    )
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    # Statistical significance better
    stat_better = significance * advantage
    stat_better_table = tabulate(
        np.concatenate((names_column, stat_better), axis=1), headers
    )
    print("Statistically significantly better:\n", stat_better_table)

# RADAR PLOT

# scores = np.load('results.npy')
# print("\nMean scores:\n", scores)
# scores[np.isnan(scores)] = 0
# scores = np.mean(scores, axis=1)


# # metryki i metody
# metrics=['Precision', "Recall",'Specificity', 'F1', 'G-mean']
# methods=["KNORAU", 'KNORAE', 'KNORAU-IM']
# N = scores.shape[0]

# # kat dla kazdej z osi
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]

# # spider plot
# ax = plt.subplot(111, polar=True)

# # pierwsza os na gorze
# ax.set_theta_offset(pi / 2)
# ax.set_theta_direction(-1)

# # po jednej osi na metryke
# plt.xticks(angles[:-1], metrics)

# # os y
# ax.set_rlabel_position(0)
# plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
# ["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
# color="grey", size=7)
# plt.ylim(0,1)

# # Dodajemy wlasciwe ploty dla kazdej z metod
# for method_id, method in enumerate(methods):
#     values=scores[:, method_id].tolist()
#     values += values[:1]
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# # Dodajemy legende
# plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# # Zapisujemy wykres
# plt.savefig("radar", dpi=200)
