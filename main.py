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
datasets = glob.glob("datasets/*.dat")

metrics = {
    'precision': precision,
    "recall": recall,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
}

print(datasets)
cls_dict = {"positive": 1, "negative": 0}
n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((3, n_datasets, n_splits * n_repeats, len(metrics)))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt(dataset, delimiter=",", comments="@",  dtype='unicode')
    X = dataset[:, :-1].astype(float)
    y = dataset[:, -1]
    y = np.char.strip(y)
    y = [cls_dict[el] for el in y]
    y = np.array(y)
    ir = max(np.count_nonzero(y == 0)/np.count_nonzero(y == 1), (np.count_nonzero(y == 1)/np.count_nonzero(y == 0)))
    # print(ir)
    # print(y)
    # print(X)
    
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        pool_classifiers = RandomForestClassifier(n_estimators=15)
        pool_classifiers.fit(X[train], y[train])
        clfs = {
            'KNORAU': KNORAU(pool_classifiers),
            'KNORAE': KNORAE(pool_classifiers),
            'KNORA_U': KNORA_U(pool_classifiers, ir=ir),
        }
        for clf_id, clf in enumerate(clfs):
            clfs[clf].fit(X[train], y[train])

            y_pred = clfs[clf].predict(X[test])
            # print(y_pred)
            for metric_id, metric in enumerate(metrics):
                # print(metric)
                scores[clf_id, data_id, fold_id, metric_id] = metrics[metric](
                    y[test], y_pred)

np.save('results', scores)


scores = np.load('results.npy')
scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", scores)
scores[np.isnan(scores)] = 0
scores = np.mean(scores, axis=1)


# metryki i metody
metrics=['Precision', "Recall",'Specificity', 'F1', 'G-mean']
methods=["KNORAU", 'KNORAE', 'KNORAU-IM']
N = scores.shape[0]

# kat dla kazdej z osi
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# spider plot
ax = plt.subplot(111, polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)

# Dodajemy wlasciwe ploty dla kazdej z metod
for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# Dodajemy legende
plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# Zapisujemy wykres
plt.savefig("radar", dpi=200)
