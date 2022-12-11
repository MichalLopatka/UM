import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import DistanceMetric


class KNORA_U(BaseEstimator, ClassifierMixin):
    def __init__(self, pool_classifiers=None, k=7, random_state=66, ir=1):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.random_state = random_state
        self.ir = ir
        np.random.seed(self.random_state)


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

        return self


    def region_of_competence(self, xquery, vali_set):
        region = []

        for i in vali_set:
            score = np.array(
                DistanceMetric.get_metric("euclidean").pairwise([xquery, i[0]])
            ).max()
            region.append([i, score])

        region = sorted(region, key=lambda t: t[1])[: self.k]

        return region


    def selection(self, clf, region):
        for i in region:
            pred = clf.predict(i[0][0].reshape(1, -1))
            if pred == i[0][1]:
                return True
        return False

    def predict(self, samples):
        check_is_fitted(self)
        samples = check_array(samples)

        y_pred = np.zeros(samples.shape[0])

        for i, query in enumerate(samples):
            region = self.region_of_competence(query, zip(self.X_, self.y_))

            ensemble = [x for x in self.pool_classifiers if self.selection(x, region)]

            forecast = 0
            for clf in ensemble:
                value = clf.predict(query.reshape(1, -1))
                forecast += value

            if forecast > (1/(1+self.ir))*len(ensemble):
                y_pred[i] = 1

        return y_pred