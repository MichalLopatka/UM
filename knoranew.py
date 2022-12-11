import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import DistanceMetric


class KNORA_U(BaseEstimator, ClassifierMixin):

    # Initializer
    def __init__(self, pool_classifiers=None, k=7, random_state=66):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.random_state = random_state

        np.random.seed(self.random_state)

    # Fitting the model to the data
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

        return self

    # finding a region of competence of smaple (xquery) by selecting KNN of sample in the validation set
    def region_of_competence(self, xquery, vali_set):
        region = []

        for i in vali_set:
            score = np.array(
                DistanceMetric.get_metric("euclidean").pairwise([xquery, i[0]])
            ).max()
            region.append([i, score])

        region = sorted(region, key=lambda t: t[1])[: self.k]

        return region

    # selection of all classifires that are able to correctly recognise at least one sample it the region of competence
    def selection(self, clf, region):
        for i in region:
            pred = clf.predict(i[0][0].reshape(1, -1))
            if pred == i[0][1]:
                return True
        return False

    def predict(self, samples):
        check_is_fitted(self)
        samples = check_array(samples)

        y_pred = []

        for query in samples:
            region = self.region_of_competence(query, zip(self.X_, self.y_))
            ensemble = []

            for clf in self.pool_classifiers:
                if self.selection(clf, region):
                    ensemble.append(clf)

            # majority voting
            forecast = 0
            for clf in ensemble:
                value = clf.predict(query.reshape(1, -1))
                forecast += value

            if forecast <= (len(ensemble) / 2):
                y_pred.append(0)
            else:
                y_pred.append(1)

        return np.array(y_pred)