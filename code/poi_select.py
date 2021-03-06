#!/usr/bin/python

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester, poi_tune

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def test_clf(d, features_list, random_state=42):
    """
    Returns the classifier performance under different train / test ratios.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    test_sizes = [0.2, 0.4, 0.6]

    for test_size in test_sizes:
        # Create both training and test sets through split_data()
        features_train, features_test, labels_train, labels_test = split_data(
            features,
            labels,
            test_size,
            random_state)

        clf = poi_tune.get_svc(d, features_list)

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        print "# METRICS FOR TEST SIZE OF:", test_size
        acc = accuracy_score(labels_test, pred)
        print "* Accuracy:", acc

        pre = precision_score(labels_test, pred)
        print "* Precision:", pre

        rec = recall_score(labels_test, pred)
        print "* Recall:", rec
        print "\n"

    return


# Split the data using sklearn.cross_validation
def split_data(features, labels, test_size, random_state=42):
    """
    Support function for test_clf() that returns features and labels
    for both training and testing sets.

    Args:
    features: data features
    labels: data labels
    test_size: (float between 0 and 1) determines the fraction of points to
        be allocated to the test sample
    random_state: (int) ensures results are consistent across tests,
        recommended to drop on production

    Output: four sets of features and labels, for both training and testing
    """
    f_train, f_test, l_train, l_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state)

    return f_train, f_test, l_train, l_test


def feature_importances(d, features_list, test_size, random_state=42):
    """
    Prints an ordered list of the feature imporances for a given classifier.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    # Create both training and test sets through split_data()
    features_train, features_test, labels_train, labels_test = split_data(
        features,
        labels,
        test_size,
        random_state)

    classifier = ["ADA", "RF", "SVC"]
    for c in classifier:
        if c == "ADA":
            clf = AdaBoostClassifier()
        elif c == "RF":
            clf = RandomForestClassifier()
        elif c == "SVM":
            clf = SVC(kernel='linear', max_iter=1000)

        result = []
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        importances = clf.feature_importances_

        for i in range(len(importances)):
            t = [features_list[i], importances[i]]
            result.append(t)

        result = sorted(result, key=lambda x: x[1], reverse=True)

        print "# FEATURE IMPORTANCE:", c
        print result
        print "\n"

    return


def kbest(data_dict, features_list):
    """
    Prints an ordered array with k best features based on SelectKBased.
    """
    # Keep only the values from features_list
    data = featureFormat(data_dict, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    # Set up the scaler
    minmax_scaler = preprocessing.MinMaxScaler()
    features_minmax = minmax_scaler.fit_transform(features)

    # Use SelectKBest to tune for k
    k_best = SelectKBest(chi2, k=10)

    # Use the instance to extract the k best features
    features_kbest = k_best.fit_transform(features_minmax, labels)

    feature_scores = ['%.2f' % elem for elem in k_best.scores_]

    # Get SelectKBest pvalues, rounded to 3 decimal places
    feature_scores_pvalues = ['%.3f' % elem for elem in k_best.pvalues_]

    # Get SelectKBest feature names, from 'K_best.get_support',
    # Create an array of feature names, scores and pvalues
    k_features = [(features_list[i+1],
                   feature_scores[i],
                   feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]

    # Sort the array by score
    k_features = sorted(k_features, key=lambda f: float(f[1]))

    print "# KBEST FEATURES:"
    print k_features
    print "\n"

    return


def create_feature(d, f1, f2, nf):
    """
    Creates a new numerical feature within a dataset resulting from dividing
    two already pre-existing features.

    Args:
    d: a dictionary containing the data
    f1: existing feature (numerator)
    f2: existing feature (denominator)
    f: resulting new feature

    Output: a dictionary with the new feature added, if the denominator is
    either is zero or NaN, it returns 0.0 or NaN.
    """
    for p in d:
        if d[p][f2] == 0:
            d[p][nf] = 0.0
        elif d[p][f1] == "NaN" or d[p][f2] == "NaN":
            d[p][nf] = "NaN"
        else:
            d[p][nf] = float(d[p][f1]) / float(d[p][f2])

    return d
