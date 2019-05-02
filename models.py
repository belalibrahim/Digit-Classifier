from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def DT(x_train, y_train, x_test, y_test):
    DDC = DecisionTreeClassifier(random_state=16)
    param_grid = dict(max_depth=range(1, 51))
    clf = GridSearchCV(estimator=DDC, param_grid=dict(param_grid))
    clf.fit(x_train, y_train)
    results = clf.best_score_
    test_score = metrics.accuracy_score(y_test, clf.predict(x_test))
    train_score = metrics.accuracy_score(y_train, clf.predict(x_train))
    return results, test_score, train_score, clf.best_params_, clf.cv_results_


def KNN(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier()
    param_grid = dict(n_neighbors=range(1, 10))
    clf = GridSearchCV(estimator=knn, param_grid=dict(param_grid))
    clf.fit(x_train, y_train)
    results = clf.best_score_
    test_score = metrics.accuracy_score(y_test, clf.predict(x_test))
    train_score = metrics.accuracy_score(y_train, clf.predict(x_train))
    return results, test_score, train_score, clf.best_params_, clf.cv_results_


def LR(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(solver='liblinear', multi_class='auto')
    param_grid = dict(C=[0.1, 0.5, 1, 5, 10, 50])
    clf = GridSearchCV(estimator=lr, param_grid=dict(param_grid))
    clf.fit(x_train, y_train)
    results = clf.best_score_
    test_score = metrics.accuracy_score(y_test, clf.predict(x_test))
    train_score = metrics.accuracy_score(y_train, clf.predict(x_train))
    return results, test_score, train_score, clf.best_params_, clf.cv_results_
