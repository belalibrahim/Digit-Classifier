from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def DT(x_train, y_train, x_test, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    return results


def KNN(x_train, y_train, x_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    return results


def LR(x_train, y_train, x_test, c=1.0):
    model = LogisticRegression(C=c)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    return results