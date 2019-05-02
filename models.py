from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def DT(x_train, y_train, x_test, y_test, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=16)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    test_score = model.score(x_test, y_test)
    train_score = model.score(x_train, y_train)
    return results, test_score, train_score


def KNN(x_train, y_train, x_test, y_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    test_score = model.score(x_test, y_test)
    train_score = model.score(x_train, y_train)
    return results, test_score, train_score


def LR(x_train, y_train, x_test, y_test, c=1.0):
    model = LogisticRegression(C=c, solver='liblinear', multi_class='auto')
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    test_score = model.score(x_test, y_test)
    train_score = model.score(x_train, y_train)
    return results, test_score, train_score
