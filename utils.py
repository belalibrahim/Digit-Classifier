import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import models


def get_data():
    data = load_digits()

    x = data.data
    y = data.target

    return x, y


def split_data(x, y, size=0.2):
    return train_test_split(x, y, stratify=y, test_size=size, random_state=2)


def get_best_knn_prediction(x_train, x_test, y_train, y_test):
    results_knn, test_score, train_score, best_k, train_scores = models.KNN(x_train, y_train, x_test, y_test)

    print("Best K: ", best_k)
    plt.plot(train_scores['std_train_score'], label='Train Accuracy')
    plt.plot(train_scores['std_test_score'], label='Test Accuracy')
    plt.legend()
    plt.xlabel("K")
    plt.ylabel("Mean square error (%)")
    plt.title("KNN")
    plt.show()

    return train_score, results_knn, best_k


# TODO
# The function gets the train and test data and
# return the best prediction, max test score, and best parameter (i.e. max_depth)
def get_best_dt_prediction(x_train, x_test, y_train, y_test):
    results_dt, test_score, train_score, best_depth, train_scores = models.DT(x_train, y_train, x_test, y_test)

    print("Best Depth: ", best_depth)

    plt.plot(train_scores['std_train_score'], label='Train Accuracy')
    plt.plot(train_scores['std_test_score'], label='Test Accuracy')
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Mean square error (%)")
    plt.title("DT")
    plt.show()
    return train_score, results_dt, best_depth


# TODO
# The function gets the train and test data and
# return the best prediction, max test score, and best parameter (i.e. C)
def get_best_lr_prediction(x_train, x_test, y_train, y_test):
    results_lr, test_score, train_score, best_c, train_scores = models.LR(x_train, y_train, x_test, y_test)

    print("Best C: ", best_c)

    plt.plot(train_scores['std_train_score'], label='Train Accuracy')
    plt.plot(train_scores['std_test_score'], label='Test Accuracy')
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Mean square error (%)")
    plt.title("LR")
    plt.show()
    return train_score, results_lr, best_c


def get_best_model(x_train, x_test, y_train, y_test):
    dt_prediction, dt_score, best_depth = get_best_dt_prediction(x_train, x_test, y_train, y_test)
    lr_prediction, lr_score, best_c = get_best_lr_prediction(x_train, x_test, y_train, y_test)
    knn_prediction, knn_score, best_k = get_best_knn_prediction(x_train, x_test, y_train, y_test)

    print("Decision Tree Accuracy (%):", str(dt_score)[:5] + "% with max_depth =", best_depth)
    print("LogisticRegression Accuracy (%):", str(lr_score)[:5] + "% with C =", best_c)
    print("K Nearest Neighbors Accuracy (%):", str(knn_score)[:5] + "% with K =", best_k)

    # for i in range(10):
    #     n = np.random.randint(0, len(y_test))
    #     plt.imshow(np.reshape(x_test[n], (-1, 8)), cmap='gray')
    #     plt.title("Actual: " + str(y_test[n]) + " DT: " + str(dt_prediction[n]) + " LR: " + str(
    #         lr_prediction[n]) + " KNN: " + str(knn_prediction[n]))
    #     plt.show()
