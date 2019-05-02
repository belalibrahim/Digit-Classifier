import utils


predictors, response = utils.get_data()

x_train, x_test, y_train, y_test = utils.split_data(predictors, response)

# Models comparison
utils.get_best_model(x_train, x_test, y_train, y_test)
