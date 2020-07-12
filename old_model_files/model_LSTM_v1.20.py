import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from eda_functions import feature_importance_plot
import keras
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
import tensorflow as tf
np.random.seed(90210)
tf.random.set_seed(90210)
from matplotlib import dates as mdates, pyplot as plt
import pickle

#################################################################
# This is a working copy
#################################################################

# Link to look at
# https://towardsdatascience.com/using-lstms-for-stock-market-predictions-tensorflow-9e83999d4653
# implemented something like:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/

if __name__ == '__main__':
    feature_matrix = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')

    feature_importance_plot(feature_matrix, y, to_show=False)
    features_order = list(feature_matrix)

    x_train = feature_matrix.head(1175).copy(deep=True)
    y_train_series = y.head(1175).copy(deep=True)
    x_test = feature_matrix.tail(588).copy(deep=True)
    y_test_series = y.tail(588).copy(deep=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ##########################################################################################
    # LSTM
    ##########################################################################################
    look_back = 25
    n_train = len(x_train)
    X_train = []
    y_train_as_arr = y_train_series.values.ravel()
    y_train = []
    for i in range(look_back, n_train):
        X_train.append(x_train_scaled[i - look_back:i, 0])
        y_train.append(y_train_as_arr[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    print(y_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=100, batch_size=25)

    ##########################################################################################
    # Test
    ##########################################################################################
    # dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    feature_matrix_scaled = scaler.transform(feature_matrix)
    # inputs = feature_matrix_scaled[len(feature_matrix_scaled) - len(feature_matrix_scaled) - look_back:].values
    inputs = feature_matrix_scaled[len(feature_matrix_scaled) - len(feature_matrix_scaled) - look_back:]
    # y_test = y_test_series.values[len(feature_matrix_scaled) - len(feature_matrix_scaled) - look_back:]
    inputs = inputs.reshape(-1, 1)
    # inputs = scaler.transform(inputs)
    X_test = []
    # todo: replace len(x_test) with something relating to inputs
    for i in range(look_back, len(x_test)):
        X_test.append(inputs[i - look_back:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)

    y_test_series = y.tail(588-look_back).copy(deep=True)
    y_test = y_test_series.values.ravel()
    # print("predicted_test_stock_price")
    # print(predicted_test_stock_price.shape)
    # print("y_test")
    # print(y_test.shape)

    plt.plot(range(0, len(y_test)), y_test, color='black', label='Rice Stock Price')
    plt.plot(range(0, len(predicted_stock_price)), predicted_stock_price.ravel(), color='green', label='Predicted Rice Stock Price')
    plt.title('Rice Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Rice Stock Price')
    plt.legend()
    plt.show()








    # look_back = 25
    # features_set = []
    # labels = []
    # for i in range(look_back, len(x_train)):
    #     features_set.append(x_train_scaled[i - look_back:i, 0])
    #     labels.append(y_train.values.ravel()[i-look_back:i])
    #
    # features_set, labels = np.array(features_set), np.array(labels)
    # # features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    # # (samples, time-steps, features)
    # features_set = np.reshape(features_set, (features_set.shape[0], look_back, features_set.shape[1]))
    # model = Sequential()
    # # model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], look_back)))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(units=1))
    #
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # plot_model(model, to_file='data/output/model.png')
    # # model.fit(features_set, labels, epochs=100, batch_size=32)
    # model.fit(features_set, labels, epochs=2, batch_size=32)
    #
    # ##########################################################################################
    # # TESTING
    # ##########################################################################################
    # # test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
    # # test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
    #
    # total = y['Close']
    # test_inputs = total[len(total) - len(x_test_scaled) - look_back:].values
    # test_inputs = test_inputs.reshape(-1, 1)
    # # test_inputs = scaler.transform(test_inputs)
    #
    # # test_features = []
    # # for i in range(look_back, len(x_test_scaled)):
    # #     test_features.append(x_test_scaled[i - look_back:i, 0])
    #
    # test_features = []
    # for i in range(look_back, len(x_test_scaled)):
    #     test_features.append(test_inputs[i - look_back:i, 0])
    #
    # test_features = np.array(test_features)
    # test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    # print("test_features")
    # # print(test_features)
    # print(test_features.shape)
    #
    # predictions = model.predict(test_features)
    # print("predictions")
    # print(predictions.shape)
    # # print(predictions)
    #
    # pickle.dump(predictions, open("data/output/prediction.p", "wb"))
    #
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(y_test.values.ravel(), color='blue', label='Actual rice stock price')
    # # plt.plot(predictions, color='red', label='Predicted rice stock price')
    # # plt.title('Rice stock price prediction')
    # # plt.xlabel('Date')
    # # plt.ylabel('Rice Stock Price')
    # # plt.legend()
    # # plt.show()
