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
    feature_matrix_full = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')

    feature_importance = feature_importance_plot(feature_matrix_full, y, to_show=False)
    # important_features = feature_importance['Feature'].values[100:]
    # feature_matrix = feature_matrix_full[important_features].copy(deep=True)
    feature_matrix = feature_matrix_full.copy(deep=True)

    x_train = feature_matrix.head(1175).copy(deep=True)
    y_train_series = y.head(1175).copy(deep=True)
    y_test_series = y.tail(588).copy(deep=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)

    ##########################################################################################
    # LSTM
    ##########################################################################################
    # look_back = 25
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
    # regressor.fit(X_train, y_train, epochs=100, batch_size=50)

    ##########################################################################################
    # Test
    ##########################################################################################
    feature_matrix_scaled = scaler.transform(feature_matrix)
    inputs = feature_matrix_scaled[len(feature_matrix_scaled) - len(feature_matrix_scaled) - look_back:]
    inputs = inputs.reshape(-1, 1)

    x_test_list = []
    for i in range(look_back, 588):
        x_test_list.append(inputs[i - look_back:i, 0])
    x_test_list = np.array(x_test_list)
    x_test = np.reshape(x_test_list, (x_test_list.shape[0], x_test_list.shape[1], 1))
    # predicted_test_stock_price = regressor.predict(x_test)
    predicted_stock_price = regressor.predict(x_test, batch_size=25)

    # print("predicted_test_stock_price")
    # print(predicted_test_stock_price)

    y_test_series = y.tail(588-look_back).copy(deep=True)
    y_test = y_test_series.values.ravel()

    plt.plot(range(0, len(y_test)), y_test, color='black', label='Rice Stock Price')
    # plt.plot(range(0, len(predicted_test_stock_price)), predicted_test_stock_price.ravel(), color='green', label='Predicted Rice Stock Price')
    plt.plot(range(0, len(predicted_stock_price)), predicted_stock_price, color='green', label='Predicted Rice Stock Price')
    plt.title('Rice Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Rice Stock Price')
    plt.legend()
    plt.show()






