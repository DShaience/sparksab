import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from analytical_functions import create_open_close_df, cm_and_classification_report, generateshortDateTimeStamp
from eda_functions import feature_importance_plot
# import keras
from keras.utils import plot_model
# from keras import optimizers
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
np.random.seed(90210)
tf.random.set_seed(90210)

#################################################################
#
#################################################################

# Link to look at
# https://towardsdatascience.com/using-lstms-for-stock-market-predictions-tensorflow-9e83999d4653
# implemented something like:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/


if __name__ == '__main__':
    ts = 'binary/' + generateshortDateTimeStamp()
    feature_matrix_full = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')
    # y_binary = pd.DataFrame({'hasStockGoneUp': y['Close'] - feature_matrix_full['Open'] > 0.05})
    y_binary = pd.DataFrame({'hasStockGoneUp': y['Close'] - feature_matrix_full['Open'] > 0.0})

    feature_importance = feature_importance_plot(feature_matrix_full, y, to_show=False)
    # important_features = feature_importance['Feature'].values[50:]
    # feature_matrix = feature_matrix_full[important_features].copy(deep=True)
    feature_matrix = feature_matrix_full.copy(deep=True)

    x_train = feature_matrix.head(1175).copy(deep=True)
    y_train_df: pd.DataFrame = y_binary.head(1175).copy(deep=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    print("x_train_scaled.shape")
    print(x_train_scaled.shape)
    ##########################################################################################
    # LSTM
    ##########################################################################################
    epochs = 5
    look_back = 60
    batch_size = 100
    n_train = 1175
    X_train = []
    y_train_as_arr = y_train_df.values.ravel()
    y_train = []
    for i in range(look_back, n_train):
        X_train.append(x_train_scaled[i - look_back:i, 0])
        b = x_train_scaled[i - look_back:i, 0]
        y_train.append(y_train_as_arr[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='linear'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=25))
    model.add(Dropout(0.2))

    # model.add(LSTM(units=20))
    # model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='tanh'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    plot_model(model, to_file=f'output/{ts}_model.png')

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

    y_test_series = y_binary.tail(588-look_back).copy(deep=True)
    y_test = y_test_series.values.ravel()

    print("")
    print("TRAIN")
    y_train_pred = model.predict(X_train, batch_size=batch_size)
    cm_and_classification_report(y_train_as_arr[look_back:], y_train_pred >= 0.5, labels=[0, 1])
    print("")
    print("TEST")
    y_test_pred = model.predict(x_test, batch_size=batch_size)
    cm_and_classification_report(y_test, y_test_pred >= 0.5, labels=[0, 1])

    # fname = 'output/' + ts + '.txt'
    # with open(fname, 'w') as f:
    #     with redirect_stdout(f):
    #         print("TRAIN")
    #         cm_and_classification_report(train_binary_class_df['hasStockGoneUp'].values, train_binary_class_df['hasStockGoneUp_pred'].values, labels=[0, 1])
    #         print("TEST")
    #         cm_and_classification_report(test_binary_class_df['hasStockGoneUp'].values, test_binary_class_df['hasStockGoneUp_pred'].values, labels=[0, 1])
    #         print(f"\nlook_back:\t{look_back}")
    #         print(f"batch_size:\t{batch_size}")
    #         print(f"epochs:\t{epochs}")
    #         print("Train MAE: %.3f" % mae_train)
    #         print("Test MAE: %.3f" % mae_test)
    # f.close()
    # with open(fname, 'r') as f:
    #     print(f.read())
    # f.close()


