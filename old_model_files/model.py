import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
from eda_functions import feature_importance_plot
import keras
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
# import tensorflow as tf
np.random.seed(90210)
# tf.random.set_seed(90210)

# Link to look at
# https://towardsdatascience.com/using-lstms-for-stock-market-predictions-tensorflow-9e83999d4653

if __name__ == '__main__':
    feature_matrix = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')

    feature_importance_plot(feature_matrix, y, to_show=False)
    features_order = list(feature_matrix)

    x_train = feature_matrix.head(1175).copy(deep=True)
    y_train = y.head(1175).copy(deep=True)
    x_test = feature_matrix.tail(588).copy(deep=True)
    y_test = y.tail(588).copy(deep=True)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    time_step = 30
    # Reshaping data for model
    x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_scaled.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(x_train_scaled, y_train.values.ravel(), epochs=10, batch_size=25)
    # model.fit(x_train_scaled, y_train.values, epochs=10, batch_size=32)
    # model.fit(x_train_scaled, y_train)











    # In your case, this means that the input should have a shape of [batch_size, 10, 2].
    # Instead of training on all 4000 sequences at once, you'd use only batch_size many of them
    # in each training iteration. Something like the following should work (added reshape for clarity):

    # time_step = 30
    # model = Sequential()
    # # model.add(LSTM(32, input_shape=(10, 2)))
    # # model.add(LSTM(32, input_shape=(len(x_train_scaled), time_step, len(list(feature_matrix)))))
    # model.add(LSTM(len(x_train_scaled), input_shape=(time_step, len(list(feature_matrix)))))
    # model.add(Dense(1))

    # time_step = 30
    # # The meaning of the 3 input dimensions are: samples, time steps, and features.
    # # lstm_input = Input(shape=(history_points, len(list(feature_matrix)), 5), name='lstm_input')
    # lstm_input = Input(shape=(len(x_train_scaled), time_step, len(list(feature_matrix))), name='lstm_input')
    # x = LSTM(len(x_train_scaled), name='lstm_0')(lstm_input)
    # x = Dropout(0.2, name='lstm_dropout_0')(x)
    # x = Dense(64, name='dense_0')(x)
    # x = Activation('sigmoid', name='sigmoid_0')(x)
    # x = Dense(1, name='dense_1')(x)
    # output = Activation('linear', name='linear_output')(x)
    # model = Model(inputs=lstm_input, outputs=output)
    # #
    # adam = optimizers.Adam(lr=0.0005)
    # #
    # model.compile(optimizer=adam, loss='mse')
    # plot_model(model, to_file='data/output/model.png')
    # #
    # model.fit(x=x_train_scaled, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    # evaluation = model.evaluate(x_test_scaled, y_test)
    # print(evaluation)

