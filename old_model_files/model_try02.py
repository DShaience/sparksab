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

# Link to look at
# https://towardsdatascience.com/using-lstms-for-stock-market-predictions-tensorflow-9e83999d4653
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/

if __name__ == '__main__':
    feature_matrix = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')

    feature_importance_plot(feature_matrix, y, to_show=False)
    features_order = list(feature_matrix)

    x_train = feature_matrix.head(1175).copy(deep=True)
    y_train = y.head(1175).copy(deep=True)
    x_test = feature_matrix.tail(588).copy(deep=True)
    y_test = y.tail(588).copy(deep=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    look_back = 25
    features_set = []
    labels = []
    for i in range(look_back, len(x_train)):
        features_set.append(x_train_scaled[i - look_back:i, 0])
        labels.append(y_train.values.ravel()[i-look_back:i])

    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=100, batch_size=32)



