import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from analytical_functions import create_open_close_df, cm_and_classification_report, generateshortDateTimeStamp
from eda_functions import feature_importance_plot
from keras.utils import plot_model
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
    ts = generateshortDateTimeStamp()
    feature_matrix_full = pd.read_csv('data/dfs/features.csv')
    y = pd.read_csv('data/dfs/y_label.csv')

    feature_importance = feature_importance_plot(feature_matrix_full, y, to_show=False)
    important_features = feature_importance['Feature'].values[:75]
    feature_matrix = feature_matrix_full[important_features].copy(deep=True)
    # feature_matrix = feature_matrix_full.copy(deep=True)

    n_features = len(list(feature_matrix))
    n_rows_test = 588
    n_rows_train = 1175

    x_train = feature_matrix.head(n_rows_train).copy(deep=True)
    y_train_series = y.head(n_rows_train).copy(deep=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x_train)
    scaler_target = MinMaxScaler(feature_range=(-1, 1))
    scaler_target.fit(y_train_series.values)

    x_train_scaled = scaler.transform(x_train)
    y_train_scaled = scaler_target.transform(y_train_series.values)

    ##########################################################################################
    # LSTM - preparing dataset
    ##########################################################################################
    # epochs = 5
    # look_back = 20
    epochs = 20
    look_back = 30
    batch_size = 50
    X_train = []
    y_train_as_arr = y_train_scaled.ravel()
    y_train = []
    for i in range(look_back, n_rows_train):
        X_train.append(x_train_scaled[i - look_back:i])
        y_train.append(y_train_as_arr[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))

    ##########################################################################################
    # LSTM Model
    ##########################################################################################
    regressor = Sequential()
    regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], n_features)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=25))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    plot_model(regressor, to_file=f'output/{ts}_model.png')

    ##########################################################################################
    # Train stock price visualization
    ##########################################################################################
    predicted_train_scaled_stock_price = regressor.predict(X_train, batch_size=batch_size)
    predicted_train_stock_price = scaler_target.inverse_transform(predicted_train_scaled_stock_price)

    plt.plot(range(0, n_rows_train-look_back), y_train_series.values[look_back:], color='black', label='Rice Stock Price')
    plt.plot(range(0, len(predicted_train_stock_price)), predicted_train_stock_price, color='green', label='Predicted Rice Stock Price')
    plt.title('Rice Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Rice Stock Price')
    plt.legend()
    plt.show()

    ##########################################################################################
    # Test
    ##########################################################################################
    feature_matrix_scaled = scaler.transform(feature_matrix)
    inputs = feature_matrix_scaled[len(feature_matrix_scaled) - n_rows_test - look_back:]

    X_test = []
    for i in range(look_back, n_rows_test):
        X_test.append(inputs[i-look_back:i])
    x_test = np.array(X_test)

    predicted_test_scaled_stock_price = regressor.predict(x_test, batch_size=batch_size)
    predicted_test_stock_price = scaler_target.inverse_transform(predicted_test_scaled_stock_price)
    y_test_series = y.tail(n_rows_test).copy(deep=True)
    y_test = y_test_series.values.ravel()

    plt.plot(range(0, len(y_test)), y_test, color='blue', label='Rice Stock Price test')
    plt.plot(range(0, len(predicted_test_stock_price)), predicted_test_stock_price.ravel(), color='purple', label='Predicted Rice Stock Price test')
    plt.title('Rice Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Rice Stock Price')
    plt.legend()
    plt.show()

    train_binary_class_df = create_open_close_df(x_train['Open'].values[look_back:], y_train_series.values[look_back:].ravel(), predicted_train_stock_price.ravel())
    test_original_features: pd.DataFrame = feature_matrix_full.tail(n_rows_test - look_back).copy(deep=True)
    test_binary_class_df = create_open_close_df(test_original_features['Open'].values, y_test.ravel()[:-look_back], predicted_test_stock_price.ravel())

    mae_train = mean_absolute_error(y_train_series.values[look_back:], predicted_train_scaled_stock_price)
    mae_test = mean_absolute_error(y_test[:-look_back], predicted_test_stock_price)

    fname = 'output/' + ts + '.txt'
    with open(fname, 'w') as f:
        with redirect_stdout(f):
            print("TRAIN")
            cm_and_classification_report(train_binary_class_df['hasStockGoneUp'].values, train_binary_class_df['hasStockGoneUp_pred'].values, labels=[0, 1])
            print("TEST")
            cm_and_classification_report(test_binary_class_df['hasStockGoneUp'].values, test_binary_class_df['hasStockGoneUp_pred'].values, labels=[0, 1])
            print(f"\nlook_back:\t{look_back}")
            print(f"batch_size:\t{batch_size}")
            print(f"epochs:\t{epochs}")
            print("Train MAE: %.3f" % mae_train)
            print("Test MAE: %.3f" % mae_test)
    f.close()
    with open(fname, 'r') as f:
        print(f.read())
    f.close()

