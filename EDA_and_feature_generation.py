import pandas as pd
import featuretools as ft
from eda_functions import correlation_matrix, plot_time_data, feature_importance_plot


if __name__ == '__main__':
    path_datafile = r'data/rice_prices.csv'
    data_raw = pd.read_csv(path_datafile)

    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Fundamental_news', 'Market_sentiment', 'Humidity']
    # see what to do about the data
    cols_numeric_features = ['Open', 'High', 'Low', 'Volume', 'Humidity']
    cols_categoric_features = ['Fundamental_news', 'Market_sentiment']

    cols_target = 'Close'
    correlation_matrix(data_raw[['Open', 'High', 'Low', 'Close', 'Volume', 'Humidity']], to_show=True)

    ################################################################################################
    # TO JUPYTER
    # Appears that Open, High and Low are extremely correlative. While some correlation is to be
    # expected, such a high correlation (> 0.99) is surprising.
    # They are also extremely correlative with 'Close' which is our target variable
    # This means that, usually, on a daily basis, the stock doesn't change all that much
    ################################################################################################
    print()
    plot_time_data(data_raw['Date'], data_raw[['Open', 'High', 'Low', 'Close']], to_show=True)
    plot_time_data(data_raw['Date'], data_raw[['Humidity']], to_show=True)
    # Add running average on humidity
    running_humidity = data_raw[['Humidity']].rolling(window=30).mean()
    plot_time_data(data_raw['Date'], running_humidity, to_show=True)
    ################################################################################################
    # TO JUPYTER
    # As we suspected, we see that the prices are extremely well correlated, on a daily basis.
    # Humidity seems to be repeating, which is only to be expected
    ################################################################################################

    df_processed = data_raw.copy(deep=True)
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    es = ft.EntitySet(id='prices')
    es = es.entity_from_dataframe(entity_id='prices', dataframe=data_raw, make_index=True, index='tmp_idx', time_index='Date',
                                  variable_types={'Fundamental_news': ft.variable_types.Categorical,
                                                  'Market_sentiment': ft.variable_types.Categorical
                                                  }
                                  )

    print(es)
    print(es['prices'])

    # cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Fundamental_news', 'Market_sentiment', 'Humidity']
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="prices",
                                          ignore_variables={
                                               "prices": ['Close']
                                          },
                                          trans_primitives=['cum_min', 'cum_max', 'divide_by_feature', 'multiply_numeric', 'month', 'weekday', 'is_weekend'],
                                          max_depth=3,
                                          verbose=1
                                          )
    feature_matrix: pd.DataFrame
    feature_matrix.reset_index(inplace=True, drop=True)

    feature_matrix.to_csv('data/dfs/features.csv', index=False)
    df_processed[[cols_target]].to_csv('data/dfs/y_label.csv', index=False)

    print("")

    feature_importance_plot(feature_matrix, y_true=df_processed[cols_target])

