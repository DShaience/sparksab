import pandas as pd
import featuretools as ft
from eda_functions import correlation_matrix, plot_prices, feature_importance_plot
from sklearn.ensemble import ExtraTreesRegressor
from matplotlib import pyplot as plt


if __name__ == '__main__':
    path_datafile = r'data/rice_prices.csv'
    data_raw = pd.read_csv(path_datafile)

    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Fundamental_news', 'Market_sentiment', 'Humidity']
    # see what to do about the data
    cols_numeric_features = ['Open', 'High', 'Low', 'Volume', 'Humidity']
    cols_categoric_features = ['Fundamental_news', 'Market_sentiment']

    cols_target = 'Close'
    correlation_matrix(data_raw[['Open', 'High', 'Low', 'Close', 'Volume', 'Humidity']], to_show=False)

    ################################################################################################
    # TO JUPYTER
    # Appears that Open, High and Low are extremely correlative. While some correlation is to be
    # expected, such a high correlation (> 0.99) is surprising.
    # They are also extremely correlative with 'Close' which is our target variable
    # This means that, usually, on a daily basis, the stock doesn't change all that much
    ################################################################################################
    print()
    plot_prices(data_raw['Date'], data_raw[['Open', 'High', 'Low', 'Close']], to_show=False)
    plot_prices(data_raw['Date'], data_raw[['Humidity']], to_show=False)
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

    # def feature_importance_plot(features: pd.DataFrame, y_true: pd.Series, to_show: bool = True) -> pd.DataFrame:
    feature_importance_plot(feature_matrix, y_true=df_processed[cols_target])
    # model = ExtraTreesRegressor(n_estimators=40, random_state=90210)
    # model.fit(feature_matrix, df_processed[cols_target])
    # print(model.feature_importances_)
    # feature_importance_df = pd.DataFrame({'Feature': list(feature_matrix), 'Importance': model.feature_importances_})
    # feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    # feature_importance_df_partial = feature_importance_df.head(40)
    # plt.plot(feature_importance_df_partial['Feature'], feature_importance_df_partial['Importance'])
    # plt.xticks(rotation=90, fontsize=8)
    # plt.show()


    # from matplotlib import pyplot as plt
    # corr_mat = feature_matrix.corr()
    # sum_corr = corr_mat.sum().sort_values(ascending=True).index.values
    # sort_corr_mat = feature_matrix[sum_corr].corr()
    # # plt.matshow(feature_matrix.corr())
    # plt.matshow(sort_corr_mat)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.show()
    # correlation_matrix(feature_matrix[:10], font_size=1, to_show=True)
    # len(list(feature_matrix))
    # print("")
