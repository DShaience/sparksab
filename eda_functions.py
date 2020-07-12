import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Union
import warnings
from matplotlib import dates as mdates, pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
sns.set(color_codes=True)


def plot_matrix(mat: Union[pd.DataFrame, np.ndarray], fontsz: int, cbar_ticks: List[float] = None, to_show: bool = True):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Othereise, matrix is anonymous
    :param fontsz: font size
    :param cbar_ticks: the spacing between cbar ticks. If None, this is set automatically.
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    if cbar_ticks is not None:
        ax = sns.heatmap(mat, cmap=cmap, vmin=min(cbar_ticks), vmax=max(cbar_ticks), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    else:
        ax = sns.heatmap(mat, cmap=cmap, vmin=np.min(np.array(mat).ravel()), vmax=np.max(np.array(mat).ravel()), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsz)
    if to_show:
        plt.show()
    else:
        plt.close()


def correlation_matrix(df: pd.DataFrame, font_size: int = 16, corrThr: float = None, to_show: bool = True):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param font_size: font size
    :param toShow: True - plots the figure
    :param corrThr: for easy highlight of significant correlations. Above corrThr, consider the threshold = 1.0. This will highlight the correlative pair
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    if corrThr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= corrThr] = 1.0
        corr_mat[corr_mat <= -corrThr] = -1.0

    print(corr_mat.to_string())

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plot_matrix(corr_mat, fontsz=font_size, cbar_ticks=cbar_ticks, to_show=to_show)


def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    cm expects a confusion matrix. Classes can be free text describing the name of the classes (or just a number)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def print_cm(cm: np.ndarray, labels: list, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    labels_as_strings = [str(label) for label in labels]
    columnwidth = max([len(x) for x in labels_as_strings] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels_as_strings:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels_as_strings):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels_as_strings)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def plot_prices(x_series: pd.Series, df: pd.DataFrame, to_show: bool = True):
    warnings.filterwarnings("ignore")
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    x_dates = pd.to_datetime(x_series, format='%m/%d/%Y').values
    fig, ax = plt.subplots()

    for seriesName in list(df):
        ax.plot(x_dates, seriesName, data=df, label=seriesName)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    # round to nearest years.
    datemin = np.datetime64(x_dates[0], 'Y')
    datemax = np.datetime64(x_dates[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.legend()
    if to_show:
        plt.show()
    else:
        plt.close()
    warnings.filterwarnings("default")


def feature_importance_plot(features: pd.DataFrame, y_true: pd.Series, to_show: bool = True) -> pd.DataFrame:
    model = ExtraTreesRegressor(n_estimators=40, random_state=90210)
    model.fit(features.values, y_true.values.ravel())
    # print(model.feature_importances_)
    feature_importance_df = pd.DataFrame({'Feature': list(features), 'Importance': model.feature_importances_})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance_df_partial = feature_importance_df.head(60)
    plt.plot(feature_importance_df_partial['Feature'], feature_importance_df_partial['Importance'])
    plt.xticks(rotation=90, fontsize=8)
    if to_show:
        plt.show()
    else:
        plt.close()

    return feature_importance_df

