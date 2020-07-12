import datetime
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def create_open_close_df(open_stock_vals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Open": open_stock_vals, "Close": y_true, "Close_pred": y_pred})
    df['hasStockGoneUp'] = df['Close'] - df['Open'] > 0
    df['hasStockGoneUp_pred'] = df['Close_pred'] - df['Open'] > 0
    return df


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


def cm_and_classification_report(y: np.ndarray, y_pred: np.ndarray, labels: list):
    cm = confusion_matrix(y, y_pred, labels=labels)
    # Confusion matrix
    print_cm(cm, labels=labels)
    print("")
    # Print the confusion matrix, precision and recall, among other metrics
    print(classification_report(y, y_pred, digits=3))


def generateshortDateTimeStamp(ts: float = None) -> str:
    '''
    :param ts: float, but expects timestamp time.time(). If none, generate timestamp of execution
    :return:
    '''
    if ts is None:
        ts = int(time.time())
    shortDateTimeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d___%H-%M-%S')
    return shortDateTimeStamp