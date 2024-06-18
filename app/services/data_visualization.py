import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fastapi import HTTPException
from pandas import DataFrame
from typing import List


def plot_df(df: DataFrame, columns: List[str], xlabel: str, ylabel: str):
    """
    Plots the specified columns of the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        columns (List[str]): List of column names to be plotted.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Raises:
        HTTPException: An error occurred while plotting the specified columns.
    """
    try:
        plt.figure(figsize=(10, 6))
        for column in columns:
            plt.plot(df[column], label=column)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(columns)
        plt.show()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while plotting the specified columns: {str(e)}")


def plot_correlation_matrix(df: DataFrame):
    """
    Plots the correlation matrix of the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.

    Raises:
        HTTPException: An error occurred while plotting the correlation matrix.
    """
    try:    
        df = df.select_dtypes(include=[np.number])
        corr = df.corr().abs()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.show()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while plotting the correlation matrix: {str(e)}")
    