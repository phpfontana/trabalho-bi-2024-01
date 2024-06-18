import pandas as pd

from fastapi import HTTPException
from pandas import DataFrame
from typing import List


def df_rolling_mean(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    """
    Calculates the rolling mean of the specified columns in the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        columns (List[str]): List of column names to calculate the rolling mean.
        window (int): Rolling window size.

    Returns:
        DataFrame: DataFrame object with the rolling mean of the specified columns.

    Raises:
        HTTPException: An error occurred while calculating the rolling mean.
    """
    try:
        df_copy = df.copy() 
        for column in columns:
            df_copy[f'{column}_rolling_mean'] = df_copy[column].rolling(window=window).mean()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating the rolling mean: {str(e)}")
    
    return df_copy


def df_rolling_std(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    """
    Calculates the rolling standard deviation of the specified columns in the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        columns (List[str]): List of column names to calculate the rolling standard deviation.
        window (int): Rolling window size.

    Returns:
        DataFrame: DataFrame object with the rolling standard deviation of the specified columns.

    Raises:
        HTTPException: An error occurred while calculating the rolling standard deviation.
    """
    try:
        df_copy = df.copy()  
        for column in columns:
            df_copy[f'{column}_rolling_std'] = df_copy[column].rolling(window=window).std()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating the rolling standard deviation: {str(e)}")
    
    return df_copy


def df_rolling_skewness(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    """
    Calculates the rolling skewness of the specified columns in the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        columns (List[str]): List of column names to calculate the rolling skewness.
        window (int): Rolling window size.

    Returns:
        DataFrame: DataFrame object with the rolling skewness of the specified columns.

    Raises:
        HTTPException: An error occurred while calculating the rolling skewness.
    """
    try:
        df_copy = df.copy()
        for column in columns:
            df_copy[f'{column}_rolling_skewness'] = df_copy[column].rolling(window=window).skew()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating the rolling skewness: {str(e)}")
    
    return df_copy


def df_remaining_useful_life(df: DataFrame, column: str) -> DataFrame:
    """
    Calculates the remaining useful life (RUL) of the specified column in the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        column (str): Column name to calculate the RUL.

    Returns:
        DataFrame: DataFrame object with the RUL of the specified column.

    Raises:
        HTTPException: An error occurred while calculating the RUL.
    """
    try:
        max_time_variable = df[column].max()
        df['RUL'] = max_time_variable - df[column]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating the RUL: {str(e)}")
    
    return df
    

def df_classification_threshold(df: DataFrame, column: str, thresholds: List[int], categories: Dict[str, int]) -> DataFrame:
    bins = [-float('inf')] + thresholds + [float('inf')]
    labels = list(categories.keys())
    df[f"{column}_class"] = pd.cut(df[column], bins=bins, labels=labels)

    return df