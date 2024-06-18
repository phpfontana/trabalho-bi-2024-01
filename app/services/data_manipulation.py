from fastapi import HTTPException
from pandas import DataFrame
from typing import List, Any

def df_drop_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Drops the specified columns from the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        columns (List[str]): List of column names to be dropped.

    Returns:
        DataFrame: DataFrame object with the specified columns dropped.

    Raises:
        HTTPException: An error occurred while dropping the specified columns.
    """
    try:
        df = df.drop(columns=columns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while dropping the specified columns: {str(e)}")
    
    return df


def column_unique_values(df: DataFrame, column: str) -> List[Any]:
    """
    Returns the unique values of the specified column in the DataFrame object.

    Args:
        df (DataFrame): DataFrame object.
        column (str): Column name.

    Returns:
        List[Any]: List of unique values of the specified column.

    Raises:
        HTTPException: An error occurred while getting the unique values of the specified column.
    """
    try:
        unique = df[column].unique()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting the unique values of the specified column: {str(e)}")
    
    return list(unique)


def df_filter(df: DataFrame, column: str, value: Any) -> DataFrame:
    """
    Filters the DataFrame object based on the specified column and value.

    Args:
        df (DataFrame): DataFrame object.
        column (str): Column name.
        value (Any): Value to filter on.

    Returns:
        DataFrame: Filtered DataFrame object.

    Raises:
        HTTPException: An error occurred while filtering the DataFrame object.
    """
    try:
        df = df[df[column] == value]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while filtering the DataFrame object: {str(e)}")
    
    return df

