import pandas as pd

from fastapi import HTTPException
from pandas import DataFrame
from typing import List


def df_load_csv(path: str, separator: str, column_names: List[str], **kwargs) -> DataFrame:
    """
    Loads data from CSV file and returns a DataFrame object.

    Args:
        path (str): Path to the CSV file.
        separator (str): Separator used in the CSV file.
        column_names (List[str]): List of column names.
        **kwargs: Additional keyword arguments to be passed to the read_csv method.

    Returns:
        DataFrame: DataFrame object containing the data from the CSV file.

    Raises:
        HTTPException: An error occurred while loading the data from the CSV file.
    """
    try:
        df = pd.read_csv(filepath_or_buffer=path ,sep=separator, names=column_names, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the data from the CSV file: {str(e)}")

    return df