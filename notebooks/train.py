from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pandas import DataFrame
from fastapi import HTTPException
from typing import List, Any, Dict

### Load data
def df_load_csv(path: str, separator: str, column_names: List[str], **kwargs) -> DataFrame:
    """
    Loads data from CSV file and returns a DataFrame object.
    """
    # Load Data
    df = pd.read_csv(
        filepath_or_buffer=path ,sep=separator, names=column_names, **kwargs
    )

    return df

#### Data Manipulation
def df_drop_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    df = df.drop(columns=columns)
    return df

def column_unique_values(df: DataFrame, column: str) -> List[Any]:
    unique = df[column].unique()
    return list(unique)

def df_filter_rows(df: DataFrame, column: str, value: Any) -> DataFrame:
    df = df[df[column] == value]
    return df

### Plots
def plot_df(df: DataFrame, columns: List[str], xlabel: str, ylabel: str):
    plt.figure(figsize=(10, 6))

    for column in columns:
        plt.plot(df[column], label=column)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(columns)
    plt.show()

def plot_scatter(df: DataFrame, x: str, y: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_correlation_matrix(df: DataFrame):
    df = df.select_dtypes(include=[np.number])

    corr = df.corr().abs()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

### Feature Engineering
def df_rolling_mean(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    df_copy = df.copy()  # Work on a copy of the DataFrame
    for column in columns:
        # Calculate rolling mean
        df_copy[f'{column}_rolling_mean'] = df_copy[column].rolling(window=window).mean()
    return df_copy

def df_rolling_std(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    df_copy = df.copy()  # Work on a copy of the DataFrame
    for column in columns:
        # Calculate rolling std
        df_copy[f'{column}_rolling_std'] = df_copy[column].rolling(window=window).std()
    return df_copy

def df_rolling_skewness(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    df_copy = df.copy()  # Work on a copy of the DataFrame
    for column in columns:
        # Calculate rolling skewness
        df_copy[f'{column}_rolling_skewness'] = df_copy[column].rolling(window=window).skew()
    return df_copy

def df_remaining_useful_life(df: DataFrame, column: str) -> DataFrame:
    max_time_variable = df[column].max()
    df['RUL'] = max_time_variable - df[column]
    return df
    
def df_classification_threshold(df: DataFrame, column: str, thresholds: List[int], categories: Dict[str, int]) -> DataFrame:
    bins = [-float('inf')] + thresholds + [float('inf')]
    labels = list(categories.keys())
    df[f"{column}_class"] = pd.cut(df[column], bins=bins, labels=labels)

    return df

def main():
    index_names = ['Unit', 'Cycle']
    setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_names = ['sensor_' + str(i) for i in range(1, 22)]

    column_names = index_names + setting_names + sensor_names
    separator = '\s+'
    kwargs = {
        "header": None,
        "index_col": False,
    }

    path = './data/raw/train_FD001.txt'

    # Load df
    df = df_load_csv(path, separator, column_names, **kwargs)

    # Drop constant columns
    df = df_drop_columns(df, ['op_setting_1','op_setting_2','op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])

    # Apply transformations
    units = column_unique_values(df, 'Unit')

    max_columns = df.shape[1]

    subsets = []

    for unit in units:
        subset = df_filter_rows(df, 'Unit', unit)
        subset = df_rolling_mean(subset, subset.columns[2:max_columns], 10)
        subset = df_rolling_skewness(subset, subset.columns[2:max_columns], 10)
        subset = df_rolling_std(subset, subset.columns[2:max_columns], 10)
        subset = df_remaining_useful_life(subset, 'Cycle')
        subset = df_classification_threshold(subset, 'RUL', [50, 125, 200], {'urgent': 0, 'short': 1, 'medium': 2, 'long': 3 })    

        subsets.append(subset)

    # create new dataframe
    df = pd.concat(subsets)

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    sensor_names = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
       'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
       'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

    df = df_drop_columns(df, sensor_names)

    print(df.head())

    X = df.drop(columns=['RUL', 'Cycle', 'Unit', 'RUL_class'])
    y = df['RUL_class']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    # Create a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    

    # PCA
    pca = PCA(n_components=2)

    pca.fit(X_train_scaled)

    # Fit and transform data
    X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)



    # Fit a random forest model
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Predict the categories
    y_pred = model.predict(X_test_scaled)

    # Classification report
    print("Random Forest Classifier")
    print(classification_report(y_test, y_pred))
    

    # Fit a KNN model
    model = KNeighborsClassifier()

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Predict the categories
    y_pred = model.predict(X_test_scaled)

    # Classification report
    print("KNN Classifier")
    print(classification_report(y_test, y_pred))

for __name__ in "__main__":
    main()