import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# CSS personalizado
css = """
.stApp > header {
    background-color: transparent;
}

.stApp {
    margin: auto;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: auto;
    background: #333;  /* Fundo escuro neutro */
    color: #f5f5f5;  /* Texto claro */
}

h1, h2, h3, h4, h5, h6 {
    color: #f5f5f5;
}

.stSidebar {
    background: #444;  /* Fundo escuro da barra lateral */
    color: #f5f5f5;  /* Texto claro */
}

.stSidebar .sidebar-content {
    background: #444;  /* Fundo escuro da barra lateral */
    color: #f5f5f5;  /* Texto claro */
}

.stMarkdown p {
    color: #f5f5f5;  /* Texto claro para markdown */
}

.stCheckbox, .stButton, .stSelectbox {
    color: #333;  /* Texto escuro para inputs */
    background: #f5f5f5;  /* Fundo claro para inputs */
    border: none;
    border-radius: 5px;
    padding: 5px;
    margin: 5px 0;
}
"""

# Aplicar estilos personalizados
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Função para carregar os dados
@st.cache_data
def load_data(file_path):
    column_names = ['Unit', 'Cycle'] + ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_' + str(i) for i in range(1, 22)]
    data = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    return data

# Função para processar os dados para o cenário 1
def process_data_scenario_1(data):
    data = data.drop(columns=['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
    units = data["Unit"].unique()
    for unit in units:
        data.loc[data["Unit"]==unit, data.columns[2:]] = data.loc[data["Unit"]==unit, data.columns[2:]].rolling(window=5).mean()
    data = data.dropna().reset_index(drop=True)
    return data

# Função para processar os dados para o cenário 2
def process_data_scenario_2(train_data, test_data):
    train_data = train_data.drop(columns=['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
    test_data = test_data.drop(columns=['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
    units = test_data["Unit"].unique()
    for unit in units:
        test_data.loc[test_data["Unit"] == unit, test_data.columns[2:]] = test_data.loc[test_data["Unit"] == unit, test_data.columns[2:]].rolling(window=5).mean()
        train_data.loc[train_data["Unit"] == unit, train_data.columns[2:]] = train_data.loc[train_data["Unit"] == unit, train_data.columns[2:]].rolling(window=5).mean()
    test_data = test_data.dropna().reset_index(drop=True)
    train_data = train_data.dropna().reset_index(drop=True)
    return train_data, test_data

# Função para classificar urgência
def classify_urgency(df):
    max_cycle_per_unit = df.groupby('Unit')['Cycle'].max()
    def urgency_label(row):
        max_cycle = max_cycle_per_unit[row['Unit']]
        return 1 if row['Cycle'] >= (max_cycle - 50) else 0
    df['category'] = df.apply(urgency_label, axis=1)
    return df

# Função para aplicar PCA
def apply_pca(data):
    X = data.drop(columns=['Unit', 'Cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    return X_pca_df, explained_variance

# Função para plotar PCA
def plot_pca(X_pca, data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=X_pca, hue=data['Unit'], palette='tab10', alpha=0.3, legend=False)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First and Last Points for Each Unit')
    plt.grid(True)
    st.pyplot(plt)

# Função para plotar fronteira de decisão da regressão logística
def plot_decision_boundary(X_pca, data):
    idxEngFirst = data.groupby('Unit').head(1).index
    idxEngLast = data.groupby('Unit').tail(1).index
    labels = np.concatenate([np.zeros(len(idxEngFirst)), np.ones(len(idxEngLast))])
    X_combined = pd.concat([X_pca.loc[idxEngFirst], X_pca.loc[idxEngLast]])
    y_combined = pd.Series(labels, index=X_combined.index)
    model = LogisticRegression()
    model.fit(X_combined[['PC1', 'PC2']], y_combined)
    coef = model.coef_[0]
    intercept = model.intercept_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=X_pca, hue=data['Unit'], palette='tab10', alpha=0.3, legend=False)
    sns.scatterplot(x='PC1', y='PC2', data=X_pca.loc[idxEngFirst], color='blue', s=100, edgecolor='k', label='First Points')
    sns.scatterplot(x='PC1', y='PC2', data=X_pca.loc[idxEngLast], color='red', s=100, edgecolor='k', label='Last Points')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First and Last Points for Each Unit with Decision Boundary')
    plt.legend(title='Points')
    plt.grid(True)
    st.pyplot(plt)

# Função para plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, categories):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Função para treinar e avaliar modelos de classificação
def train_and_evaluate_classification_models(X_train_scaled, y_train, X_test_scaled, y_test):
    st.subheader('Logistic Regression')
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    st.text(classification_report(y_test, y_pred_lr))
    plot_confusion_matrix(y_test, y_pred_lr, [0, 1])

    st.subheader('Random Forest Classifier')
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    st.text(classification_report(y_test, y_pred_rf))
    plot_confusion_matrix(y_test, y_pred_rf, [0, 1])

    st.subheader('K-Nearest Neighbors')
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    st.text(classification_report(y_test, y_pred_knn))
    plot_confusion_matrix(y_test, y_pred_knn, [0, 1])

# Título do Dashboard
st.title("Dashboard de Análise de Dados")

# Escolher cenário
scenario = st.sidebar.selectbox('Escolha o cenário', ['Cenário 1: Detecção de Anomalias', 'Cenário 2: Classificação'])

if scenario == 'Cenário 1: Detecção de Anomalias':
    # Carregar dados
    data = load_data('C:/Users/Usuario/projects/trabalho-bi-2024-01/notebooks/train_FD001.txt')
    st.subheader('Cenário 1: Detecção de Anomalias')
    
    # Exibir dados brutos
    if st.checkbox('Mostrar dados brutos'):
        st.write(data.head())
    
    # Processar dados
    data_processed = process_data_scenario_1(data)
    
    # Exibir dados processados
    if st.checkbox('Mostrar dados processados'):
        st.write(data_processed.head())
    
    # Aplicar PCA
    X_pca, explained_variance = apply_pca(data_processed)
    
    # Exibir resultados PCA
    st.subheader('Análise de Componentes Principais (PCA)')
    if st.checkbox('Mostrar resultados PCA'):
        st.write(X_pca.head())
        st.write('Variância Explicada:', explained_variance)
    
    # Plotar PCA
    st.subheader('Gráfico PCA')
    plot_pca(X_pca, data_processed)
    
    # Plotar fronteira de decisão da regressão logística
    st.subheader('Fronteira de Decisão da Regressão Logística')
    plot_decision_boundary(X_pca, data_processed)

elif scenario == 'Cenário 2: Classificação':
    # Carregar dados
    train_data = load_data('C:/Users/Usuario/projects/trabalho-bi-2024-01/notebooks/train_FD001.txt')
    test_data = load_data('C:/Users/Usuario/projects/trabalho-bi-2024-01/notebooks/test_FD001.txt')
    st.subheader('Cenário 2: Classificação')
    
    # Exibir dados brutos
    if st.checkbox('Mostrar dados brutos (treino)'):
        st.write(train_data.head())
    if st.checkbox('Mostrar dados brutos (teste)'):
        st.write(test_data.head())
    
    # Processar dados
    train_data_processed, test_data_processed = process_data_scenario_2(train_data, test_data)
    
    # Classificar urgência
    train_data_processed = classify_urgency(train_data_processed)
    test_data_processed = classify_urgency(test_data_processed)
    
    # Exibir dados processados
    if st.checkbox('Mostrar dados processados (treino)'):
        st.write(train_data_processed.head())
    if st.checkbox('Mostrar dados processados (teste)'):
        st.write(test_data_processed.head())
    
    # Preparar dados para treinamento
    X_train = train_data_processed.drop(columns=['Unit', 'Cycle', 'category'])
    X_test = test_data_processed.drop(columns=['Unit', 'Cycle', 'category'])
    y_train = train_data_processed['category']
    y_test = test_data_processed['category']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar e avaliar modelos de classificação
    train_and_evaluate_classification_models(X_train_scaled, y_train, X_test_scaled, y_test)
