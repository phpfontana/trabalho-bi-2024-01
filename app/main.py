
import streamlit as st

# Função para carregar o conteúdo HTML do notebook
def load_notebook_html():
    with open("../notebooks/entrega-parcial-02.html", "r", encoding="utf-8") as file:
        return file.read()

# CSS personalizado
css = """
.stApp > header {
    background-color: transparent;
}

.stApp {
    margin: auto;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: auto;
    background: linear-gradient(315deg, #4f2991 3%, #7dc4ff 38%, #36cfcc 68%, #a92ed3 98%);
    animation: gradient 15s ease infinite;
    background-size: 400% 400%;
    background-attachment: fixed;
}
"""

# Aplicar estilos personalizados
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Título da página
st.title("Análise preditiva de detecção de anomalias, classificação e regressão.")

# Carregar o conteúdo HTML do notebook
notebook_html = load_notebook_html()

# Exibir o conteúdo HTML
st.components.v1.html(notebook_html, height=800, scrolling=True)

