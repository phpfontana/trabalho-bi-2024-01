import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Trabalho de Business Intelligence - 2024/01")

st.sidebar.success("Select a demo above.")
