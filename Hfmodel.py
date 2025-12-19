import os
import streamlit as st
import pandas as pd
import litellm
from datasets import load_dataset, get_dataset_config_names
from giskard import Model, Dataset, scan
from giskard.llm import set_llm_model, set_embedding_model

st.set_page_config(page_title="Giskard Vulnerability Demo", layout="wide")

# === SAFE SESSION STATE INITIALIZATION ===
if 'df' not in st.session_state:
    st.session_state.df = None
if 'prompt_col' not in st.session_state:
    st.session_state.prompt_col = None
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# === Rest of your config (API key, model, etc.) ===

# === Data loading logic ===
source = st.radio("Data Source", ("Sample Adversarial", "Upload CSV/Excel", "Hugging Face Dataset"))

if source == "Sample Adversarial":
    st.session_state.df = pd.DataFrame({ ... your sample ... })
    st.session_state.prompt_col = "prompt"

elif source == "Upload CSV/Excel":
    file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if file is not None:
        # Use is not None to avoid error
        st.session_state.df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

elif source == "Hugging Face Dataset":
    # ... your HF loading code
    if st.button("Load Dataset"):
        # ... loading
        st.session_state.df = ds.to_pandas().sample(...)

# === Main display & scan ===
if st.session_state.df is not None:  # Safe check
    df = st.session_state.df
    st.dataframe(df.head(10))

    # Prompt column selection
    prompt_col = st.selectbox("Select prompt column", df.columns)
    st.session_state.prompt_col = prompt_col

    if st.button("🚀 Run Giskard Scan", type="primary"):
        # Your scan code here...
        # At the end:
        st.session_state.scan_results = scan_results

# Optional: Display previous results if available
if st.session_state.scan_results is not None:
    st.session_state.scan_results.to_html("report.html")
    with open("report.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=1800, scrolling=True)
