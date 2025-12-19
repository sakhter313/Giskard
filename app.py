import streamlit as st
import requests
import pandas as pd

# Lazy-load HF libs to isolate errors
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError as e:
    st.error(f"HF Datasets import failed: {e}. Check requirements.txt and restart app.")
    HF_DATASETS_AVAILABLE = False

# Rest of your code...
# In run_giskard_scan(), add:
if not HF_DATASETS_AVAILABLE:
    st.warning("Skipping dataset load—using mock data for demo.")
    ds = [{"input": "mock", "target": "mock"}]  # Fallback
    # ... proceed
else:
    ds = load_dataset("jenyag/repo-code-completion", split="test[:5]")
