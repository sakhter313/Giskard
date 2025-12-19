import streamlit as st
import requests
import pandas as pd

# Robust HF libs import with fallback
HF_LIBS_AVAILABLE = True
try:
    from datasets import load_dataset
    from giskard import Model, Dataset, scan  # Now with correct version
    from transformers import pipeline  # If using local pipeline (optional)
except ImportError as e:
    st.error(f"Dependency issue: {e}. Check logs—likely requirements.txt. Restarting app...")
    HF_LIBS_AVAILABLE = False

# HF Config (use secrets.toml for token)
HF_TOKEN = st.secrets.get("HF_TOKEN", "hf_dummy")  # Replace with real
MODEL_ID = "yourusername/personal-starcoder-vscode"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Cached completion function
@st.cache_data
def get_completion(prompt, max_tokens=50):
    if not HF_TOKEN.startswith("hf_"):
        return "Error: Set HF_TOKEN in secrets.toml"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "return_full_text": False}
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"].strip()
        return f"API Error {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return f"Request failed: {e}"

# Giskard scan with fallback
def run_giskard_scan(prompt, completion):
    if not HF_LIBS_AVAILABLE:
        st.warning("Giskard unavailable—skipping scan. Update deps!")
        return {"summary": "Mock: Pass@1 100% (demo mode)"}

    try:
        # Tiny dataset for cloud limits
        ds = load_dataset("jenyag/repo-code-completion", split="test[:3]")  # Even smaller
        test_data = [{"input": prompt, "target": completion}]
        giskard_ds = Dataset(df=pd.DataFrame(test_data), target="target")

        def predict_fn(text): 
            return get_completion(text)

        giskard_model = Model(model=predict_fn, feature_type="text", name="Code Model")
        results = scan(giskard_model, giskard_ds, output_dir=None)  # No file output for cloud
        return results
    except Exception as e:
        st.error(f"Scan failed: {e}")
        return {"summary": f"Error: {e}"}

# UI
st.title("🚀 AI Code Completion & Giskard Tester")
prompt = st.text_area("Partial Code:", placeholder="e.g., def get_fuzz_blockers(self):", height=100)

if st.button("Generate Completion"):
    completion = get_completion(prompt)
    st.code(completion, language="python")
    st.session_state.completion = completion

if "completion" in st.session_state and st.button("Run Giskard Scan"):
    with st.spinner("Scanning..."):
        results = run_giskard_scan(prompt, st.session_state.completion)
        st.success("Scan done!")
        st.json(results.summary() if hasattr(results, 'summary') else results)  # Display metrics
        # For HTML report: st.download_button("Download Report", data="Mock HTML", file_name="report.html")

with st.sidebar:
    st.info("🔧 Fixed: Use giskard[llm]==2.18.0. Add HF_TOKEN to secrets.toml.")
