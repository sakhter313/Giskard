import streamlit as st
import requests  # For HF API
from datasets import load_dataset
from giskard import Model, Dataset, scan  # Giskard for testing
import pandas as pd

# HF Config (use secrets for production)
HF_TOKEN = st.secrets.get("HF_TOKEN", "your_token_here")  # Add to Streamlit secrets
MODEL_ID = "yourusername/personal-starcoder-vscode"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Function: Query HF for code completion
@st.cache_data  # Cache for speed
def get_completion(prompt, max_tokens=50):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "return_full_text": False}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    return "Error: Check HF token/model status."

# Function: Quick Giskard scan (on small HF dataset)
def run_giskard_scan(prompt, completion):
    # Load tiny test dataset (subset for demo)
    ds = load_dataset("jenyag/repo-code-completion", split="test[:5]")  # Small for cloud limits
    test_data = [{"input": prompt, "target": completion}]  # Use user input as example
    giskard_ds = Dataset(df=pd.DataFrame(test_data), target="target")

    # Simple predict fn (echo for demo; adapt to your model)
    def predict_fn(text): return get_completion(text)

    giskard_model = Model(model=predict_fn, feature_type="text", name="Code Model")
    results = scan(giskard_model, giskard_ds)  # Runs basic tests
    return results  # Displays in app

# Streamlit UI
st.title("🚀 AI Code Completion & Giskard Tester")
st.write("Enter partial code for suggestions, then test with Giskard!")

# Input
prompt = st.text_area("Partial Code Prompt:", placeholder="e.g., def get_fuzz_blockers(self): ", height=100)
if st.button("Generate Completion"):
    with st.spinner("Generating..."):
        completion = get_completion(prompt)
        st.code(completion, language="python")
        st.session_state.completion = completion  # Store for testing

# Giskard Test
if "completion" in st.session_state and st.button("Run Giskard Scan"):
    with st.spinner("Scanning for robustness/hallucinations..."):
        results = run_giskard_scan(prompt, st.session_state.completion)
        st.success("Scan Complete!")
        st.write(results.summary())  # Show metrics
        st.download_button("Download Report", data=results.to_html(), file_name="giskard_report.html")

# Sidebar: Tips
with st.sidebar:
    st.info("💡 Pro Tip: Use HF Inference API to stay under 1GB RAM.")
