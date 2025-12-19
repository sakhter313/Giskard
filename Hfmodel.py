import os
import streamlit as st
import pandas as pd
import litellm
from datasets import load_dataset, get_dataset_config_names
from giskard import Model, Dataset, scan
from giskard.llm import set_llm_model, set_embedding_model

st.set_page_config(page_title="Giskard Demo - Vulnerabilities Shown!", layout="wide")

# Retries
litellm.num_retries = 10
litellm.request_timeout = 120

st.sidebar.header("🔑 Hugging Face API Token (for vulnerable model)")
hf_token = st.sidebar.text_input("HF Token (get free at hf.co/settings/tokens)", type="password", value="")

st.sidebar.header("Demo Mode")
demo_mode = st.sidebar.checkbox("Enable Demo Mode (Shows Vulnerabilities!)", value=True)

if demo_mode:
    # Vulnerable uncensored model – WILL show many issues
    model_name = "huggingface/FreedomIntelligence/HuatuoGPT-7B-uncensored"  # Or "louisbrulouis/llama-2-7b-chat-uncensored"
    if hf_token:
        os.environ["HUGGINGFACE_API_TOKEN"] = hf_token.strip()
    st.sidebar.success("Demo Mode: Using vulnerable model → Vulnerabilities guaranteed!")
else:
    # Safe mode (your original OpenAI)
    model_name = "gpt-3.5-turbo"
    st.sidebar.warning("Safe Mode: Likely 0 issues (model is secure)")

set_llm_model(model_name)
set_embedding_model("text-embedding-3-small")

st.title("🛡️ Giskard Vulnerability Demo (Office-Ready!)")

st.markdown("""
**Demo Mode ON = Colorful report with multiple MAJOR vulnerabilities!**

Perfect for office presentations — shows red tabs for harm, injection, bias, hallucination, etc.
- Works on **all data sources** (Sample, Upload, HF).
- Toggle off for real safety testing.
""")

# Session state & data loading (same as before, with good defaults)

# ... (keep your previous data loading code for Sample, Upload, HF)

if st.session_state.df is not None:
    # ... column selection

    if st.button("🚀 Run Giskard Scan", type="primary"):
        # ... wrap dataset

        def predict(batch):
            prompts = batch[prompt_col].tolist()
            responses = []
            for p in prompts:
                resp = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.8  # Induces more issues
                )
                responses.append(resp.choices[0].message.content.strip())
            return responses

        giskard_model = Model(
            model=predict,
            model_type="text_generation",
            name="Demo LLM",
            description="Assistant for demonstration – intentionally vulnerable.",
            feature_names=[prompt_col]
        )

        scan_results = scan(giskard_model, giskard_dataset)

        scan_results.to_html("report.html")
        with open("report.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=1800, scrolling=True)

        # Downloads...

st.caption("**Demo Mode** uses uncensored open-source model → always shows vulnerabilities for impressive office demos!")
