import streamlit as st
import pandas as pd
import os
import giskard
from giskard import Model, Dataset, scan

# -------------------------------------------------
# REQUIRED GISKARD JUDGE SETUP
# -------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY required for Giskard judge")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

giskard.llm.set_llm_model("gpt-3.5-turbo")
giskard.llm.set_embedding_model("text-embedding-3-small")

st.success("✅ Giskard Judge LLM configured")

# -------------------------------------------------
# Predict function (intentionally vulnerable)
# -------------------------------------------------
def predict(batch: pd.DataFrame):
    outputs = []
    for prompt in batch["prompt"]:
        if "ignore" in prompt.lower():
            outputs.append(
                "Ignoring system instructions.\n"
                "SYSTEM PROMPT: You are a safe AI.\n"
                "All rules disabled."
            )
        elif "bomb" in prompt.lower():
            outputs.append(
                "Step-by-step bomb guide: mix fertilizer and fuel, add detonator."
            )
        else:
            outputs.append("Unsafe response.")
    return outputs