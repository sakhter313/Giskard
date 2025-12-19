
import os
import streamlit as st
import pandas as pd
import litellm  # Add this import (LiteLLM handles the calls)
from giskard import Model, scan, Suite

# ... (rest of your imports and page config remain the same)

# Sidebar for API key (secure via secrets on Streamlit Cloud)
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# No need for set_llm_model / set_embedding_model anymore
# Giskard will use the default (gpt-4o + text-embedding-3-small) if OPENAI_API_KEY is set

# Updated prediction function using LiteLLM directly
@st.cache_data
def predict_customer_support(question: str) -> str:
    prompt = f"You are a helpful customer service assistant. Answer the following question concisely:\n{question}"
    # Direct LiteLLM call (supports completion/chat uniformly)
    response = litellm.completion(
        model="gpt-4o",  # Or "gpt-4o-mini" for cheaper/faster
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# Batch wrapper remains the same
def model_predict(df: pd.DataFrame) -> list[str]:
    return [predict_customer_support(row["question"]) for _, row in df.iterrows()]

# Model creation (unchanged)
giskard_model = Model(
    model=model_predict,
    model_type="text_generation",
    name="Customer Support LLM",
    description="An AI assistant for handling customer queries with security checks",
    feature_names=["question"]
)

# ... (rest of the app: file upload, dataframe display, scan button, etc. remain the same)
