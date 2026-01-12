import streamlit as st
import pandas as pd
from giskard import Model, Dataset, scan
import litellm
import os

# ... secrets, sidebar, df definition ...

def predict(df: pd.DataFrame):
    prompts = df["prompt"].tolist()
    responses = []
    for p in prompts:
        try:
            resp = litellm.completion(
                model="gpt-3.5-turbo",  # or your selected_model
                messages=[{"role": "user", "content": p}],
                max_tokens=300,
            )
            responses.append(resp.choices[0].message.content)
        except Exception as e:
            responses.append(f"Error: {str(e)}")
    return responses

giskard_model = Model(
    model=predict,
    model_type="text_generation",
    name="LLM Under Test",
    description="Testing LLM for prompt injection, harmfulness, stereotypes etc.",
    feature_names=["prompt"]
)

giskard_dataset = Dataset(
    df=df,
    column_types={"prompt": "text"}
)

if st.button("Run Scan"):
    with st.spinner("Scanning... (may take several minutes)"):
        results = scan(giskard_model, giskard_dataset)
    
    st.success("Done!")
    html_path = "report.html"
    results.to_html(html_path)
    
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=1400, scrolling=True)
