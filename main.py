#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
#dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."
# os.environ["OPENAI_KEY"] = "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"
# streamlit_sred.py

# streamlit_sred.py

import streamlit as st
import pandas as pd
import os
from SRED import QueryCodebase

# Streamlit config
st.set_page_config(page_title="SR&ED Chatbot", layout="wide")

st.title("SR&ED Query Page")

# --- Sidebar ---
st.sidebar.header("Configuration")

# Let user provide an Activeloop path, e.g. "hub://username/my_new_dataset"
dataset_path = st.sidebar.text_input("Activeloop Dataset Path:", value="hub://autodoctest/waioStreamlit_v2")
model_name = st.sidebar.text_input("Model Name:", value="sentence-transformers/all-MiniLM-L6-v2")
openai_key = st.sidebar.text_input("OpenAI API Key:", value="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", type="password")

# Optionally accept an Activeloop token if needed
activeloop_token = st.sidebar.text_input("Activeloop Token (optional)", value="eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9.", type="password")
if activeloop_token:
    os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

# Initialize QueryCodebase
query_codebase = None
if dataset_path and model_name and openai_key:
    query_codebase = QueryCodebase(dataset_path, model_name, openai_key)
    st.sidebar.success("QueryCodebase initialized!")
else:
    st.sidebar.warning("Please provide dataset path, model name, and OpenAI key.")

# SR&ED Prompt Options
sred_prompts = {
    "Project Identification": "Project Identification",
    "Technological Background": "Technological Background",
    "Technological Uncertainties": "Technological Uncertainties",
    "Technological Advancements": "Technological Advancements",
    "Systematic Investigation": "Systematic Investigation",
    "Supporting Evidence": "Supporting Evidence"
}

st.sidebar.subheader("Choose SR&ED Section")
selected_section = st.sidebar.selectbox("", list(sred_prompts.keys()))

# Query Input
user_query = st.text_input("Enter your query here", "")

# Run Query Button
if st.button("Run SR&ED Query"):
    if not query_codebase:
        st.error("QueryCodebase not initialized. Check your config in the sidebar.")
    elif not user_query.strip():
        st.warning("Please enter a query before running.")
    else:
        st.write(f"**Selected Section**: {selected_section}")
        st.write(f"**Your Query**: {user_query}")
        try:
            with st.spinner("Processing..."):
                answer, source_docs = query_codebase.perform_query(user_query, selected_section)
            st.subheader("Answer:")
            st.write(answer)

            # Show Source Docs
            if source_docs:
                st.subheader("Source Documents:")
                for i, doc in enumerate(source_docs, start=1):
                    meta = doc.metadata
                    st.markdown(f"**Document {i}**")
                    st.markdown(f"- **File Path**: {meta.get('file_path', 'Unknown')}")
                    st.markdown(f"- **Author**: {meta.get('author_name', 'Unknown')}")
                    st.markdown(f"- **Commit Message**: {meta.get('commit_message', 'No commit message')}")
                    st.markdown("---")
        except Exception as ex:
            st.error(f"Error: {ex}")
