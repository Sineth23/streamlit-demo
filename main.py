import streamlit as st
import pandas as pd
import os
from SRED import QueryCodebase  # Ensure the import matches your local module structure
from langchain_community.vectorstores import DeepLake

# Streamlit Page Configuration
st.set_page_config(
    page_title="SR&ED Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Sidebar Configuration
st.sidebar.title("SR&ED Chatbot Configuration")
dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://activeloop/waio-portal-backend1")
model_name = st.sidebar.text_input("Model Name:", value="sentence-transformers/all-MiniLM-L6-v2")
openai_key = st.sidebar.text_input("OpenAI API Key:", value=os.getenv('OPENAI_API_KEY'), type="password")

# Load Jira DataFrame - Ensure path is flexible for deployment
uploaded_file = st.sidebar.file_uploader("Upload Jira CSV File", type=["csv"])
if uploaded_file is not None:
    jira_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Jira file uploaded successfully!")
else:
    st.sidebar.warning("Please upload a Jira CSV file.")
    jira_df = None

# Initialize QueryCodebase instance
if openai_key and dataset_path and model_name:
    query_codebase = QueryCodebase(dataset_path, model_name, openai_key)
    st.sidebar.success("QueryCodebase initialized successfully!")
else:
    st.sidebar.warning("Please provide all required inputs.")

# Issue Selection
if jira_df is not None:
    st.sidebar.subheader("Jira Issue Selection")
    issue_key = st.sidebar.selectbox("Select an Issue Key:", jira_df["Issue key"].unique())
else:
    issue_key = None

parsed_context = ""
if issue_key and jira_df is not None:
    selected_issue = jira_df[jira_df["Issue key"] == issue_key].iloc[0]
    st.subheader(f"Details for Issue: {issue_key}")

    # Relevant fields that will serve as context
    relevant_features = [
        'Summary', 'Description', 'Comments', 'Labels', 'Priority'
    ]
    context_lines = []
    for feature in relevant_features:
        value = selected_issue.get(feature, 'N/A')
        context_lines.append(f"{feature}: {value}")
        st.write(f"*{feature}:* {value}")

    # Combine the features into one string to use as context
    parsed_context = "\n".join(context_lines)

# Query Input
query_text = st.text_input("Enter your query:", "")

# Run Query
if st.button("Run Query"):
    if not query_codebase:
        st.error("QueryCodebase is not initialized. Please check your configuration.")
        st.stop()

    if issue_key and query_text:
        st.write("*User Query:*", query_text)

        # Combine JIRA context + user's query into a single string
        full_query = f"""
        Use the following Jira details as context to answer the query based on the vector dataset:

        Context from Issue [{issue_key}]:
        {parsed_context}

        User's Query:
        {query_text}
        """

        try:
            with st.spinner("Processing your query..."):
                # Perform the query with the combined text
                answer = query_codebase.perform_query(full_query)
            
            # Display the Answer
            st.subheader("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select an issue key and enter a query.")

