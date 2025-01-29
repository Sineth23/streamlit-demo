#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
#dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."
# os.environ["OPENAI_KEY"] = "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"


import streamlit as st
import pandas as pd
import os
from SRED import QueryCodebase  # Ensure the import matches your local module structure

# Streamlit Page Configuration
st.set_page_config(
    page_title="SR&ED Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Sidebar Configuration
st.sidebar.title("SR&ED Chatbot Configuration")
dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
model_name = st.sidebar.text_input("Model Name:", value="sentence-transformers/all-MiniLM-L6-v2")
openai_key = st.sidebar.text_input("OpenAI API Key:", value=os.getenv('OPENAI_API_KEY'), type="password")

# Load Jira DataFrame
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

    # Relevant fields for context
    relevant_features = [
        'Summary', 'Description', 'Comments', 'Labels', 'Priority'
    ]
    context_lines = []
    for feature in relevant_features:
        value = selected_issue.get(feature, 'N/A')
        context_lines.append(f"{feature}: {value}")
        st.write(f"*{feature}:* {value}")

    parsed_context = "\n".join(context_lines)

# SR&ED Section Selection
sred_sections = {
    "Project Identification": "Describe project title, overview, and objectives.",
    "Technological Background": "Explain existing technology and its limitations.",
    "Technological Uncertainties": "Describe key technological challenges faced.",
    "Technological Advancements": "Detail the innovations and new knowledge gained.",
    "Systematic Investigation": "Summarize experiments, tests, and development cycles.",
    "Supporting Evidence": "Provide documentation, code snippets, and test results.",
}

st.sidebar.subheader("Available SR&ED Sections")
section_name = st.sidebar.selectbox("Choose a section:", list(sred_sections.keys()))

# Query Input
query_text = st.text_area("Enter your query:")

# Run Query
if st.button("Run Query"):
    if not query_codebase:
        st.error("QueryCodebase is not initialized. Please check your configuration.")
        st.stop()

    if issue_key and query_text and section_name:
        st.write(f"*Selected Section:* {section_name}")
        st.write("*User Query:*", query_text)

        # Combine Jira context + user's query
        full_query = f"""
        Use the following Jira details as context to answer the query based on the vector dataset:

        Context from Issue [{issue_key}]:
        {parsed_context}

        Section: {section_name}

        User's Query:
        {query_text}
        """

        try:
            with st.spinner("Processing your query..."):
                # Perform the query and retrieve the answer
                answer, sources = query_codebase.perform_query(full_query, section_name)

            # Display Answer
            st.subheader("Generated Response:")
            st.write(answer)

            # Display Source Documents
            if sources:
                st.subheader("Source Documents:")
                for doc in sources:
                    st.write("- *File Path:*", doc.metadata.get("file_path", "Unknown"))
                    st.write("- *Commit Message:*", doc.metadata.get("commit_message", "No commit message provided"))
                    st.write("---")
            else:
                st.write("No relevant documents found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select an issue key, choose a section, and enter a query.")

