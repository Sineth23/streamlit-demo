#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
#dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."
# os.environ["OPENAI_KEY"] = "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"
import streamlit as st
import pandas as pd
from SRED import QueryCodebase
import os
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."
# Streamlit Page Configuration
st.set_page_config(
    page_title="SR&ED Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---- SIDEBAR CONFIG ----
st.sidebar.title("SR&ED Chatbot Configuration")

# Collect Activeloop / DeepLake dataset path
dataset_path = st.sidebar.text_input("DeepLake Dataset Path", value="hub://username/dataset_name")

# Model Name and OpenAI Key
model_name = st.sidebar.text_input("Model Name", value="sentence-transformers/all-MiniLM-L6-v2")
openai_key = st.sidebar.text_input("OpenAI API Key", value="", type="password")

# Upload Jira CSV
uploaded_jira = st.sidebar.file_uploader("Upload Jira CSV", type=["csv"])

# Initialize QueryCodebase
query_codebase = None
if openai_key and dataset_path and model_name:
    try:
        query_codebase = QueryCodebase(dataset_path, model_name, openai_key)
        st.sidebar.success("QueryCodebase initialized successfully!")
    except Exception as e:
        st.sidebar.error(f"Error initializing QueryCodebase: {e}")
else:
    st.sidebar.info("Please provide dataset path, model name, and OpenAI key.")

# If no CSV uploaded, prompt user to do so
if not uploaded_jira:
    st.warning("Please upload a JIRA CSV file to proceed.")
    st.stop()

# Once CSV is uploaded, read into dataframe
jira_df = pd.read_csv(uploaded_jira)

# ---- MAIN PAGE ----
st.title("SR&ED Chatbot")

# Issue Selection
st.sidebar.subheader("Jira Issue Selection")
issue_keys = jira_df["Issue key"].unique()
if len(issue_keys) == 0:
    st.error("No issue keys found in CSV.")
    st.stop()

issue_key = st.sidebar.selectbox("Select an Issue Key:", issue_keys)

parsed_context = ""
if issue_key:
    selected_issue = jira_df[jira_df["Issue key"] == issue_key].iloc[0]
    st.subheader(f"Details for Issue: {issue_key}")

    # Relevant fields that will serve as context
    relevant_features = [
        "Summary", "Description", "Comments", "Labels", "Priority"
    ]

    context_lines = []
    for feature in relevant_features:
        # safer to use .get() in case column doesn't exist
        value = selected_issue.get(feature, "N/A")
        context_lines.append(f"{feature}: {value}")
        st.markdown(f"**{feature}:** {value}")

    # Combine the features into one string to use as context
    parsed_context = "\n".join(context_lines)

# SR&ED Prompt Selection
sred_prompts = {
    "Project Identification": """
Generate a detailed Project Identification section for an SR&ED report using the data from [Repository Name]. 
Include:
    - Project Title: [Project Name]
    - Overview: Provide a brief summary of the project's purpose and scope.
    - Project Objectives: Clearly state the technological objectives and intended advancements.
Ensure the information reflects the project's goals and aligns with SR&ED requirements. 
Include specific code references and Jira tickets where applicable.
""",
    "Technological Background": """
Draft the Technological Background section for the SR&ED report by:
    - Existing Technology State: Describe the state of technology before the project began
    - Limitations: Explain the limitations of these existing technologies
Use specific examples from the codebase in [Repository Name] and relevant Jira tickets.
""",
    "Technological Uncertainties": """
Compose the Technological Uncertainties section by identifying and elaborating on:
    - Challenges
    - Nature of Uncertainties
Reference specific code commits, code snippets, and Jira tickets.
""",
    "Technological Advancements": """
Create the Technological Advancements section by detailing:
    - Innovations Achieved
    - Impact
""",
    "Systematic Investigation": """
Outline the Systematic Investigation process by:
    - Methodology
    - Experiments and Analysis
    - Iterations and Findings
""",
    "Supporting Evidence": """
Compile the Supporting Evidence section by:
    - Code Snippets
    - Architectural Diagrams
    - Test Results
    - Meeting Notes
"""
}

st.sidebar.subheader("Available SR&ED Prompts")
selected_prompt_name = st.sidebar.selectbox("Choose a prompt:", list(sred_prompts.keys()))

# User Query input
query_text = st.text_input("Enter your query:", "")

# Run Query
if st.button("Run Query"):
    if not query_codebase:
        st.error("QueryCodebase is not initialized. Please check your configuration.")
        st.stop()

    if issue_key and selected_prompt_name and query_text.strip():
        st.write(f"**Selected Prompt:** {selected_prompt_name}")
        st.write("**User Query:**", query_text)

        # Combine JIRA context + user's query into a single string
        full_query = f"""
        Use the following Jira details as context to answer the query and fetch relevant code snippets and source documents:

        Context from Issue [{issue_key}]:
        {parsed_context}

        User's Query:
        {query_text}
        """

        st.write("Debug - Combined Query Text:", full_query)

        try:
            with st.spinner("Processing your query..."):
                # Perform the query by passing:
                #   the combined text
                #   the user-selected SR&ED prompt name
                answer, source_documents = query_codebase.perform_query(full_query, selected_prompt_name)

            # Display the Answer
            st.subheader("Answer:")
            st.write(answer)

            # Display Source Documents
            if source_documents:
                st.subheader("Source Documents:")
                for doc in source_documents:
                    st.markdown("- **File Path:** " + str(doc.metadata.get("file_path", "Unknown")))
                    st.markdown("- **Author:** " + str(doc.metadata.get("author_name", "Unknown")))
                    st.markdown("- **Commit Message:** " + str(doc.metadata.get("commit_message", "No commit message provided")))
                    st.markdown("---")
            else:
                st.write("No source documents were returned or matched.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select an issue key, choose a prompt, and enter a query.")
