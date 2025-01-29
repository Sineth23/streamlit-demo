#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
#dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
import streamlit as st
import pandas as pd
import os
from SRED import QueryCodebase  # Ensure the import matches your local module structure
from langchain_community.vectorstores import DeepLake
#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."

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

# SR&ED Prompt Selection
sred_prompts = {
    "Project Identification": """
Generate a detailed Project Identification section for an SR&ED report using the data from [Repository Name]. Include:
    - Project Title: [Project Name]
    - Overview: Provide a brief summary of the project's purpose and scope.
    - Project Objectives: Clearly state the technological objectives and intended advancements.
""",
    "Technological Background": """
Draft the Technological Background section for the SR&ED report by:
    - Existing Technology State: Describe the state of technology before the project began, including existing solutions or approaches.
    - Limitations: Explain the limitations of these existing technologies and why they were insufficient for achieving [Project Name]'s objectives.
""",
    "Technological Uncertainties": """
Compose the Technological Uncertainties section by identifying and elaborating on:
    - Challenges: Detail the key technological uncertainties or challenges the project aimed to overcome.
    - Nature of Uncertainties: Explain why these challenges were not readily solvable by professionals in the field and required experimental development.
""",
    "Technological Advancements": """
Create the Technological Advancements section by detailing:
    - Innovations Achieved: Describe the new knowledge gained or significant improvements made to existing technology through the project.
    - Impact: Explain how these advancements contribute to the field or improve upon existing solutions.
""",
    "Systematic Investigation": """
Outline the Systematic Investigation process undertaken during the project by:
    - Methodology: Describe the systematic approach used to address the technological uncertainties.
    - Experiments and Analysis: Summarize the experiments, tests, and analyses conducted.
    - Iterations and Findings: Document the iterative process, including both successful outcomes and failures.
""",
    "Supporting Evidence": """
Compile the Supporting Evidence section by:
    - Code Snippets: Include relevant code examples that demonstrate the work performed.
    - Architectural Diagrams: Provide diagrams illustrating system architecture or components.
    - Test Results: Include summaries of test results and performance metrics.
"""
}

st.sidebar.subheader("Available SR&ED Sections")
section_name = st.sidebar.selectbox("Choose a section:", list(sred_prompts.keys()))

# Query Input
query_text = st.text_input("Enter your query:", "")

# Run Query
if st.button("Run Query"):
    if not query_codebase:
        st.error("QueryCodebase is not initialized. Please check your configuration.")
        st.stop()

    if issue_key and query_text and section_name:
        st.write(f"*Selected Section:* {section_name}")
        st.write("*User Query:*", query_text)

        # Combine JIRA context + user's query into a single string
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
                # Perform the query with the combined text and selected section
                answer = query_codebase.perform_query(full_query, section_name)
            
            # Display the Answer
            st.subheader("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select an issue key, choose a section, and enter a query.")

