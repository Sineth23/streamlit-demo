#os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
#dataset_path = st.sidebar.text_input("Dataset Path:", value="hub://autodoctest/smiddygroup-waio-portal-backend")
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InNpbmV0aDIzIiwiYXBpX2tleSI6Im50anUzeGhLN0xpTzFQWlpxWXZ6UG83Nm5hSVBJMVNDWTRQM3RJRjI0NDhpZiJ9."
# os.environ["OPENAI_KEY"] = "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"
import streamlit as st
from sred import QueryCodebase

# Streamlit Page Configuration
st.set_page_config(page_title="SR&ED Chatbot", page_icon="🤖", layout="wide")

# Sidebar Configuration
st.sidebar.title("SR&ED Chatbot Configuration")
dataset_path = "hub://your-org/your-dataset"  # Replace with actual ActiveLoop dataset path
model_name = "sentence-transformers/all-MiniLM-L6-v2"
openai_key = st.sidebar.text_input("OpenAI API Key:", type="password")

# Initialize QueryCodebase instance
if openai_key:
    query_codebase = QueryCodebase(dataset_path, model_name, openai_key)
    st.sidebar.success("QueryCodebase initialized successfully!")
else:
    st.sidebar.warning("Please enter your OpenAI API Key.")

# SR&ED Section Selection
sred_sections = ["Project Identification", "Technological Background", "Technological Uncertainties"]
section_name = st.sidebar.selectbox("Select SR&ED Section:", sred_sections)

# Query Input
query_text = st.text_area("Enter your query:")

if st.button("Run Query"):
    if not openai_key:
        st.error("Please enter your OpenAI API Key.")
    elif not query_text:
        st.error("Please enter a query.")
    else:
        try:
            with st.spinner("Processing query..."):
                answer = query_codebase.perform_query(query_text, section_name)
            st.subheader("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
