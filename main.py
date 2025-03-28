# Querying script for ticketlabs project with S3 dataset support

import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import os

# Streamlit Page Configuration
st.set_page_config(
    page_title="Code Query Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Streamlit Sidebar Configuration
st.sidebar.title("Configuration")
s3_dataset_path = st.sidebar.text_input("S3 Dataset Path:", value="s3://autodocsolutions/waio-2025/")
aws_access_key = st.sidebar.text_input("AWS Access Key:", value="AKIASFUIRTPQ5MZ4P4D3")
aws_secret_key = st.sidebar.text_input("AWS Secret Key:", value="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")
model_name = st.sidebar.text_input("Model Name:", value="sentence-transformers/all-MiniLM-L6-v2")
openai_key = st.sidebar.text_input("OpenAI API Key:", value="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")

# Initialize components
def initialize_components(s3_dataset_path, aws_access_key, aws_secret_key, model_name, openai_key):
    """Initialize embeddings, vectorstore, and retriever."""
    # Configure AWS credentials
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = DeepLake(
        dataset_path=s3_dataset_path,
        read_only=True,
        embedding=embeddings,
        creds={
            'aws_access_key_id': aws_access_key,
            'aws_secret_access_key': aws_secret_key,
        }
    )
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs['k'] = 20
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        openai_api_key=openai_key,
        temperature=0.3,            
    )
    return retriever, llm

# Chatbot Interface
st.title("Code Query Chatbot")
st.write("""
Welcome to the Code Query Chatbot! This tool assists with querying code files and scripts from your codebase stored in S3.
""")

# Input for Query
query_text = st.text_input("Enter your query (e.g., 'Show me any code that handles partial or fractional token transactions'):")

# Streamlit button for running query
if st.button("Run Query"):
    if query_text and s3_dataset_path and aws_access_key and aws_secret_key and model_name and openai_key:
        st.write("**Query:**", query_text)

        try:
            with st.spinner("Processing your query..."):
                # Initialize components
                retriever, llm = initialize_components(s3_dataset_path, aws_access_key, aws_secret_key, model_name, openai_key)

                # Define the prompt template
                prompt_template = """
You are a technical documentation assistant. Your task is to:
1. Analyze the provided code snippets and extract meaningful information.
2. Focus on explaining what the code does and how it works.
3. Provide specific details about functionality, features, and components.
4. Include code examples and references where applicable.
5. If you can't find meaningful information about the code's purpose, honestly state that.

Using the following code snippets and their metadata, answer the question in detail.

Code Snippets and Metadata:
{context}

Question:
{question}

Your answer should reference the file paths and provide a comprehensive and informative response.
"""

                custom_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=prompt_template,
                )

                # Create the QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type='stuff',
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": custom_prompt,
                    },
                )

                # Perform the query
                result = qa_chain.invoke({"query": query_text})
                answer = result['result']
                source_documents = result['source_documents']

                # Filter out commit-related metadata
                filtered_documents = []
                for doc in source_documents:
                    # Only include file path and code content
                    filtered_doc = {
                        "file_path": doc.metadata.get("file_path", "Unknown"),
                        "code_content": doc.page_content,
                    }
                    filtered_documents.append(filtered_doc)

            # Display Answer
            st.subheader("Answer:")
            st.write(answer)

            # Display Filtered Source Documents
            if filtered_documents:
                st.subheader("Relevant Code Snippets:")
                for doc in filtered_documents:
                    st.write("- **File Path:**", doc["file_path"])
                    st.code(doc["code_content"], language="python")
                    st.write("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query and provide all required inputs.")

# Footer
st.sidebar.info("Powered by AutoDoc AI")
