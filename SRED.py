#SRED.py
# sred.py
import os
from dotenv import load_dotenv
from typing import Dict, Tuple, List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

# =====================================
# SR&ED Context and Prompts Definition
# =====================================

sred_context = """
*Important Instructions for SR&ED Report Generation*

When generating the SR&ED report sections, please adhere to the following guidelines to ensure accuracy, compliance, and effectiveness:

1. *Use Only Provided Information:*
   - *Data Sources:* Utilize only the information from the retrieved code repository data, the Jira analysis provided, and this context.
   - *Relevant Repository Only:* Focus exclusively on the repository being investigated.
   - *No Fabrication:* Do not invent or assume information not explicitly provided.
   - *Acknowledgment of Gaps:* If certain information is unavailable, clearly state that and proceed with the available data.

2. *Focus on Demonstrating Innovation:*
   - *Technological Advancements:* Emphasize any new technologies, algorithms, or methods developed within the repository.
   - *Problem-Solving:* Highlight how the project addressed specific technological challenges or uncertainties.
   - *Impact:* Explain the significance of the innovations in advancing the field or improving existing solutions.

3. *Reference Specific Code Examples and Jira Tickets:*
   - *Code Commits:* Cite specific commits or code snippets that illustrate the work performed.
   - *Jira Tickets:* Reference relevant Jira tickets from the analysis that apply to the repository.
   - *Features and Modules:* Reference particular features, modules, or components developed.

4. *Compliance with CRA Requirements:*
   ...
[Truncated here for brevity; keep the same context as your original SR&ED instructions]
"""

system_prompt = """You are a technical documentation assistant. Your task is to:
1. Analyze the provided content and extract meaningful information.
2. Organize the information in a clear, structured way.
3. Focus on explaining what the code or project does.
4. Provide specific details about functionality, features, and components.
5. Include code examples and references where applicable.
6. If you can't find meaningful information about the project's purpose, honestly state that.

Always maintain technical accuracy while making the explanation accessible."""

# ===============================
# SR&ED Prompts Definition
# ===============================

sred_prompts = {
    "Project Identification": """
Generate a detailed Project Identification section for an SR&ED report using the data from [Repository Name]. Include:
    - *Project Title:* [Project Name]
    - *Overview:* Provide a brief summary of the project's purpose and scope.
    - *Project Objectives:* Clearly state the technological objectives and intended advancements.
""",
    "Technological Background": """
Draft the Technological Background section for the SR&ED report by:
    - *Existing Technology State:* Describe the state of technology before the project began
    - *Limitations:* Explain the limitations
""",
    "Technological Uncertainties": """
Compose the Technological Uncertainties section by identifying:
    - *Challenges*
    - *Nature of Uncertainties*
""",
    "Technological Advancements": """
Create the Technological Advancements section by detailing:
    - *Innovations Achieved*
    - *Impact*
""",
    "Systematic Investigation": """
Outline the Systematic Investigation process by:
    - *Methodology*
    - *Experiments and Analysis*
    - *Iterations and Findings*
""",
    "Supporting Evidence": """
Compile the Supporting Evidence section by:
    - *Code Snippets*
    - *Architectural Diagrams*
    - *Test Results*
    - *Meeting Notes*
"""
}

class QueryCodebase:
    def __init__(self, dataset_path: str, model_name: str, openai_key: str, activeloop_token: str):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.openai_key = openai_key
        self.activeloop_token = activeloop_token
        self.embeddings = None
        self.vectorstore = DeepLake(
            dataset_path=self.dataset_path,
            token=self.activeloop_token,
            embedding=self.embeddings,
            read_only=True
        )
        self.retriever = None
        self.llm = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embeddings, vectorstore, and retriever."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = DeepLake(
            dataset_path=self.dataset_path,
            read_only=True,
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()
        # Adjust number of retrieved documents as needed
        self.retriever.search_kwargs['k'] = 20

        self.llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=self.openai_key,
            temperature=0.3,
        )

    def perform_query(self, query_text: str, section_name: str) -> Tuple[str, List[Dict]]:
        """Perform retrieval and generate an SR&ED-focused response."""
        # Retrieve the SR&ED prompt
        sred_prompt = sred_prompts.get(section_name, "")
        if not sred_prompt:
            print(f"Invalid section name: {section_name}")
            return "", []

        # Combine the prompts into a single template
        full_prompt_template = f"""
{sred_context}

{system_prompt}

{sred_prompt}

Using the following code snippets and their metadata, answer the question in detail.

Code Snippets and Metadata:
{{context}}

Question:
{{question}}

Your answer should reference the authors and any relevant metadata, providing a comprehensive and informative response.
"""

        # Create a custom prompt for the chain
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=full_prompt_template,
        )

        # Each retrieved document is formatted with this prompt
        document_prompt = PromptTemplate(
            input_variables=[
                "page_content", "author_name", "author_email", "commit_date",
                "commit_message", "file_path", "start_line", "end_line",
                "chunk_type", "language"
            ],
            template="""
Document:

Metadata:
Author: {author_name}
Author Email: {author_email}
Commit Date: {commit_date}
Commit Message: {commit_message}
File Path: {file_path}
Start Line: {start_line}
End Line: {end_line}
Chunk Type: {chunk_type}
Language: {language}

Content:
{page_content}
""",
        )

        # Build the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": custom_prompt,
                "document_prompt": document_prompt,
            },
        )

        # Invoke the chain with the user's query
        result = qa_chain.invoke({"query": query_text})
        answer = result["result"]
        source_documents = result["source_documents"]

        # Ensure default metadata if missing
        for doc in source_documents:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata.setdefault("author_name", "Unknown")
            doc.metadata.setdefault("author_email", "Unknown")
            doc.metadata.setdefault("commit_date", "Unknown")
            doc.metadata.setdefault("commit_message", "No commit message provided")
            doc.metadata.setdefault("file_path", "Unknown")
            doc.metadata.setdefault("start_line", 1)
            doc.metadata.setdefault("end_line", 1)
            doc.metadata.setdefault("chunk_type", "Unknown")
            doc.metadata.setdefault("language", "Unknown")

        print("\nAnswer:\n", answer)
        return answer, source_documents


if __name__ == '__main__':
    # Example usage if you run this script directly (not usually needed in Streamlit)
    dataset_path = 'hub://username/dataset_name'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    openai_key = os.getenv('OPENAI_API_KEY') or "your_openai_key_here"

    query_codebase = QueryCodebase(dataset_path, model_name, openai_key)

    print("Enter your queries below. Type 'exit' or 'quit' to end the session.")
    print("Available SR&ED Sections:")
    for section in sred_prompts.keys():
        print(f"- {section}")

    while True:
        query_text = input("\nEnter your query: ")
        if query_text.lower() in ['exit', 'quit']:
            print("Exiting the query session.")
            break

        print("\nSelect SR&ED Section:")
        for idx, section in enumerate(sred_prompts.keys(), start=1):
            print(f"{idx}. {section}")

        section_choice = input("Enter the number of the section you want to generate: ")
        try:
            section_index = int(section_choice) - 1
            if 0 <= section_index < len(sred_prompts):
                section_name = list(sred_prompts.keys())[section_index]
            else:
                print("Invalid selection. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number corresponding to the section.")
            continue

        try:
            answer, docs = query_codebase.perform_query(query_text, section_name)
        except Exception as e:
            print(f"An error occurred: {e}")
