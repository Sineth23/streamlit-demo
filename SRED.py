#SRED.py
import os
from dotenv import load_dotenv
from typing import Dict, Tuple, List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

sred_prompts = {
    "Project Identification": """
Generate a detailed Project Identification section for an SR&ED report.
Include project title, objectives, and a brief overview based on the provided repository data.
""",
    "Technological Background": """
Draft the Technological Background section detailing the existing technology state,
limitations, and why improvements were needed based on repository insights.
""",
    "Technological Uncertainties": """
Identify key technological uncertainties, why they posed challenges,
and how they were addressed in development.
""",
    "Technological Advancements": """
Describe the innovations achieved, their impact, and improvements over existing solutions.
""",
    "Systematic Investigation": """
Outline the systematic approach used, including methodology, experiments, and iterations.
""",
    "Supporting Evidence": """
Compile supporting evidence such as test results, architectural diagrams, and code snippets.
"""
}

class QueryCodebase:
    def __init__(self, dataset_path: str, model_name: str, openai_key: str):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.openai_key = openai_key
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embeddings, vectorstore, and retriever."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = DeepLake(dataset_path=self.dataset_path, read_only=True, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.retriever.search_kwargs['k'] = 20
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=self.openai_key, temperature=0.3)

    def perform_query(self, query_text: str, section_name: str) -> str:
        """Perform retrieval and generate an SR&ED response."""
        sred_prompt = sred_prompts.get(section_name, "")
        if not sred_prompt:
            return "Invalid section name."

        full_prompt_template = f"""
{sred_prompt}

Based on the following retrieved data, provide a comprehensive response.

Code Context:
{{context}}

Query:
{{question}}
"""
        
        custom_prompt = PromptTemplate(input_variables=["context", "question"], template=full_prompt_template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": custom_prompt},
        )

        result = qa_chain.invoke({"query": query_text})
        return result['result']

if __name__ == '__main__':
    dataset_path = 'hub://autodoctest/smiddygroup-waio-portal-backend'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    openai_key = os.getenv('OPENAI_API_KEY')
    
    query_codebase = QueryCodebase(dataset_path, model_name, openai_key)
    
    while True:
        query_text = input("Enter your query: ")
        if query_text.lower() in ['exit', 'quit']:
            break
        
        print("Select SR&ED Section:")
        for idx, section in enumerate(sred_prompts.keys(), start=1):
            print(f"{idx}. {section}")
        
        section_choice = input("Enter the section number: ")
        try:
            section_index = int(section_choice) - 1
            section_name = list(sred_prompts.keys())[section_index]
        except (ValueError, IndexError):
            print("Invalid selection.")
            continue
        
        try:
            response = query_codebase.perform_query(query_text, section_name)
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")
