#SRED.py
# sred.py
import os
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

# Initialize environment variables
DATASET_PATH = "hub://your-org/your-dataset"  # Replace with your actual ActiveLoop dataset path
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

sred_prompts = {
    "Project Identification": """
Generate a detailed Project Identification section for an SR&ED report using the data from [Repository Name]. Include:
    - Project Title: [Project Name]
    - Overview: Provide a brief summary of the project's purpose and scope.
    - Project Objectives: Clearly state the technological objectives and intended advancements.

Ensure the information reflects the project's goals and aligns with SR&ED requirements. Include specific code references and Jira tickets where applicable.
""",
    "Technological Background": """
Draft the Technological Background section for the SR&ED report by:
    - Existing Technology State: Describe the state of technology before the project began, including existing solutions or approaches.
    - Limitations: Explain the limitations of these existing technologies and why they were insufficient for achieving [Project Name]'s objectives.

Use specific examples from the codebase in [Repository Name] and relevant Jira tickets to support this background.
""",
    "Technological Uncertainties": """
Compose the Technological Uncertainties section by identifying and elaborating on:
    - Challenges: Detail the key technological uncertainties or challenges the project aimed to overcome.
    - Nature of Uncertainties: Explain why these challenges were not readily solvable by professionals in the field and required experimental development.

Reference specific code commits, code snippets, and Jira tickets that illustrate these uncertainties.
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
        """Initialize embeddings, vectorstore, retriever."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = DeepLake(dataset_path=self.dataset_path, read_only=True, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.retriever.search_kwargs['k'] = 20
        self.llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=self.openai_key,
            temperature=0.3,
        )

    def perform_query(self, query_text: str, section_name: str) -> str:
        """Perform retrieval and generate a response."""
        sred_prompt = sred_prompts.get(section_name, "")
        full_prompt_template = f"""
You are a technical documentation assistant. Using the provided information, generate a structured response.

{sred_prompt}

Context:
{{context}}

Question:
{{question}}
"""

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=full_prompt_template,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

        result = qa_chain.invoke({"query": query_text})
        return result['result']

if __name__ == '__main__':
    query_codebase = QueryCodebase(DATASET_PATH, MODEL_NAME, OPENAI_API_KEY)
    print("Enter your queries below. Type 'exit' to quit.")
    while True:
        query_text = input("\nEnter your query: ")
        if query_text.lower() in ['exit', 'quit']:
            print("Exiting query session.")
            break
        print("\nSelect SR&ED Section:")
        for idx, section in enumerate(sred_prompts.keys(), start=1):
            print(f"{idx}. {section}")
        section_choice = input("Enter the number of the section: ")
        try:
            section_index = int(section_choice) - 1
            section_name = list(sred_prompts.keys())[section_index]
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
            continue
        try:
            answer = query_codebase.perform_query(query_text, section_name)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"An error occurred: {e}")
