#SRED.py
# sred.py
# SRED.py
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
   - *Eligibility Criteria:* Ensure that all described activities meet the CRA's SR&ED eligibility criteria:
     - *Technological Advancement*
     - *Technological Uncertainty*
     - *Systematic Investigation*
   - *Documentation Standards:* Provide detailed and accurate documentation as per CRA guidelines.
   - *Avoid Ineligible Activities:* Do not include routine development, aesthetic design changes, or business activities.

5. *Clarity and Precision:*
   - *Avoid Jargon:* Use clear language that can be understood by someone without specialized knowledge.
   - *Consistency:* Ensure consistency between all sections of the report.
   - *Technical Accuracy:* Verify that all technical details are correct and supported by the data.

6. *Addressing Technological Uncertainties:*
   - *Challenges Faced:* Detail the specific technological challenges encountered.
   - *Limitations of Existing Technology:* Explain why existing technologies or methods were insufficient.
   - *Necessity for Experimental Development:* Justify the need for a systematic investigation to overcome uncertainties.

7. *Documenting the Systematic Investigation:*
   - *Methodology:* Describe the approach taken to address the uncertainties.
   - *Experiments and Tests:* Summarize experiments, tests, and analyses conducted.
   - *Results and Learnings:* Present the outcomes, including both successes and failures, and what was learned.

8. *Avoiding Hallucinations:*
   - *Fact-Checking:* Cross-verify all statements with the retrieved data and Jira analysis.
   - *No Assumptions:* Do not make assumptions beyond the provided information.
   - *Transparency:* If certain details are unknown or not retrieved, mention that explicitly.

9. *Handling Jira Analysis with Multiple Repositories:*
   - *Selective Use:* Only use Jira tickets that are relevant to the repository being investigated.
   - *Avoid Cross-Referencing Unrelated Repositories:* Do not include information from Jira tickets that pertain to other repositories.
   - *Accuracy in Representation:* Ensure that any Jira ticket referenced is accurately represented and applicable.

*Jira Analysis*

The following Jira analysis contains detailed innovation highlights for multiple repositories. Use only the information relevant to the repository being investigated.

*Project Title: waio-backend Development*

*Project Timeframe:* September 16, 2024 to October 17, 2024

*Summary:* The waio-backend project focuses on developing and maintaining the backend API and related infrastructure. Key activities include optimizing database queries, implementing dynamic load balancing for APIs, and addressing synchronization issues.

*Detailed Innovation Highlights*

*waio-backend*

Specific Innovations:

- *Optimized Database Queries:* Reduced data retrieval time by 40% through indexing and optimized query structures.
- *Dynamic Load Balancing for APIs:* Developed a mechanism for adjusting API request distribution based on server load, reducing downtimes during peak usage.

Challenges Addressed:

- *Balancing High Data Throughput:* Needed to handle high volumes of data without compromising API response times.
- *Synchronization Issues:* Addressed discrepancies in data synchronization between services.

Contributions to the Industry:

- *API Infrastructure Optimization:* Provides a model for optimizing API infrastructures, valuable for platforms handling high transaction volumes.

Proof Required:

- *Database Query Logs:* Documentation of query performance before and after optimization.
- *API Load Testing Results:* Demonstrating the impact of load balancing on request handling.
- *Code Snippets:* Methods implementing indexing and query improvements.

*Conclusion:*

The waio-backend project demonstrates significant technological advancements and systematic investigation to overcome technological uncertainties, aligning with the SR&ED eligibility criteria.

---

*Understanding SR&ED*

The Scientific Research and Experimental Development (SR&ED) program is a Canadian federal tax incentive initiative designed to encourage businesses to conduct R&D in Canada. Administered by the Canada Revenue Agency (CRA), SR&ED offers income tax credits and refunds for qualifying R&D expenditures.

*Key Objectives of SR&ED:*

- *Encourage Innovation:* Stimulate technological advancement by reducing financial risks associated with R&D.
- *Economic Growth:* Promote job creation and economic development within Canada.
- *Global Competitiveness:* Help Canadian companies remain competitive internationally.

*Types of SR&ED Work:*

- *Basic Research:* Advancement of scientific knowledge without a specific application.
- *Applied Research:* Research aimed at solving a specific problem or achieving a particular objective.
- *Experimental Development:* Systematic work to achieve technological advancement, creating new or improving existing materials, devices, products, or processes.

*Eligibility Criteria*

To qualify for SR&ED incentives, R&D activities must meet three core criteria:

1. *Technological Advancement:*
   - *Goal:* Generate information that advances understanding of scientific relations or technologies.
   - *Relevance:* Advancement should be in a field relevant to your business.

2. *Technological Uncertainty:*
   - *Challenge:* Presence of technological uncertainties that cannot be resolved using standard practices.
   - *Documentation:* Clearly document the uncertainties and why they were not solvable through existing knowledge.

3. *Systematic Investigation:*
   - *Methodology:* Use a systematic approach, including hypothesis formulation, testing, and analysis.
   - *Process:* Document each step of the investigation, experiments, and conclusions.

*Common Eligible Activities for SaaS Companies:*

- *Developing New Algorithms:* Creating innovative algorithms to solve complex problems.
- *Optimizing Performance:* Enhancing software performance beyond existing standards.
- *Integration Challenges:* Overcoming difficulties in integrating disparate systems or technologies.
- *Security Enhancements:* Developing advanced security protocols or encryption methods.

*Ineligible Activities:*

- *Routine Testing:* Standard debugging or testing without technological uncertainties.
- *Aesthetic Design:* Cosmetic changes that do not affect underlying technology.
- *Business or Market Research:* Activities focused on market analysis or business strategy.

*Requirements to Pass an Audit and Approve an Application*

1. *Detailed Documentation:*
   - *Project Records:* Maintain comprehensive records, including plans, designs, and test results.
   - *Technical Reports:* Prepare narratives explaining technological uncertainties and advancements.
   - *Time Tracking:* Accurately track time spent on SR&ED activities.

2. *Financial Documentation:*
   - *Expense Tracking:* Keep detailed records of all SR&ED-related expenditures.
   - *Allocation Methods:* Clearly explain expense allocations to SR&ED activities.

3. *Compliance with CRA Guidelines:*
   - *Policy Adherence:* Align claims with CRA SR&ED policies, especially for software development.
   - *Legislative Requirements:* Familiarize with relevant sections of the Income Tax Act.

4. *Effective Communication:*
   - *Clarity:* Use clear, concise descriptions focusing on technological challenges and advancements.
   - *Consistency:* Ensure consistency between technical and financial information.

5. *Audit Preparation:*
   - *Internal Review:* Conduct internal audits to identify and rectify issues.
   - *Staff Training:* Educate the team on SR&ED requirements and documentation importance.
   - *Professional Assistance:* Consider consulting SR&ED experts for preparation and representation.

6. *Responding to CRA Queries:*
   - *Timeliness:* Respond promptly to requests for additional information.
   - *Transparency:* Be open and honest in communications with auditors.

*Common Pitfalls to Avoid*

- *Insufficient Evidence:* Failing to provide adequate documentation.
- *Overstating Activities:* Including routine work or ineligible activities.
- *Inconsistent Information:* Discrepancies between different sections of the claim.
- *Lack of Technological Uncertainty:* Not clearly demonstrating the technological challenges faced.

*Best Practices for a Successful Claim*

1. *Early Planning:*
   - *Identify Projects Early:* Recognize potential SR&ED projects from the outset.
   - *Set Up Systems:* Implement tracking systems for time, expenses, and progress.

2. *Regular Documentation:*
   - *Frequent Updates:* Encourage regular documentation of work and challenges.
   - *Version Control:* Use tools to track code changes and document iterations.

3. *Collaboration:*
   - *Cross-Departmental Communication:* Ensure alignment between technical and financial teams.
   - *Stakeholder Involvement:* Include key personnel in claim preparation.

4. *Continuous Learning:*
   - *Stay Informed:* Keep up-to-date with SR&ED policy changes.
   - *Training Sessions:* Attend workshops on SR&ED compliance and best practices.

*Final Reminders*

- *Accuracy is Crucial:* Ensure all information is accurate and verifiable.
- *Demonstrate Value:* Clearly show how the project's innovations contribute to technological advancement.
- *Align with CRA Expectations:* Structure the report to meet CRA's requirements and expectations.
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

Ensure the information reflects the project's goals and aligns with SR&ED requirements. Include specific code references and Jira tickets where applicable.
""",
    "Technological Background": """
Draft the Technological Background section for the SR&ED report by:
    - *Existing Technology State:* Describe the state of technology before the project began, including existing solutions or approaches.
    - *Limitations:* Explain the limitations of these existing technologies and why they were insufficient for achieving [Project Name]'s objectives.

Use specific examples from the codebase in [Repository Name] and relevant Jira tickets to support this background.
""",
    "Technological Uncertainties": """
Compose the Technological Uncertainties section by identifying and elaborating on:
    - *Challenges:* Detail the key technological uncertainties or challenges the project aimed to overcome.
    - *Nature of Uncertainties:* Explain why these challenges were not readily solvable by professionals in the field and required experimental development.

Reference specific code commits, code snippets, and Jira tickets that illustrate these uncertainties.
""",
    "Technological Advancements": """
Create the Technological Advancements section by detailing:
    - *Innovations Achieved:* Describe the new knowledge gained or significant improvements made to existing technology through the project.
    - *Impact:* Explain how these advancements contribute to the field or improve upon existing solutions.

Include references to specific code changes, features developed, and solutions implemented, as documented in [Repository Name] and relevant Jira tickets.
""",
    "Systematic Investigation": """
Outline the Systematic Investigation process undertaken during the project by:
    - *Methodology:* Describe the systematic approach used to address the technological uncertainties.
    - *Experiments and Analysis:* Summarize the experiments, tests, and analyses conducted.
    - *Iterations and Findings:* Document the iterative process, including both successful outcomes and failures.

Extract details from development logs, test results, code version histories, and Jira tickets available in the vector database.
""",
    "Supporting Evidence": """
Compile the Supporting Evidence section by:
    - *Code Snippets:* Include relevant code examples that demonstrate the work performed.
    - *Architectural Diagrams:* Provide diagrams illustrating system architecture or components.
    - *Test Results:* Include summaries of test results and performance metrics.
    - *Meeting Notes:* Reference notes that document decisions and progress.

Ensure that all evidence is clearly linked to the technological uncertainties and advancements previously described.
"""
}

class QueryCodebase:
    def __init__(self, dataset_path: str, model_name: str, openai_key: str):
        """Initialize embeddings, vectorstore, and retriever."""
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.openai_key = openai_key
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = DeepLake(
            dataset_path=self.dataset_path,
            read_only=True,
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()
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

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=full_prompt_template,
        )

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

        result = qa_chain.invoke({"query": query_text})
        answer = result['result']
        source_documents = result['source_documents']

        # Ensure default metadata if missing
        for doc in source_documents:
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata.setdefault('author_name', 'Unknown')
            doc.metadata.setdefault('author_email', 'Unknown')
            doc.metadata.setdefault('commit_date', 'Unknown')
            doc.metadata.setdefault('commit_message', 'No commit message provided')
            doc.metadata.setdefault('file_path', 'Unknown')
            doc.metadata.setdefault('start_line', 1)
            doc.metadata.setdefault('end_line', 1)
            doc.metadata.setdefault('chunk_type', 'Unknown')
            doc.metadata.setdefault('language', 'Unknown')

        print("\nAnswer:")
        print(answer)

        return answer, source_documents
