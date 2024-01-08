import os
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


path= os.getcwd()
pdf_folder_path= 'C:\\Users\\tahar\\Downloads\\CV Shortlisting\\CV Data'
print(os.listdir(pdf_folder_path))


loader = PyPDFDirectoryLoader(pdf_folder_path)
docs= loader.load()

embeddings = HuggingFaceEmbeddings()


Embedding_store_path = 'C:/Users/tahar/Downloads/CV Shortlisting'

def load_embeddings(sotre_name, path):
    with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore


# db_instructEmbedd = FAISS.from_documents(docs, instructor_embeddings)
db_instructEmbedd = load_embeddings(sotre_name='Embeddings', path=Embedding_store_path)

retrievers = db_instructEmbedd.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .25})


inputt = """Job Summary:As a Software Enginee, you will play a pivotal role in designing, developing, and implementing software solutions that drive our products and services. You'll collaborate closely with cross-functional teams, including product managers, designers, and other engineers, to deliver high-quality, scalable, and reliable software. Your expertise will contribute to the success of our projects, and you'll have the opportunity to grow professionally in a collaborative and supportive environment.

Responsibilities:

Collaborate with product managers and stakeholders to understand software requirements and translate them into technical specifications.
Design, develop, test, and deploy high-quality software solutions that meet customer needs and project requirements.
Write clean, efficient, and maintainable code using industry best practices and coding standards.
Participate in code reviews to provide and receive constructive feedback, ensuring code quality and consistency across the team.
Troubleshoot and resolve software defects and technical issues, identifying root causes and implementing appropriate solutions.
Stay up-to-date with emerging technologies, tools, and trends in software engineering and apply them to improve development processes.
Collaborate with cross-functional teams to define system architecture, integrate software components, and optimize applications for performance and scalability.
Contribute to the documentation of software designs, processes, and procedures to ensure knowledge sharing and maintainability.
Participate in Agile or other development methodologies to ensure timely and effective project delivery.
Qualifications:

Bachelors degree in Computer Science, Software Engineering, or a related field. Masters degree preferred.
Proven experience (5 years) as a Software Engineer, developing and delivering software applications.
Proficiency in one or more programming languages such as Java, Python, C++, or similar.
Strong understanding of software development principles, algorithms, data structures, and design patterns.
Experience with version control systems (e.g., Git) and collaborative development workflows.
Familiarity with Agile methodologies and tools for project management and issue tracking.
Excellent problem-solving skills and the ability to debug complex software systems.
Strong communication skills to effectively collaborate with cross-functional teams and articulate technical concepts to non-technical stakeholders.
Experience with cloud platforms (e.g., AWS, Azure, GCP) and containerization (e.g., Docker) is a plus.
Demonstrated ability to adapt to a fast-paced, dynamic work environment and learn new technologies quickly. """

docs = retrievers.get_relevant_documents(inputt)

for doc in docs:
    source = doc.metadata.get('source')  # Access the 'source' from the metadata
    if source:
        path.append(source)


