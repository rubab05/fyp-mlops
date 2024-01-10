import os
import pickle
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

path = os.getcwd()
pdf_folder_path = 'cv-data/'
print(os.listdir(pdf_folder_path))

loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

embeddings = HuggingFaceEmbeddings()

#embedding_store_path = 'faiss_Embeddings.pkl'


# def load_embeddings(store_name, path):
#     with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
#         VectorStore = pickle.load(f)
#     return VectorStore


#db_instructEmbedd = FAISS.from_documents(docs, embeddings)
# db_i = load_embeddings(store_name='faiss_Embeddings',
# path=embedding_store_path)


# retrievers = db_i.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": .25}
# )

# input_text = """Software Engineer"""
# docs = retrievers.get_relevant_documents(input_text)

# sources = []  # Corrected variable name
# for doc in docs:
#     source = doc.metadata.get('source')
#     if source:
#         sources.append(source)  # Append source to the list

# Continue with the rest of your script as needed
