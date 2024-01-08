import unittest
from unittest.mock import MagicMock
from CVShortlisting import load_embeddings

class TestLoadEmbeddings(unittest.TestCase):
    def test_load_embeddings(self):
        # Mocking the pickle.load function
        with unittest.mock.patch("builtins.open"), unittest.mock.patch("pickle.load") as mock_load:
            # Setting up the mock return value
            mock_load.return_value = "MockVectorStore"

            # Calling the function
            result = load_embeddings(store_name='Embeddings', path='mock_path')

            # Assertions
            self.assertEqual(result, "MockVectorStore")
            mock_load.assert_called_once_with('mock_path/faiss_Embeddings.pkl', 'rb')

# Assuming your_script_filename.py contains the code you provided
class TestRetrievalProcess(unittest.TestCase):
    def test_retrieval_process(self):
        # Mocking the necessary objects and methods
        db_i = MagicMock()
        retrievers = MagicMock()
        docs = MagicMock()
        docs.get_relevant_documents.return_value = ["doc1", "doc2"]

        # Setting up the mock return values
        db_i.as_retriever.return_value = retrievers
        retrievers.get_relevant_documents.return_value = docs

        # Calling the function
        result = retrieve_documents()

        # Assertions
        self.assertEqual(result, ["doc1", "doc2"])
        db_i.as_retriever.assert_called_once_with(search_type="similarity_score_threshold",
                                                 search_kwargs={"score_threshold": .25})

if __name__ == '__main__':
    unittest.main()
