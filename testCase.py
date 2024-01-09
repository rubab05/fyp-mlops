import unittest
from unittest.mock import MagicMock
from CVShortlisting import load_embeddings

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
