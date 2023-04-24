from haystack.nodes import DensePassageRetriever
from haystack import Document
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever
import os

class Event_qa():
    def __init__(self):
        self.document_store = WeaviateDocumentStore(index="cs",
                                          embedding_dim=128,
                                          return_embedding=True)
        self.retriever = DensePassageRetriever(
                                        document_store=self.document_store,
                                        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
                                        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
                                    )
        print("loading farmreader")
        self.reader = FARMReader(model_name_or_path="dmis-lab/biobert-large-cased-v1.1-squad", use_gpu=True)
        print("loaded farmreader")
        self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)
        
    def query(self, e_id, query):
        filter_name = {"name": e_id}
        result = self.pipe.run(query=query, params={"Retriever": {"filters": filter_name, "top_k": 3}})
        answer = None
        if result and result.get("answers"):
            answer = result["answers"][0].answer
            print(f"Answer: {result['answers'][0]}")
        return answer
if __name__ == "__main__":
    context_qa = Context_qa()
    query = 'What is the name of the alert?'
    res = context_qa.query(s_id = "12", query=query)
    print('Results: ')
    print(res)
