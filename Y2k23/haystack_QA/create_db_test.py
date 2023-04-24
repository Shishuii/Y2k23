#!/usr/bin/env python
# coding: utf-8

# In[1]:


from haystack.nodes import DensePassageRetriever
from haystack import Document
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
from haystack.nodes import DensePassageRetriever, FARMReader
import pandas as pd
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever
import os


# In[2]:


document_store_weaviate = WeaviateDocumentStore(index="events_cs_ms", 
					  recreate_index=True,
                                          embedding_dim=128, 
                                          return_embedding=True)


# In[3]:


retriever = DensePassageRetriever(
    document_store=document_store_weaviate,
    query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
)


# In[10]:


# Delete existing documents in documents store
document_store_weaviate.delete_documents()
all_docs = []

df = pd.read_csv("../data_extraction/test_combined.csv")[["combined_data", "event_id", "AlertID"]]
import pdb;pdb.set_trace()
#df = df.rename(columns={'AlertID' : 'event_id'})
def get_id(row):
    if pd.isna(row["event_id"]):
        row["event_id"] = row["AlertID"]
    return row
print(df.shape)
df = df.apply(lambda row: get_id(row), axis=1)
print(df.shape)
import pdb;pdb.set_trace()
df = df.fillna("")
document_store_weaviate.write_documents(
df[["combined_data", "event_id"]].rename(columns={ 
'combined_data':'content',
'event_id' : 'name'
}
).to_dict(orient='records'))

document_store_weaviate.update_embeddings(retriever)
print(f"Number of documents created:{document_store_weaviate.get_document_count()}") 


# In[18]:


#generator = Seq2SeqGenerator(model_name_or_path="dmis-lab/biobert-large-cased-v1.1-squad")
reader = FARMReader(model_name_or_path="dmis-lab/biobert-large-cased-v1.1-squad", use_gpu=True)


# In[19]:


pipe = ExtractiveQAPipeline(reader, retriever)


# In[23]:


filter_name = { 
        "name": "12345"
    }
    
res = pipe.run(
            query="What is the description of the event?",
            params={"Retriever": {"top_k": 3}, "filters": filter_name}
        )
print(res)
