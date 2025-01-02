import os
import sys
import logging

import pandas as pd
import numpy  as np
from datasets import load_dataset

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext
from llama_index.core import Settings

import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaDBSettings
from llama_index.vector_stores.chroma import ChromaVectorStore

# logging settings
logger = logging.getLogger("fact-checker.data-loader")

DOWNLOAD_LOCATION: str = "data"
if not os.path.exists(DOWNLOAD_LOCATION): os.mkdir(DOWNLOAD_LOCATION)
def download_climate_fever(streaming: bool = False): 
    """Load HuggingFace dataset `climate-fever` with a predefined cache directory.
    Args:
    * streaming (bool, optional). Load data in streaming mode. Defaults False
    
    """
    try:
        logger.info("climate fever loading")
        return load_dataset("tdiggelm/climate_fever", cache_dir = DOWNLOAD_LOCATION, streaming = streaming)
    except:
        logger.critical("failed to load climate-fever dataset")
        sys.exit(1)

def process_climate_fever(save=True):
    ds = download_climate_fever() 
    df = ds["test"].to_pandas()
    
    def accumulate_evidence(evidence_list):
        sep = "\n*"
        complete_evidence = sep
        for evidence in evidence_list:
            complete_evidence = complete_evidence + sep + evidence["evidence"]
        return complete_evidence
        
    df["evidence"] = df["evidences"].apply(accumulate_evidence)
    df["text"]     = "Evidence: " + df["evidence"] + "\nClaim: " + df["claim"]
    df["label"]    = df["claim_label"]
    df["unanswerable"] = df["label"] == 2 # wheter it is not supported 

    logger.info("parsed climate fever")

    if save:
        logger.info("parsed climate fever was saved in 'data/climate_fever.csv'")
        df.to_csv("data/climate_fever.csv", index = False)
    return df

#----------------------------------------------------------------------------------------
# Vector Database
#----------------------------------------------------------------------------------------

# Load the embedding model
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
logger.debug(f"loaded embedding model: {embed_model_id}")

# Configuración de ChromaDB
if not os.path.exists(DOWNLOAD_LOCATION + "/chromadb"): 
    os.mkdir(DOWNLOAD_LOCATION + "/chromadb")

chroma_client = chromadb.PersistentClient(path=DOWNLOAD_LOCATION+"/chromadb")
collection = chroma_client.get_or_create_collection("index_climate_fever")

logger.debug("sucessfully created chromaDB instance")

# settings = Settings(embed_model=embed_model)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def load_dataset_vector(preloaded: bool = True):
    if preloaded and os.path.exists("data/climate_fever.csv"):
        df = pd.read_csv("data/climate_fever.csv")
    else:
        df = process_climate_fever()

    df['claim_id'] = df['claim_id'].astype(str) # id must be a string

    # create data iterator
    nodes = []
    for row in df.to_dict("records"):
        new_metadata = {
            "id": row['claim_id'],
            "support": row['unanswerable'],
            "label": row['label'],  
            "evidence_length": len(row['evidence'].split("\n")) 
        }

        node = TextNode(text=row['text'], metadata=new_metadata)
        nodes.append(node)
    
    return nodes

index = None
try:
    if collection.count() > 0:
        logger.warning("chroma instance was not empty. using preloaded data")
    else:
        nodes = load_dataset_vector(True)
        index = VectorStoreIndex(
            nodes,  # Pass the list of Document objects
            storage_context=storage_context,
            embed_model=embed_model
        )
        logger.info(f"dataset is loaded and ready in the chromadb instance.")
    logger.debug(f"chroma contains {collection.count()} data points")
except Exception as error:
    logger.critical(f"failed to load data to chromaDB. error: {error}")
    sys.exit(1)

if index is None:
    index = VectorStoreIndex(
        [],  # Initialize with an empty index if no new nodes are added
        storage_context=storage_context,
        embed_model=embed_model,
    )

retrieve_engine = index.as_retriever(embed_model=embed_model, similarity_top_k=5)

if __name__ == '__main__':
    # from llm_model import llm
    #query_engine = index.as_query_engine(embed_model=embed_model, llm_moled = llm)
    response = retrieve_engine.retrieve("There is an ecological collapse")
    print("nº of reponse:", type(response[0]))
    for res in response: print(res, end = "\n\n")