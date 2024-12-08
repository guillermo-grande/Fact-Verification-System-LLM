import os
import logging

import pandas as pd
from datasets import load_dataset

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core import StorageContext
from llama_index.core import Settings

import chromadb
from chromadb.config import Settings as ChromaDBSettings
from llama_index.vector_stores.chroma import ChromaVectorStore

# display options
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 10)

# logging settings
logger = logging.getLogger("data-loader")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")
stream = logging.StreamHandler()
stream.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(stream)

DOWNLOAD_LOCATION: str = "data"
if not os.path.exists(DOWNLOAD_LOCATION): os.mkdir(DOWNLOAD_LOCATION)
def download_climate_fever(streaming: bool = False): 
    """Load HuggingFace dataset `climate-fever` with a predefined cache directory.
    Args:
    * streaming (bool, optional). Load data in streaming mode. Defaults False
    
    """
    logger.info("climate fever loading")
    return load_dataset("tdiggelm/climate_fever", cache_dir = DOWNLOAD_LOCATION, streaming = streaming)

def process_climate_fever(save=True):
    ds = download_climate_fever() 
    df = ds["test"].to_pandas()
    
    def accumilate_evidence(evidence_list):
        sep = "\n*"
        complete_evidence = sep
        for evidence in evidence_list:
            complete_evidence = complete_evidence + sep + evidence["evidence"]
        return complete_evidence
        
    df["evidence"] = df["evidences"].apply(accumilate_evidence)
    df["text"]  = "Evidence: " + df["evidence"] + "\nClaim: " + df["claim"]
    df["label"] = df["claim_label"]
    df["unanswerable"] = df["label"] == 2 # wheter it is not supported 

    if save:
        df.to_csv("data/climate_fever.csv", index = False)
    return df

#----------------------------------------------------------------------------------------
# Vector Database
#----------------------------------------------------------------------------------------

# Load the embedding model
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Configuración de ChromaDB
if not os.path.exists(DOWNLOAD_LOCATION + "/chromadb"): os.mkdir(DOWNLOAD_LOCATION + "/chromadb")
chroma_client = chromadb.Client(ChromaDBSettings(
    persist_directory=DOWNLOAD_LOCATION + "/chromadb"  # Ruta para almacenar los datos
))

collection = chroma_client.get_or_create_collection(name="index_climate_fever")
vector_store = ChromaVectorStore(chroma_collection=collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

def load_dataset_vector(preloaded: bool = True):
    if preloaded and os.path.exists("data/climate_fever.csv"):
        df = pd.read_csv("data/climate_fever.csv")
    else:
        df = process_climate_fever()

    # create data iterator
    text, metadata, ids = [], [], []
    for _, row in df.iterrows():
        ids.append(row['claim_id'])
        text.append(row['text'])
        metadata.append(dict(id=row['claim_id'], support=row['unanswerable']))
    
    collection.add(documents = text, metadatas = metadata, ids = ids)
    return documents, metadata, ids

documents, metadata, ids = load_dataset_vector(True)
index = VectorStoreIndex.from_documents(
    documents=documents, 
    storage_context=storage_context, 
    embed_model=embed_model
)
