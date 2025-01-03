import os
import sys
import logging

from typing import Tuple, List

import pandas as pd
from datasets import load_dataset
from operator import getitem, attrgetter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

from llama_index.core.vector_stores import FilterOperator, FilterCondition
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
from llama_index.core.schema import NodeWithScore

from llama_index.core.retrievers import BaseRetriever

import chromadb

# logging
logger = logging.getLogger("fact-checker.data-loader")
logger.setLevel(logging.DEBUG)

#----------------------------------------------------------------------------------------
# Embedding Model
#----------------------------------------------------------------------------------------

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
logger.debug(f"loaded embedding model: {embed_model_id}")

#----------------------------------------------------------------------------------------
# Dataset Loading / Pre-processing
#----------------------------------------------------------------------------------------

DOWNLOAD_LOCATION: str = "data"
if not os.path.exists(DOWNLOAD_LOCATION): os.mkdir(DOWNLOAD_LOCATION)
def download_climate_fever(streaming: bool = False): 
    """Load HuggingFace dataset `climate-fever` with a predefined cache directory.

    Args:
    * streaming (bool, optional). Load data in streaming mode. Defaults False
    
    """
    try:
        logger.debug("loading climate-fever dataset from hugging-face")
        return load_dataset("tdiggelm/climate_fever", cache_dir = DOWNLOAD_LOCATION, streaming = streaming)
    except:
        logger.critical("failed to load climate-fever dataset")
        sys.exit(1)

def build_evidence_frame(original: pd.DataFrame) -> pd.DataFrame:
    """Builds the evidence-only dataset from the original dataset from climate-fever.
    This function expands the `evidences` column into a full dataframe. The columns
    claim_id that are added to the results are carried from the original row. 

    Args:
        original (pd.DataFrame): original climate-fever dataset

    Returns:
        pd.DataFrame: new evidence-only dataset
    """
    evidence_df = None
    for _, row in original.iterrows():
        evidence = pd.DataFrame(list(row['evidences']))
        evidence['claim_id'] = row['claim_id'] # claim id is added to filter in the retriever
        evidence_df = pd.concat([evidence_df, evidence])

    evidence_df['id'] = evidence_df.index.values
    evidence_df = evidence_df.drop('votes', axis = 1)

    evidence_df['evidence'] = evidence_df['evidence'] \
        .str.lower()\
        .str.strip()\
        .str.replace('"', '')
    
    evidence_df =  evidence_df.rename(columns = {'evidence': 'text'})

    return evidence_df

def process_climate_fever(save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-processing pipeline of the original climate-fever dataset

    Args:
        save (bool, optional): wheter to save or not the processed data. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: evidence and claim dataframes.
    """
    dataset  = download_climate_fever() 
    original = dataset["test"].to_pandas()

    # build each respective dataset
    evidence = build_evidence_frame(original)

    claims   = original.drop('evidences', axis = 1)
    claims   = claims.rename(columns = {'claim': 'text'})
    claims['id'] = claims['claim_id'].values

    # save processed data if it's indicated
    if save:
        evidence.to_parquet("data/climate-evidence.parquet.gzip", index = False, compression='gzip')
        logger.info("evidence datataset is saved in: data/climate-evidence.parquet.gzip")
        claims.to_parquet("data/climate-claims.parquet.gzip", index = False, compression='gzip')
        logger.info("claims datataset is saved in: data/climate-claims.parquet.gzip")

    return evidence, claims

def download_get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get or Download and Preprocess the climate fever dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: evidence and claim dataframes.
    """
    if  not os.path.exists("data/climate-evidence.parquet.gzip") or \
        not os.path.exists("data/climate-claims.parquet.gzip"):
        logger.warning("climate-fever was not pre-downloaded. Dataset is going to be downloaded")
        evidence, claims = process_climate_fever(True)
    else:
        logger.warning("pre-downloaded dataset was found. ")
        evidence = pd.read_parquet("data/climate-evidence.parquet.gzip")
        claims   = pd.read_parquet("data/climate-claims.parquet.gzip")

    return evidence, claims

#----------------------------------------------------------------------------------------
# Chroma DB Client
#----------------------------------------------------------------------------------------
CHROMA_PATH: str = "data/chromadb/"
os.makedirs(CHROMA_PATH, exist_ok=True) # make sure the folder exists

# create client
client = chromadb.PersistentClient(path=CHROMA_PATH)

#----------------------------------------------------------------------------------------
# Retrievers
#----------------------------------------------------------------------------------------

def frame_document(data: pd.DataFrame) -> list[TextNode]:
    """Transforms a dataframe into a list of documents. Each row in
    the dataframe is transform into a document. This function
    requieres that the dataframe contains the columns `id` and `text`. The
    rest of columns are considered metadata

    Args:
        data (pd.DataFrame): original dataframe

    Returns:
        list[TextNode]: list of documents. 
    """
    assert 'id' in data.columns and 'text' in data.columns, \
        "The dataframe must contains the columns 'id' and 'text'"
    
    documents = []
    for record in data.to_dict('records'):
        did  = record.pop('id'  , None)
        text = record.pop('text', None)
        doc  = TextNode(id = did, text = text, metadata = record)
        documents.append(doc)
    
    return documents


def create_vector_index(collection_name: str, data: pd.DataFrame) -> VectorStoreIndex:
    """Creates or loads and vector store index by using a collection of data points from chromadb.
    If the collection is empty (or is created from zero), this function populates the database using
    the original dataset information. 

    It is expected that if the folder `data/chromadb` is deleted, this function will have a slower start.
    Following runs won't have any problem. A logging message with level info will notify if there was any available data point.

    Args:
        collection_name (str): name of the collection with data to retriever
        data (pd.DataFrame): collection of data points. if the collection is empty, they are not used.
    Returns:
        VectorStoreIndex: The loaded or newly created index.
    """
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() == 0:
        logger.info(f"no record found in {collection_name} collection. Populating DB from dataset")
        documents = frame_document(data)
        index = VectorStoreIndex(
            documents,
            storage_context=storage_context,
            embed_model=embed_model
        )
    else:
        logger.info(f"{collection.count()} records found in {collection_name} collection. Using preloaded data")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
    
    return index

#----------------------------------------------------------------------------------------
# Rationale Retriever
#----------------------------------------------------------------------------------------

class EvidenceClaimRetriever(BaseRetriever):
    CLAIM_THRESS: float = 0.4
    EVIDENCE_THRESS: float = 0.4

    def __init__(self, claim_top: int = 3, evidence_top: int = 5):
        """This retriever corresponds with the rationale extraction pipeline from a RAG-System. Based on a series of atomic claims, the
        retriever obtains a list of evidences and they are used to determine the rationale of the original claim. 

        Args:
            claim_top (int, optional): number of claims that are retrieved for each atomic claim. Defaults to 3
            evidence_top (int, optional): number of evidences that are retrieved for each retrieved claim. Defaults to 5
        """
        # initialize inner retriever
        evidence, claims = download_get_dataset()

        self.claims_index = create_vector_index("claim", claims)
        self.evidence_index = create_vector_index("evidence", evidence)

        # create retriever engines
        self.claims_retriever = self.claims_index.as_retriever(embed_model=embed_model, similarity_top_k=claim_top)
        # self.evidence_retriever = self.evidence_index.as_retriever(embed_model=embed_model, similarity_top_k=evidence_top)

        # retrieve thress        
        self.claim_top = claim_top
        self.evidence_top = evidence_top

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for given query."""
        claims = self.claims_retriever.retrieve(query_bundle)

        # Filter by score
        filtered_claims = list(filter(lambda c: c.score >= self.CLAIM_THRESS, claims))        
        if len(filtered_claims) == 0: return []

        claim_ids = [claim.metadata.get('claim_id') for claim in filtered_claims]

        # Define metadata filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="claim_id", operator = FilterOperator.IN, value = claim_ids)
            ]
        )
    
        # Initialize the evidence retriever with the filter
        # this is required to set each time the filters as they cann't be set at query time.
        evidence_retriever = self.evidence_index.as_retriever(
            filters=filters,
            embed_model=embed_model,
            similarity_top_k=self.evidence_top
        )
        
        # claims -> evidences (metadata-filter: [claims-id])
        evidences = evidence_retriever.retrieve(query_bundle)
        evidences = list(filter(lambda c: c.score >= self.EVIDENCE_THRESS, evidences))        
        
        return evidences

if __name__ == '__main__':
    from utils import configure_logger

    logger = configure_logger(logger, 'INFO')
    retriever = EvidenceClaimRetriever(3, 5)

    query     = "The ozone layer is broken due to the climate change"
    claims    = retriever.retrieve(query)    
    for p in claims:
        print(f"evidence: {p.id_} ({p.metadata.get('claim_id')}) score: {p.score} \ntext: {p.text}", end = "\n\n")