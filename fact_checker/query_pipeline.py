from collections import defaultdict
from dotenv import load_dotenv; load_dotenv() # load API KEYs

from llama_index.core.schema import NodeWithScore

import os
import requests
from numpy import argmin 

# Configuración de logging
import logging
from   utils import configure_logger
logger = logging.getLogger("fact-checker")
logger = configure_logger(logger, 'INFO')

from llm_model    import llm, LLAMA_AVAILABLE
from decomposer   import decompose_query
from utils        import load_prompt
from hier_loaders import EvidenceClaimRetriever

#--------------------------------------------------------------------
# Data Retriever
#--------------------------------------------------------------------
retrieve_engine = EvidenceClaimRetriever(3, 25)


# URL del endpoint para el modelo Llama 3.2
LLAMA_API_URL = "http://kumo01:11434/api/generate"

def query_llama(prompt: str, temperature: float = 0.2, max_tokens: int = 256):
    """
    Consulta el modelo a través de una API remota.

    Args:
        prompt (str): Texto de entrada para el modelo.
        temperature (float): Controla la aleatoriedad en las respuestas.
        max_tokens (int): Límite de longitud de la respuesta generada.

    Returns:
        str: Respuesta generada por el modelo.
    """
    try:
        logger.info("Consultando el modelo")
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(LLAMA_API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info("Respuesta recibida del modelo.")
            return response.json().get("text", "")
        else:
            logger.error(f"Error en la consulta al modelo: {response.status_code} {response.text}")
            response.raise_for_status()
    except Exception as e:
        logger.critical(f"Error al consultar el modelo Llama 3.2: {e}")
        raise

def verification_consensus(claim: str, evidence: list[NodeWithScore]) -> tuple[int, str]:
    """Calls an LLM model to determine the veracity of an atomic claim based on a list of evidence
    notes. The model uses determine if the evidence supports or refutes the claim or if there is no
    enough evidence to generate a conclusion.

    Args:
        claim (str): atomic claim that is going to be verified.
        evidence (list[NodeWithScore]): list of evidences text nodes. 

    Returns:
        int: integer value of the conclusion. It has the same map as the evidence label.
        str: full message of the llm response. This can be used to report to the user why it obtained the results.
    """
    # TODO
    # my recommeendation is to use QueryEngine so it returns the text with the citations.
    # https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/

    return 0, ""

def verification_pipeline(query: str) -> str:
    """
    Runs the full pipeline

    Args:
    - query (str). User query with the original claims
    
    Returns:
    - str. Text with full response.
    """
    # decompose into multiple atomic claims
    # atomic_claims = decompose_query(query)
    atomic_claims = [
        "The ozone layer is open over the Red Sea.", 
        "The sea level of the Red Sea is rising.",
        "The rising sea level of the Red Sea is due to climate change."
    ]

    # claim grande -> atomic -> [yes | no] -> yes -> no (no se puede, pq esta claim esta dentro es falsa)

    map_evidence_label = {
        '0': 'supports',
        '1': 'refutes',
        '2': 'not_enough_information'
    }

    # get claims
    all_evidences = []
    all_consensus = []
    for claim_id, atomic in enumerate(atomic_claims):
        evidences = retrieve_engine.retrieve(atomic)

        if len(evidences) == 0: 
            # TODO: deal with this case. It is neccesary
            print("soporte: ni idea")
            continue

        consensus = verification_consensus(atomic, evidences)
        all_consensus.append(consensus)
        all_evidences.extend(evidences)
        
        # support   = {0: [], 1: [], 2: []}
        # print("claim:", atomic)
        # for e in evidences:
        #     label = map_evidence_label[str(e.metadata.get("evidence_label"))]
        #     label_id = int(e.metadata.get("evidence_label"))
        #     support[label_id].append(e.metadata.get("entropy"))

        #     print(f"Evidencia: 1\nText:{e.text}\nevidence_label: {label}\nentropy: {e.metadata.get('entropy')}")
        #     print()


        # print(support)
        # avg_support = {}
        # for id in support.keys(): 
        #     if len(support[id]) > 0: avg_support[id] = sum(support[id]) / len(support[id])
        
        # min_id = min(avg_support, key = avg_support.get)
        # print(f"soporte: {map_evidence_label[str(min_id)]} (score: {avg_support[min_id]})")
        # print()


    return all_consensus

if __name__ == "__main__":
    # Pregunta del usuario
    query = "La capa de ozono está abierta sobre el mar rojo, cuyo nivel está subiendo debido a el cambio climático"
    
    # Consultar y generar respuesta
    # try:
    documents = verification_pipeline(query)
    # except Exception as e:
    #     logger.error(f"Error al ejecutar la consulta: {e}")
