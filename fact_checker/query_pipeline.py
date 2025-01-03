from collections import defaultdict
from dotenv import load_dotenv; load_dotenv() # load API KEYs

import os
import requests

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
    for claim_id, atomic in enumerate(atomic_claims):

        support   = defaultdict(lambda : 0)
        evidences = retrieve_engine.retrieve(atomic)
        
        print("claim:", atomic)
        for e in evidences:
            label = map_evidence_label[str(e.metadata.get("evidence_label"))]
            support[label] += e.metadata.get('entropy') 
            print(f"Evidencia: 1\nText:{e.text}\nevidence_label: {label}\nentropy: {e.metadata.get('entropy')}")
            print()
        
        print("soporte: ", dict(support))
        print()
        all_evidences.extend(evidences)

    """
    
    """

    return all_evidences

if __name__ == "__main__":
    # Pregunta del usuario
    query = "La capa de ozono está abierta sobre el mar rojo, cuyo nivel está subiendo debido a el cambio climático"
    
    # Consultar y generar respuesta
    # try:
    documents = verification_pipeline(query)
    # except Exception as e:
    #     logger.error(f"Error al ejecutar la consulta: {e}")
