from dotenv import load_dotenv
load_dotenv()

import os
import logging
import requests

# Configuración de logging
logger = logging.getLogger("fact-checker") # .query-pipeline
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream = logging.StreamHandler()
stream.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(stream)

from llm_model import llm, LLAMA_AVAILABLE
from decomposer import decompose_query
from data_loaders import retrieve_engine
from utils import load_prompt



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

def query_retriever(query: str, retriever):
    """
    Ejecuta una consulta al retriever y genera una respuesta con Llama 3.2.

    Args:
        query (str): Pregunta del usuario.
        retriever: Motor de recuperación.

    Returns:
        str: Respuesta generada.
    """
    try:
        logger.info("Ejecutando consulta al retriever.")
        retrieved_documents = retriever.retrieve(query)
        logger.info(f"Documentos recuperados: {len(retrieved_documents)}")

        # Construir contexto para el modelo (Cambiar para seguir metodología del paper)
        context = "\n\n".join([doc.text for doc in retrieved_documents])
        prompt = f"Contexto: {context}\n\nPregunta: {query}\n\nRespuesta:"

        # Consultar
        response = query(prompt)
        return response
    except Exception as e:
        logger.critical(f"Error durante la consulta: {e}")
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
    atomic_claims = decompose_query(query)
    # atomic_claims = [
    #     "The ozone layer is open over the Red Sea.",
    #     "The sea level of the Red Sea is rising.",
    #     "The rising sea level of the Red Sea is due to climate change."
    # ]

    # get claims
    for a, atomic in enumerate(atomic_claims):

        documents = retrieve_engine.retrieve(atomic)

        print(f"[ {a:2d} ] claim: {atomic}")
        print(f"[ {a:2d} ] nº retrieved documents:", len(documents))
        for i, doc in enumerate(documents, start = 1):
            print(f"[ {a:2d}-{i:03d} ]{repr(doc.text[:50])}")
        print("")
    
    return ""

if __name__ == "__main__":
    # Pregunta del usuario
    query = "La capa de ozono está abierta sobre el mar rojo, cuyo nivel está subiendo debido a el cambio climático"

    # Consultar y generar respuesta
    try:
        verification_pipeline(query)
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta: {e}")
