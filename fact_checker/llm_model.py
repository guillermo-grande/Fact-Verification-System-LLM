from dotenv import load_dotenv
load_dotenv()

import os
import socket
from ping3 import ping

import logging

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

# Configuración de logging
logger = logging.getLogger("fact-checker.llm-selector")

# URL del endpoint para el modelo Llama 3.2
LLAMA_API_URL = "http://kumo01:11434/api/generate"

def test_llama() -> bool:
    # ping llama instance
    try: found = ping("http://kumo01:11434")
    except socket.error: found = False

    # report to user
    if found: logger.info("using llama Local Model")
    else: logger.warning("failed to reach llama local install. Defaults to using OpenAI")
    
    return found

LLAMA_AVAILABLE: bool = test_llama()
if not LLAMA_AVAILABLE:
    # Leer la API key desde el archivo .env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("La API key de OpenAI no se encontró en el archivo .env.")
        raise ValueError("Falta la API key de OpenAI en el archivo .env.")

    llm = OpenAI(model = "gpt-4o-mini", temperature=0.1, max_tokens=200)
else:
    llm = Ollama("llama2:latest", "http://kumo01:11434", temperature=0.1, request_timeout=120.0)
