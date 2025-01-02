
import re
from llm_model import llm, LLAMA_AVAILABLE
from utils import load_prompt

PROMPT_VERSION: int = 1
DECOMPOSE_PROMPT = load_prompt("decompose", PROMPT_VERSION)

if LLAMA_AVAILABLE: 
    raise NotImplementedError("Llama Model is not supported yet")

response_pattern = re.compile("\* ([\w\s\d]*)")
def decompose_query(query: str) -> list[str]:
    """
    Decompose a given query into a series of multiple atomic claims.
    This function uses the default LLM model.

    Args:
        * query (str). user query with one or multiple claims.
    Returns:
        * list[str]. list of atomic claims.
    """
    prompt   = DECOMPOSE_PROMPT.format(query)
    # print(f"prompt: \n{prompt}\n")

    response = llm.complete(prompt)
    response = str(response)
    # print(f"full response: \n{response}")

    # split in lines and remove *
    processed_response = [line[2:] for line in response.split("\n")]
    return processed_response

if __name__ == '__main__':
    test_claim = "La capa de ozono está abierta sobre el mar rojo, cuyo nivel está subiendo debido a el cambio climático"
    atomic_claims = decompose_query(test_claim)
    print("atomic claims: ")
    for i, claim in enumerate(atomic_claims):
        print(f"[{i:>3d} ] {claim}")
        