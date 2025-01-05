# this file will be run as exec while dev-testing
if __name__ == '__main__':
    import sys
    from pathlib import Path
    root  =  Path(__file__).resolve().parent.parent
    sys.path.append(str(root))


from dotenv import load_dotenv; load_dotenv() # load API KEYs

# Configuración de logging
import  logging
from   fact_checker.utils import configure_logger
logger = logging.getLogger("fact-checker")
logger = configure_logger(logger, 'INFO')

from fact_checker.llm_model    import llm, LLAMA_AVAILABLE
from fact_checker.utils        import load_prompt
from fact_checker.hier_loaders import EvidenceClaimRetriever, EvidenceEnum

from fact_checker.decomposer   import decompose_query

from collections import defaultdict
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode, get_response_synthesizer

import re

#--------------------------------------------------------------------
# Data Retriever
#--------------------------------------------------------------------
retrieve_engine = EvidenceClaimRetriever(3, 25)

#--------------------------------------------------------------------
# Prompts Static Loading
#--------------------------------------------------------------------
VERIFICATION_VERSION: int = 2
ATOMIC_VERIFICATION: str = load_prompt("verification", VERIFICATION_VERSION)
PROMPT_VERIFICATION = PromptTemplate(ATOMIC_VERIFICATION)

CONSOLIDATION_VERSION: int = 0
CLAIM_CONSOLIDATION: str = load_prompt("consolidation", CONSOLIDATION_VERSION)
PROMPT_CONSOLIDATION = PromptTemplate(CLAIM_CONSOLIDATION)


#--------------------------------------------------------------------
# Verification Model
#--------------------------------------------------------------------
conclusion_pattern = re.compile('CONCLUSION: "([\w\s\d]*)"')
def verification_consensus(claim: str) -> tuple[EvidenceEnum, str]:
    """Calls an LLM model to determine the veracity of an atomic claim based on a list of evidence
    notes. The model uses determine if the evidence supports or refutes the claim or if there is no
    enough evidence to generate a conclusion.

    Args:
        claim (str): atomic claim that is going to be verified.

    Returns:
        EvidenceEnum: integer value of the conclusion. It has the same map as the evidence label.
        str: full message of the LLM response. This can be used to report to the user why it obtained the results.
    """
    # TODO
    # my recommendation is to use QueryEngine so it returns the text with the citations.
    # https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/citation_query_engine/

    # create citation engine
    # response synthe
    response_synthetizer = get_response_synthesizer(
        llm = llm,
        text_qa_template = PROMPT_VERIFICATION, 
        response_mode = ResponseMode.COMPACT,
    )
    citation_engine = CitationQueryEngine(retrieve_engine, llm = llm, response_synthesizer=response_synthetizer)

    # Query the citation engine
    response = citation_engine.query(claim)

    # process text
    if len(response.source_nodes) == 0: 
        # # TODO: no determine what to return
        response.response = "Sorry, there is not enough information about this topic in the database"
        return EvidenceEnum.NO_EVIDENCE, response
    
    claim_check = conclusion_pattern.match(response.response).group(1)
    return EvidenceEnum.from_str(claim_check), response

#--------------------------------------------------------------------
# Verification Pipeline
#--------------------------------------------------------------------
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

    atomic_claims = [
        "The ozone layer is open over the Red Sea.",                    # no evidence 
        "The sea level of the Red Sea is rising.",                      # no evidence
        "The rising sea level of the Red Sea is due to climate change." # not enough
    ]

    # get claims
    all_consensus = []
    consolidation_atomics = ""
    for claim_id, atomic in enumerate(atomic_claims):
        decision, response = consensus = verification_consensus(atomic)
        all_consensus.append((atomic, *consensus))
        
        consensus_atomic = f"atomic: {atomic}\nvalidation: {str(decision)}"
        consolidation_atomics += consensus_atomic + "\n"

    """
    consolidation_atomics:

    atomic: The ozone layer is open over the Red Sea.
    validation: no evidence
    atomic: The sea level of the Red Sea is rising.
    validation: no evidence
    atomic: The rising sea level of the Red Sea is due to climate change.
    validation: not enough evidence
    """
    
    consolidation_prompt = PROMPT_CONSOLIDATION.format(query=query, atomics=consolidation_atomics)
    consolidation_response = llm.complete(consolidation_prompt)
    print("consolidation response: ")
    print(consolidation_response.text)

    ret =  {
        'claim': query,
        'general': consolidation_response.text,
        'atomics': [
            {
                'atomic'   : consensus[0], 
                'consensus': str(consensus[1]),
                'response' : str(consensus[2]) ,
                'sources'  : [
                    {
                        'article': source.node.metadata.get("article") ,
                        'evidence': source.node.get_text()
                    }
                    for source in consensus[2].source_nodes
                ]
            }
            for consensus in all_consensus
        ]
    }


    from pprint import pprint
    from json   import dumps
    print("final output: ")
    print(dumps(ret, indent = 2, ))

    return all_consensus

if __name__ == "__main__":
    
    # Pregunta del usuario
    query = "Los osos polares se mueren por el cambio climático."
    
    # Consultar y generar respuesta
    # try:
    documents = verification_pipeline(query)
    # except Exception as e:
    #     logger.error(f"Error al ejecutar la consulta: {e}")
