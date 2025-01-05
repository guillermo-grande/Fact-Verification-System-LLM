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

#from transformers import MarianMTModel, MarianTokenizer
#from langdetect import detect

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

CONSOLIDATION_VERSION: int = 2
CLAIM_CONSOLIDATION: str = load_prompt("consolidation", CONSOLIDATION_VERSION)
PROMPT_CONSOLIDATION = PromptTemplate(CLAIM_CONSOLIDATION)


#--------------------------------------------------------------------
# Verification Model
#--------------------------------------------------------------------
conclusion_filter  = re.compile('CONCLUSION: "([\w\s\d]*)"\s*EXPLANATION:\s+((.|\n)*)')
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
    
    claim_group = conclusion_filter.match(response.response)
    claim_check = claim_group.group(1)
    response.response = claim_group.group(2)
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

    # get claims
    all_consensus = []
    evidence_found = False
    consolidation_atomics = ""
    for claim_id, atomic in enumerate(atomic_claims):
        decision, _ = consensus = verification_consensus(atomic)
        all_consensus.append((atomic, *consensus))
        evidence_found = evidence_found or decision != EvidenceEnum.NO_EVIDENCE
        # print(f"{claim_id:>2d} - validation: {str(decision)} - atomic: {atomic}")

        consensus_atomic = f"atomic: {atomic}\nvalidation: {str(decision)}"
        consolidation_atomics += consensus_atomic + "\n"
    
    if evidence_found:
        consolidation_prompt = PROMPT_CONSOLIDATION.format(query=query, atomics=consolidation_atomics)
        consolidation_response = llm.complete(consolidation_prompt).text
    else:
        consolidation_response = "No evidence. The database does not contain evidence to answer the claim."
    
    ret =  {
        'claim': query,
        'general': consolidation_response,
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
    
    return ret
'''
def translate_query(query, source_language, target_language):
    model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
'''
if __name__ == "__main__":
    
    # Pregunta del usuario
    user_query = "La quema de combustibles fósiles es la principal causa del cambio climático."
    '''
    input_language = detect(user_query)
    if input_language != "en":  # Si no está en inglés, traduce
        query = translate_query(user_query, target_language="en")
    '''
    # Consultar y generar respuesta
    # try:
    documents = verification_pipeline(user_query)
    # except Exception as e:
    #     logger.error(f"Error al ejecutar la consulta: {e}")
