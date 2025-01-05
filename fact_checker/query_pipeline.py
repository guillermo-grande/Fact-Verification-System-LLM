# this file will be run as exec while dev-testing
if __name__ == '__main__':
    import sys
    from pathlib import Path
    root  =  Path(__file__).resolve().parent.parent
    sys.path.append(str(root))


from pprint import pprint
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

from transformers import pipeline
from langdetect import detect

#----------------------------------------------------------------------------------------
# Translator
#----------------------------------------------------------------------------------------

# # Use a pipeline as a high-level helper
# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")
# logger.debug("loaded translation model: Helsinki-NLP/opus-mt-en-mul")
    

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

TRANSLATION_VERSION: int = 0
TRANSLATION: str = load_prompt("translation", TRANSLATION_VERSION)
PROMPT_TRANSLATION = PromptTemplate(TRANSLATION)


#--------------------------------------------------------------------
# Translation
#--------------------------------------------------------------------
def translate_text(text: str, src_lang: str, dst_lang: str) -> str:
    if src_lang == dst_lang: return text
    
    translation_prompt = TRANSLATION.format()
    translation_response = llm.complete(translation_prompt).text

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
def verification_pipeline(user_query: str) -> str:
    """
    Runs the full pipeline

    Args:
    - query (str). User query with the original claims
    
    Returns:
    - str. Text with full response.
    """
    # translate the query if it is not in English
    input_language = detect(user_query)
    # if input_language != "en": 
    #     query = translate_query(user_query, source_language=input_language, target_language="en")
    # else:
    #     query = user_query
        
    query = user_query

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
        verified = True
    else:
        consolidation_response = "No evidence. The database does not contain evidence to answer the claim."
        verified = False
    
    ret =  {
        'claim': query,
        'verified': verified,
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

# # Define a function for translation
# def translate_text(text):
#     # Translate the text using the pipeline
#     translation = pipe(text)
#     # The output is a list of dictionaries, so we extract the translated text
#     return translation[0]['translation_text']

if __name__ == "__main__":
    
    # Pregunta del usuario
    user_query = "Costs of burning fossil fuels are higher than the costs of renewable energy sources."
    
    #translated_text = translate_text(user_query)

    #pprint(translated_text)

    
    # Consultar y generar respuesta
    # try:
    # documents = verification_pipeline(query)
    # except Exception as e:
    #     logger.error(f"Error al ejecutar la consulta: {e}")
