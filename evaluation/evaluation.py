
import tqdm
import pandas as pd
from sklearn.metrics import classification_report
from typing import Dict, List
import statistics

from bert_score import score

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './')

from fact_checker.hier_loaders import EvidenceClaimRetriever
from fact_checker.query_pipeline import verification_pipeline

from llama_index.core import QueryBundle

from tabulate import tabulate

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

EVALUATION_VERSION: int = 0
RETRIEVER_EVALUATION_VERSION: int = 1
BERT_EVALUATION_VERSION: int =0

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
evidence = pd.read_parquet("./data/climate-evidence.parquet.gzip")
evidence['claim_id'] = evidence['claim_id'].astype(int)

# ------------------------------------------------------------------------------
# Método 1: Evaluación de respuestas de la pipeline (lee dataset, llama al pipeline y guarda resultados)
# ------------------------------------------------------------------------------
def evaluate_verification_pipeline(evaluation_version: int = 0) -> None:
    """
    Lee un dataset CSV que contiene claims y sus etiquetas ('label'),
    pasa cada claim por la 'verification_pipeline', guarda la respuesta en la columna 'response'
    y finalmente imprime un classification_report y un CSV con los resultados.
    
    Args:
        evaluation_version (int, optional): Versión del archivo CSV a leer/guardar. 
                                            Por defecto es 0.
    """

    # 1. Leer dataset
    df = pd.read_csv(f"evaluation-dataset-v{evaluation_version}.csv")
    df['label'] = df['label'].str.strip()  # Asegurar que la columna 'label' no tenga espacios extra
    df['response'] = ""  # Inicializa la columna de respuesta en blanco

    print(f"Ejemplo de claim:\n{df['claim'][0]}\n")

    # 2. Proceso de verificación claim por claim
    for i, row in tqdm.tqdm(df.iterrows(), desc="Proceso Evaluación: ", total=df.shape[0]):
        response = verification_pipeline(row["claim"])

        # Normalizar algunas respuestas del pipeline
        if response['result'] == "no evidence" or response['result'] == "not enough evidence": 
            response['result'] = "other"

        df.loc[i, "response"] = response['result']

        # Ver si coincide con la etiqueta original
        correct = "0"
        if response['result'] == row['label']: correct = "1"
        print(response['result'] + ', ' + row['label'] + ', ' + correct)

    # 3. Informe y guardado de resultados
    print("\n=== Classification Report ===")
    print(classification_report(df['response'], df['label']))
    df.to_csv(f"evaluated-dataset-v{evaluation_version}.csv", index=False)

# ------------------------------------------------------------------------------
# Método 2: Evaluación del retriever (cálculo de Recall@K y MRR)
# ------------------------------------------------------------------------------
def evaluate_retriever(
    retriever,
    paraphrased_claims_dataset: List[Dict],
    k: int = 5
):
    """
    Evalúa qué tan bien el retriever recupera las evidencias originales
    cuando se usan versiones reformuladas (parafraseadas) de las claims
    como queries.

    Args:
        retriever: instancia de tu EvidenceClaimRetriever (o similar),
                   con el método _retrieve(query_bundle).
        paraphrased_claims_dataset (List[Dict]): 
            Estructura que contiene al menos:
              - "claim_id": ID de la claim
              - "claim_reformulated": texto de la claim reformulada
              - "evidence_ids": lista de IDs de evidencias relevantes
        k (int): número de evidencias a recuperar (top-k).

    Returns:
        (float, float):
            - recall_at_k
            - mean_reciprocal_rank (MRR)
    """
    total_queries = 0
    recall_count = 0
    reciprocal_ranks = []

    claims_dataset = pd.read_csv(f"./evaluation/evaluation-retriever-v{RETRIEVER_EVALUATION_VERSION}.csv")
    claims_dataset = claims_dataset.to_dict('records')

    for i, claim in enumerate(claims_dataset):
        print(evidence[evidence['claim_id'] == claim['claim_id']]['evidence_id'].tolist())
        claim['evidence_ids'] = evidence[evidence['claim_id'] == claim['claim_id']]['evidence_id'].tolist()
    
    retriever = EvidenceClaimRetriever(claim_top=3, evidence_top=5)

    for claim_data in tqdm.tqdm(paraphrased_claims_dataset, desc="Proceso Evaluación Retriever: ", total=len(paraphrased_claims_dataset)):
        print(claim_data)
        # 1. Texto reformulado que sirve como query
        query_text = claim_data["claim_reformulated"]
        # 2. Conjunto de IDs de evidencias relevantes (del dataset original)
        relevant_evidence_ids = set(claim_data["evidence_ids"])

        # 3. Recuperamos evidencias del retriever
        retrieved_evidences = retriever.retrieve(query_text)

        # Filtramos para quedarnos sólo con las top-k
        retrieved_evidences = retrieved_evidences[:k]

        # Buscamos la posición donde aparece la primera evidencia "relevante"
        # (ID de la evidencia recuperada está en relevant_evidence_ids)
        relevant_positions = []
        for idx, ev in enumerate(retrieved_evidences):
            ev_id = ev.metadata.get("evidence_id", None)
            if ev_id in relevant_evidence_ids:
                relevant_positions.append(idx + 1)  # +1 para índice 1-based

        # 4. Cálculo de Recall@K para esta query
        if len(relevant_positions) > 0:
            # Si al menos una de las evidencias relevantes apareció en top-k
            recall_count += 1

            # 5. Para MRR usamos 1 / rank de la primera evidencia relevante
            first_relevant_rank = min(relevant_positions)
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            # No se recuperó nada relevante en el top-k
            reciprocal_ranks.append(0.0)

        total_queries += 1

    # Métricas globales
    recall_at_k = recall_count / total_queries if total_queries else 0.0
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    print(f"Recall@5: {recall_at_k:.3f}")
    print(f"MRR@5: {mrr:.3f}")


# ------------------------------------------------------------------------------
# Método 3: Evaluación mediante BertScore de las fuentes y la respuesta generada
# ------------------------------------------------------------------------------
def evaluate_bert_score():
    # Supongamos que tienes un texto generado por tu sistema (candidato)
    user_queries = pd.read_csv(f"./evaluation/evaluation-bert-v{BERT_EVALUATION_VERSION}.csv")

    # Listas para almacenar las métricas de cada fila
    precisions = []
    recalls = []
    f1_scores = []

    for i, row in tqdm.tqdm(user_queries.iterrows(), desc="Proceso Evaluación: ", total=user_queries.shape[0]):
        
        response = verification_pipeline(row["claim"])

        if not response['verified']:
            continue

        # chat-gpt (atomic - response)
        candidate_texts = [ atomic['response']  for atomic in response['atomics'] if len(atomic['sources'])>0 ] 

        # Y un texto de referencia (ground truth) con el que compararlo
        # atomic-evidence
        reference_texts = [
            [
                source['evidence'].split(']', 2)[1]
                for source in atomic['sources']
            ]
            for atomic in response['atomics']
            if len(atomic['sources'])>0
        ]

        # Cálculo de las métricas
        P, R, F1 = score(candidate_texts, reference_texts, lang="en")

        # Almacenar las métricas promedio para esta fila
        precisions.append(P.mean().item())
        recalls.append(R.mean().item())
        f1_scores.append(F1.mean().item())

        print(precisions, recalls, f1_scores)

    # Calcular las métricas globales
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # Imprimir las métricas en formato de tabla
    metrics_table = [
        ["Precisión Promedio", avg_precision],
        ["Recall Promedio", avg_recall],
        ["F1 Promedio", avg_f1]
    ]

    print("\nResumen de métricas:\n")
    print(tabulate(metrics_table, headers=["Métrica", "Valor"], tablefmt="grid", floatfmt=".5f"))

"""
Resumen de métricas:

+--------------------+---------+
| Métrica            |   Valor |
+====================+=========+
| Precisión Promedio | 0.87137 |
+--------------------+---------+
| Recall Promedio    | 0.93011 |
+--------------------+---------+
| F1 Promedio        | 0.89785 |
+--------------------+---------+
"""

# ------------------------------------------------------------------------------
# Método4: Evaluación mediante FactScore de las fuentes y la respuesta generada
# ------------------------------------------------------------------------------
def evaluate_factscore() -> List[float]:
    """
    Recorre un dataset de claims, obtiene el texto generado y las evidencias
    de cada atomic en la verificación y calcula FactScore usando 'compute_factscore'.

    Args:
        dataset (List[Dict]): Lista de diccionarios con los datos de cada claim.
                              Se asume que cada elemento tiene al menos la clave "claim".
    
    Returns:
        List[float]: Una lista con los FactScores promediados para cada claim del dataset.
    """
    
    dataset = pd.read_csv(f"./evaluation/evaluation-bert-v{BERT_EVALUATION_VERSION}.csv")

    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    all_factscores = []

    # Recorre cada fila del dataset
    for i, row in tqdm.tqdm(dataset.iterrows(), desc="Proceso Evaluación: ", total=dataset.shape[0]):
        claim_text = row["claim"]

        # 1) Obtenemos la respuesta del pipeline
        response = verification_pipeline(claim_text)

        if not response['verified']:
            continue

        # 2) Extraemos los textos generados en cada atomic
        generated_texts = [atomic['response'] for atomic in response['atomics'] if len(atomic['sources'])>0]

        # 3) Construimos la lista de referencias para cada atomic
        #    Se usa el split(']', 2)[1] para quitar el prefijo "[n]" de la evidencia
        reference_texts_list = [
            [
                source['evidence'].split(']', 2)[1] 
                for source in atomic['sources']
            ]
            for atomic in response['atomics']
            if len(atomic['sources'])>0
        ]

        # 4) Calculamos FactScore para cada atomic y promediamos
        atomic_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts_list):
            fs = compute_factscore(nli_pipeline, gen_text, ref_texts)
            atomic_scores.append(fs)
        print(atomic_scores)
        if atomic_scores:
            avg_factscore = statistics.mean(atomic_scores)
        else:
            # Si no hubiera atomics, evitamos error de mean()
            avg_factscore = 0.0

        all_factscores.append(avg_factscore)

    # Retornamos la lista de FactScores, uno por cada claim
    return all_factscores

def compute_factscore(nli_pipeline, generated_text: str, evidence_texts: list) -> float:
    """
    Computa un 'FactScore' promedio basado en la similitud NLI.
    
    Args:
        generated_text (str): respuesta generada por el sistema.
        evidence_texts (list): lista de textos de evidencia (premisas).
        
    Returns:
        float: un puntaje (0 a 1) de qué tan consistente es 'generated_text'
               con la evidencia.
    """
    # 1. Divide el texto generado en oraciones (muy simplificado aquí).
    #    Podrías usar nltk.sent_tokenize o spacy para algo más robusto.
    sentences = generated_text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    # 2. Para cada oración, mediremos la "fuerza" de entailment
    scores = []

    for sent in sentences:
        # Evitar strings vacíos
        if not sent:
            continue

        # 3. Ver qué tan "respaldada" está la oración en todas las evidencias
        #    (Por simplicidad, tomamos la evidencia que mayor prob. de entailment dé)
        max_entail_score = 0.0

        for ev in evidence_texts:
            # Estructura tipo "premise (evidence) => hypothesis (sent)"
            # Crearemos un input para el pipeline
            to_classify = f"premise: {ev}\nhypothesis: {sent}"
            
            # El pipeline "text-classification" con 'bart-large-mnli' típicamente
            # espera el texto directamente; la parte "premise/hypothesis" es conceptual.
            # Ej: bart-large-mnli a veces funciona con:
            # to_classify = [ev, sent], pero probemos con un input unificado:
            result = nli_pipeline(to_classify, top_k=None, truncation=True)
            # 'result' es una lista de dicts con 'label' y 'score'
            # Ej: [{'label': 'CONTRADICTION', 'score': 0.1}, 
            #       {'label': 'ENTAILMENT', 'score': 0.8}, 
            #       {'label': 'NEUTRAL', 'score': 0.1}]
            
            if isinstance(result, list) and len(result) > 0:
                # Ubicar la prob. de entailment
                entail_score = 0.0
                for d in result:  # dependemos de la estructura devuelta
                    if d['label'] == 'entailment':
                        entail_score = d['score']
                        break

                # Nos quedamos con la mayor que encontremos en cualquier evidence_text
                if entail_score > max_entail_score:
                    max_entail_score = entail_score
            
        # Guardamos la puntuación de esta oración
        scores.append(max_entail_score)

    if not scores:
        return 0.0

    # 4. Tomar la media de las probabilidades de entailment
    fact_score = sum(scores) / len(scores)
    return fact_score

if __name__ == "__main__":

    # Evaluar Salida Final
    # evaluate_verification_pipeline()

    # Evaluar retriever
    # evaluate_retriever(k=5)

    # Evaluar BERT Score
    # evaluate_bert_score()

    # Evaluar Fact Score
    evaluate_factscore()


""" RESULTS """

""" Prueba 8 - Dataset v0

"""

""" Prueba 6 - Dataset v0
              precision    recall  f1-score   support

       other       0.96      0.90      0.93        30
      refute       0.80      0.89      0.84         9
     support       0.91      1.00      0.95        10

    accuracy                           0.92        49
   macro avg       0.89      0.93      0.91        49
weighted avg       0.92      0.92      0.92        49
"""
