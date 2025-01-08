import time
import tqdm
import pandas as pd
from sklearn.metrics import classification_report
from typing import Dict, List
import statistics

from bert_score import score

import sys
sys.path.insert(1, './')

from fact_checker.hier_loaders import EvidenceClaimRetriever
from fact_checker.query_pipeline import verification_pipeline

from llama_index.core import QueryBundle

from tabulate import tabulate

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

EVALUATION_VERSION: int = 0
RETRIEVER_EVALUATION_VERSION: int = 1
BERT_EVALUATION_VERSION: int = 0
FACTSCORE_EVALUATION_VERSION: int = 0
TIME_EVALUATION_VERSION: int = 1

# ------------------------------------------------------------------------------
# Carga de modelos
# ------------------------------------------------------------------------------

model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
evidence = pd.read_parquet("./data/climate-evidence.parquet.gzip")
evidence['claim_id'] = evidence['claim_id'].astype(int)

# ------------------------------------------------------------------------------
# Método 1: Evaluación de respuestas de la pipeline (lee dataset, llama al pipeline y guarda resultados)
# ------------------------------------------------------------------------------
def evaluate_verification_pipeline() -> None:
    """
    Lee un dataset CSV que contiene claims y sus etiquetas ('label'),
    pasa cada claim por la 'verification_pipeline', guarda la respuesta en la columna 'response'
    y finalmente imprime un classification_report y un CSV con los resultados.
    
    Args:
        evaluation_version (int, optional): Versión del archivo CSV a leer/guardar. 
                                            Por defecto es 0.
    """

    # 1. Leer dataset
    df = pd.read_csv(f"evaluation/evaluation-dataset-v{EVALUATION_VERSION}.csv")
    df['label'] = df['label'].str.strip()  # Asegurar que la columna 'label' no tenga espacios extra
    df['response'] = ""  # Inicializa la columna de respuesta en blanco

    #print(f"Ejemplo de claim:\n{df['claim'][0]}\n")

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
    df.to_csv(f"evaluation/evaluated-dataset-v{EVALUATION_VERSION}.csv", index=False)

""" RESULTS """

""" Prueba Final - Dataset v0
=== Classification Report ===
              precision    recall  f1-score   support

       other       0.96      0.75      0.84        32
      refute       0.80      0.95      0.87        21
     support       0.88      1.00      0.94        22

    accuracy                           0.88        75
   macro avg       0.88      0.90      0.88        75
weighted avg       0.89      0.88      0.88        75
"""

# ------------------------------------------------------------------------------
# Método 2: Evaluación del retriever (cálculo de Recall@K y MRR)
# ------------------------------------------------------------------------------
def evaluate_retriever(
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

    for claim_data in tqdm.tqdm(claims_dataset, desc="Proceso Evaluación Retriever: ", total=len(claims_dataset)):
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

    all_factscores = []

    for i, row in tqdm.tqdm(dataset.iterrows(), desc="Proceso Evaluación: ", total=dataset.shape[0]):
        claim_text = row["claim"]

        response = verification_pipeline(claim_text)

        if not response['verified']:
            continue

        atomic_claims = [atomic['atomic'] for atomic in response['atomics'] if len(atomic['sources'])>0]

        atomic_consensus = [atomic['consensus'] for atomic in response['atomics'] if len(atomic['sources'])>0]

        for atomic in response['atomics']:
            for source in atomic['sources']:
                aux = source['evidence'].split(']')
                aux.pop(0)
                source['evidence'] = "".join(x for x in aux)
        reference_texts_list = [
            [
                source['evidence']
                for source in atomic['sources']
            ]
            for atomic in response['atomics']
            if len(atomic['sources'])>0
        ]

        atomic_scores = []
        for gen_text, consensus, ref_texts in zip(atomic_claims, atomic_consensus, reference_texts_list):
            if consensus == "support":
                consensus = 0
            elif consensus == "refute":
                consensus = 1
            else:
                consensus = 2
            fs = compute_factscore(gen_text, consensus, ref_texts)
            print(fs)
            atomic_scores.append(fs)
        if atomic_scores:
            avg_factscore = statistics.mean(atomic_scores)
        else:
            avg_factscore = 0.0

        all_factscores.append(avg_factscore)

    print("Factscore promedio: ",statistics.mean(all_factscores))

    return all_factscores

def compute_factscore(generated_text: str, consensus: int, evidence_texts: list) -> float:
    """
    Computa un 'FactScore' promedio basado en la similitud NLI.
    
    Args:
        generated_text (str): respuesta generada por el sistema.
        evidence_texts (list): lista de textos de evidencia (premisas).
        
    Returns:
        float: un puntaje (0 a 1) de qué tan consistente es 'generated_text'
               con la evidencia.
    """

    evidence_text = "\n".join(ev for ev in evidence_texts)

    to_classify = f"premise: {evidence_text}\nhypothesis: {generated_text}"
    
    result = nli_pipeline(to_classify, top_k=None, truncation=True)

    if(consensus==0):
        for d in result:
            if d['label'] == 'entailment':
                fact_score = d['score']
    elif(consensus==1):
        for d in result:
            if d['label'] == 'contradiction':
                fact_score = d['score']
    else:
        for d in result:
            if d['label'] == 'neutral':
                fact_score = d['score']

    return fact_score


def evaluate_time_response():
    """
    Este método evalúa el tiempo de respuesta del pipeline de verificación para cada consulta
    y calcula la media al finalizar.
    """
    user_queries = pd.read_csv(f"./evaluation/evaluation-bert-v{TIME_EVALUATION_VERSION}.csv")
    total_time = 0
    response_times = []  # Lista para almacenar tiempos individuales

    for i, row in tqdm.tqdm(user_queries.iterrows(), desc="Proceso Evaluación: ", total=user_queries.shape[0]):
        start_time = time.perf_counter()  # Usa perf_counter para mayor precisión
        response = verification_pipeline(row["claim"])  # Llama al pipeline de verificación
        end_time = time.perf_counter()  # Finaliza la medición del tiempo
        
        response_time = end_time - start_time  # Calcula el tiempo de respuesta
        response_times.append(response_time)  # Almacena el tiempo individual
        total_time += response_time  # Suma al tiempo total

        print(f"Consulta {i}: Tiempo de respuesta = {response_time:.4f} segundos")

    # Calcula la media de tiempo de respuesta
    mean_time = total_time / len(response_times) if response_times else 0
    print(f"Tiempo medio de respuesta: {mean_time:.4f} segundos")

""" RESULTS: Tiempo medio de respuesta:  segundos"""

if __name__ == "__main__":

    # Evaluar Salida Final (Precision, Recall, F1 de Verdadero / Falso / Other)
    evaluate_verification_pipeline()

    # Evaluar Retriever (Recall@k y MRR al hace retrieval de evidencias)
    evaluate_retriever(k=5)

    # Evaluar BERT Score (Similaridad respuesta del LLM y evidencias)
    evaluate_bert_score()

    # Evaluar Fact Score
    evaluate_factscore()

    # Evaluar Tiempo de Respuesta (Media de 30 consultas)
    evaluate_time_response()
