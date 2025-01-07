
import tqdm
import pandas as pd
from sklearn.metrics import classification_report

from fact_checker.query_pipeline import verification_pipeline

EVALUATION_VERSION: int = 0

df = pd.read_csv(f"evaluation-dataset-v{EVALUATION_VERSION}.csv")
print(df['claim'][0])
df['label']    = df['label'].str.strip()
df['response'] = ""

for i, row in tqdm.tqdm(df.iterrows(), desc = "Proceso Evaluación: ", total = df.shape[0]):
    correct = "0"
    response = verification_pipeline(row["claim"])

    if response['result'] == "no evidence":
        response['result'] = "other"
    elif response['result'] == "not enough evidence":
        response['result'] = "other"

    df.loc[i, "response"] = response['result']
    if response['result'] == row['label']:
        correct = "1"
    print(response['result'] + ', ' + row['label'] + ', ' + correct)

print(classification_report(df['response'], df['label']))
df.to_csv(f"evaluated-dataset-v{EVALUATION_VERSION}.csv", index=False)

""" Prueba 6 - Dataset v0
              precision    recall  f1-score   support

       other       0.96      0.90      0.93        30
      refute       0.80      0.89      0.84         9
     support       0.91      1.00      0.95        10

    accuracy                           0.92        49
   macro avg       0.89      0.93      0.91        49
weighted avg       0.92      0.92      0.92        49
"""

""" Prueba 5 - Dataset v0
                     precision    recall  f1-score   support

            "other"       0.00      0.00      0.00         0
        no evidence       0.00      0.00      0.00        19
not enough evidence       0.00      0.00      0.00        10
              other       0.00      0.00      0.00         0
             refute       0.80      0.80      0.80        10
            support       0.91      1.00      0.95        10

           accuracy                           0.37        49
          macro avg       0.28      0.30      0.29        49
       weighted avg       0.35      0.37      0.36        49
"""

""" Prueba 4 - Dataset v0
                       precision    recall  f1-score   support

"not enough evidence"       0.00      0.00      0.00         0
          no evidence       1.00      0.57      0.73        21
  not enough evidence       0.53      0.80      0.64        10
               refute       0.67      0.80      0.73        10
              support       0.91      1.00      0.95        10

             accuracy                           0.75        51
            macro avg       0.62      0.63      0.61        51
         weighted avg       0.83      0.75      0.75        51
"""

""" Prueba 3 - Dataset v0
                     precision    recall  f1-score   support

        no evidence       1.00      0.77      0.87        13
not enough evidence       0.60      0.75      0.67         4
             refute       0.70      0.88      0.78         8
            support       0.90      0.90      0.90        10

           accuracy                           0.83        35
          macro avg       0.80      0.82      0.80        35
       weighted avg       0.86      0.83      0.83        35
"""

""" Prueba 2 - Dataset v1
                     precision    recall  f1-score   support

        no evidence       1.00      1.00      1.00        10
not enough evidence       0.40      0.40      0.40         5
             refute       0.90      0.60      0.72        15
            support       0.40      0.80      0.53         5

           accuracy                           0.71        35
          macro avg       0.67      0.70      0.66        35
       weighted avg       0.79      0.71      0.73        35
"""

""" Prueba 1 - Dataset v1
                     precision    recall  f1-score   support

        no evidence       1.00      0.83      0.91        12
not enough evidence       0.80      0.29      0.42        14
             refute       0.60      0.86      0.71         7
            support       0.20      1.00      0.33         2

           accuracy                           0.63        35
          macro avg       0.65      0.74      0.59        35
       weighted avg       0.79      0.63      0.64        35
"""
