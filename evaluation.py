
import tqdm
import pandas as pd
from sklearn.metrics import classification_report

from fact_checker.query_pipeline import verification_pipeline

df = pd.read_csv("evaluation-dataset.csv")
print(df['claim'][0])
df['label']    = df['label'].str.lower().str.strip()
df['response'] = ""

for i, row in tqdm.tqdm(df.iterrows(), desc = "Proceso Evaluación: ", total = df.shape[0]):
    response = verification_pipeline(row["claim"])
    df.loc[i, "response"] = response['result']
    # print(response['result'] + ', ' + row['label'])

print(classification_report(df['response'], df['label']))
df.to_csv("evaluated-dataset.csv", index=False)

# "no evidence"       0.00      0.00      0.00         0
#            no evidence       0.00      0.00      0.00        14
#    not enough evidence       0.00      0.00      0.00        15
# not enough information       0.00      0.00      0.00         0 -> No existe
#                 refute       0.40      1.00      0.57         4
#                support       0.20      1.00      0.33         2

#               accuracy                           0.17        35
#              macro avg       0.10      0.33      0.15        35
#           weighted avg       0.06      0.17      0.08        35

"""       "no evidence"       0.00      0.00      0.00         0
              false       0.00      0.00      0.00         4
       inconclusive       0.00      0.00      0.00         5
        no evidence       0.00      0.00      0.00        13
not enough evidence       0.00      0.00      0.00         0
             refute       0.50      1.00      0.67         5
            support       0.80      1.00      0.89         8

           accuracy                           0.37        35
          macro avg       0.19      0.29      0.22        35
       weighted avg       0.25      0.37      0.30        35 """