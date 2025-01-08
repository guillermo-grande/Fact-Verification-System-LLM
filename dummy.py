from openai import OpenAI
import pandas as pd
import json 

claims = pd.read_parquet('./data/climate-claims.parquet.gzip')
column_map = { 'text': 'claim', 'id': 'claim_id'}
claims = claims[['text', 'claim_label', 'id']].rename(columns = column_map)

client = OpenAI()
batch_input_file = client.files.create(
    file=open("batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

