import pandas as pd

df = pd.read_json('obfuscated_data.jsonl', orient='records')

for row in df.sample(100).itertuples():
    for i, label in enumerate(row.labels):
        if label != 'O':
            print(label, row.tokens[i])