import pandas as pd
import json

df = pd.read_csv("sahih_bukhari_hadiths (1)(1).csv")  # Your CSV file
jsonl_file = "hadith_grammar.jsonl"

with open(jsonl_file, "w") as f:
    for sentence in df['Hadith English']:
        item = {
            "instruction": "Correct the grammar of the following sentence",
            "input": sentence,
            "output": sentence  # For now, you can keep same sentence if no corrections; better if you manually correct some examples
        }
        f.write(json.dumps(item) + "\n")
