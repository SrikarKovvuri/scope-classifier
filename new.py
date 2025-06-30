# csv2json.py
import pandas as pd, json

# 1) Read your exported CSV
df = pd.read_csv("sheet.csv")

# 2) Rename to match classifier.py’s columns
df = df.rename(columns={"invalid_human_answer": "invalid_HA"})

# 3) Only keep the fields classifier.py needs:
df = df[["question","out-of-scope","human_answer","invalid_HA"]]

# 4) Convert to { index: record } shape
data = df.to_dict(orient="index")

# 5) Dump JSON
with open("900_data.json","w") as f:
    json.dump(data, f, indent=2)

print(f"→ 900_data.json ready with {len(data)} entries")
