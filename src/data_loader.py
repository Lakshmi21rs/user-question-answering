import json
import pandas as pd

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)
