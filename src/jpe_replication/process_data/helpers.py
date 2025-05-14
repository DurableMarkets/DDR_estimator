import pandas as pd
import os

def read_file(fpath: str):
    assert os.path.isfile(fpath), f"File {os.path.basename(fpath)} does not exist"
    counts = pd.read_excel(fpath)
    return counts

