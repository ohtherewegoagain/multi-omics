import pandas as pd

def load_multiomics_data(file_path='/mnt/data/synthetic_multiomics_data.csv'):
    """Loads the multi-omics dataset from a CSV file."""
    return pd.read_csv(file_path)
