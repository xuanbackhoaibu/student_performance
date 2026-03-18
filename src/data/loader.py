import pandas as pd

def load_dataset(path):
    """
    Load dataset from csv file
    """
    df = pd.read_csv(path, sep=";")
    return df