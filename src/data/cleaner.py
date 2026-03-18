import pandas as pd

def check_missing(df):
    """
    Check missing values
    """
    return df.isnull().sum()

def remove_duplicates(df):
    """
    Remove duplicate rows
    """
    return df.drop_duplicates()

def create_pass_label(df):
    """
    Create pass / fail label
    """
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df