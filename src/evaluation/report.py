import pandas as pd

def save_results(results, path):

    df = pd.DataFrame([results])

    df.to_csv(path, index=False)

    print("Report saved:", path)