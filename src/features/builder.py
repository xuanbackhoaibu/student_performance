from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def encode_categorical(df):
    """
    Encode categorical features
    """
    categorical_cols = df.select_dtypes(include=['object']).columns

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def scale_features(df):
    """
    Scale numeric features
    """
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df