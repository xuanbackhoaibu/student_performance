from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters=3):
    """
    Perform KMeans clustering
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(X)

    return model, labels