from sklearn.semi_supervised import LabelPropagation


def train_label_propagation(X, y):

    """
    Semi supervised learning model
    """

    model = LabelPropagation()

    model.fit(X, y)

    return model