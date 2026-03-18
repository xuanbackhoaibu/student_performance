from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd


def run_apriori(df, min_support=0.1):

    """
    Generate frequent itemsets using Apriori
    """

    itemsets = apriori(df, min_support=min_support, use_colnames=True)

    return itemsets


def generate_rules(itemsets, min_confidence=0.5):

    """
    Generate association rules
    """

    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    return rules