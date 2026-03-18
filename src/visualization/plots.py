import matplotlib.pyplot as plt
import seaborn as sns


def plot_grade_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(df["G3"], bins=20)

    plt.title("Final Grade Distribution")

    plt.show()


def plot_correlation(df):

    plt.figure(figsize=(10,8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, cmap="coolwarm")

    plt.title("Correlation Matrix")

    plt.show()