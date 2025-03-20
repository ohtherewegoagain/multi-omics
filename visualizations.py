import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def plot_heatmap(data):
    sns.heatmap(data, cmap='coolwarm', annot=True)
    plt.show()
