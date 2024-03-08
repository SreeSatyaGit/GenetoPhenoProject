import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def cm(y_true, predict):
    cell_types = {'CD+4 T':0,'CD8+ T':1,'B':2,'MO':3,'DC':4,'NE':5,'EO':6,'BA':7}
    labels = list(cell_types.keys())
    cm = confusion_matrix(y_true, predict)
    confusion_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create heatmap
    ax = plt.subplot()
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    
    # Set labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    return ax


def accuracy(y_true,predict):
    cm = confusion_matrix(y_true, predict)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    for i in range(len(class_accuracy)):
        class_accuracy[i] = class_accuracy[i] *100
    cell_types = ['CD+4 T','CD8+ T','B','MO','DC','NE','EO','BA']
    plt.bar(cell_types, class_accuracy)
    plt.xlabel('Cell Types')
    plt.ylabel('Accuracy in %')
    plt.title('Accuracy for Each Cell Type')
    plt.ylim([0, 100]) 
    plt.show()
    return plt
