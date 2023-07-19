import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def save_confusionmatrix(y_true, y_pred, path_save, phase='validation'):
    path_save = os.path.join(path_save, 'confusion_matrix_{}.png'.format(phase))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(path_save)
    plt.close()
