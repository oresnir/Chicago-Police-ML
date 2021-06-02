import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pdpbox.pdp import pdp_isolate, pdp_plot
import gc
import psutil
import os
from sklearn.metrics import mean_absolute_error

labels = {0:'BATTERY', 1:'THEFT', 2:'CRIMINAL DAMAGE',
          3:'DECEPTIVE RACTICE', 4:'ASSAULT'}

def clean_data(data):  # receives X m*d
    data = data.dropna()
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Location', axis=1)

    # to check if needs to be removed - not given in instructions
    data = data.drop('Case Number', axis=1)
    data = data.drop('Description', axis=1)
    data = data.drop('Block', axis=1)

    # data = data.drop('IUCR', axis=1)
    # data = data.drop('FBI Code', axis=1)

    print(data)


if __name__ == '__main__':
    full_data = pd.read_csv("dataset_crimes.csv")
    msk = np.random.rand(len(full_data)) < 0.75

    train = full_data[msk]
    test = full_data[~msk]
    print(train)
    clean_data(train)

    # sns.set(style='ticks', context='talk')
    # sns.swarmplot(x='Domestic', y='Primary Type', data=test)
    # sns.despine()
    # print(train.describe())
    # plt.matshow(test)
    # plt.show()
