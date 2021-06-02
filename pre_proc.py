import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import seaborn as sns
# from pdpbox.pdp import pdp_isolate, pdp_plot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

df= pd.read_csv('../data_set/Chicago_Crimes_2015_to_2017.csv', sep=",")

#Drop nan values
df= df.dropna()


#Drop column Unnamed
df=df.drop('Unnamed: 0',axis=1)
df.head()

#Convert Date column to Date Time format
df['Date']= pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S %p' , errors='coerce')

#Eliminate duplicate rows
df=df.drop_duplicates()

#Visualization of the Longitude and Latitude.
plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)

plt.show();

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

    df = pd.read_csv('../data_set/Chicago_Crimes_2015_to_2017.csv', sep=",")

    # Drop nan values
    df = df.dropna()

    # Drop column Unnamed
    df = df.drop('Unnamed: 0', axis=1)
    df.head()

    # Convert Date column to Date Time format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S %p', errors='coerce')

    # Eliminate duplicate rows
    df = df.drop_duplicates()

    # Visualization of the Longitude and Latitude.
    plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)

    plt.show();


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
