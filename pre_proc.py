import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
# from pdpbox.pdp import pdp_isolate, pdp_plot
import sns as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Chicago-Police-ML/Dataset_crimes.csv', sep=",")

# Visualization of the Longitude and Latitude.
# plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)
#
# plt.show()

labels = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
          3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def split_data(data):
    # split to train and test data
    train, validate, test = np.split(data.sample(frac=1), [int(.7 * len(df)), int(.5 * len(df))])
    return train, validate, test


def split_x_y(data):
    y = data['Primary Type']
    X = data.drop('Primary Type', axis=1)

    return X, y


def clean_data(data):  # receives X m*d
    # delete nall values
    data = data.dropna()

    #delete outliers
    z_score = sp.zscore(data)
    abs_z_score = np.abs(z_score)
    filter_entire = (abs_z_score < 3).all(axis=1)
    data = data[filter_entire]

    # delete irrelevant fetcher
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Location', axis=1)
    data = data.drop('Case Number', axis=1)
    data = data.drop('Description', axis=1)
    data = data.drop('Block', axis=1)
    data = data.drop('IUCR', axis=1)
    data = data.drop('FBI Code', axis=1)
    # data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M:%S %p', errors='coerce')
    data = data.drop_duplicates()

    # print(data)
    return data


if __name__ == '__main__':
    full_data = pd.read_csv("dataset_crimes.csv")
    train, validate, test = split_data(full_data)
    print(train)
    train = clean_data(train)
    print(train)



