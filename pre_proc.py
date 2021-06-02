import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.stats as sp
# from pdpbox.pdp import pdp_isolate, pdp_plot
import sns as sns
from sklearn.model_selection import train_test_split

target = 'Primary Type'

df = pd.read_csv('../Chicago-Police-ML/Dataset_crimes.csv', sep=",")

# Visualization of the Longitude and Latitude.
# plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)
#
# plt.show()

labels = {'BATTERY': 0, 'THEFT': 1,  'CRIMINAL DAMAGE': 2,
          'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def split_data(data):
    # split to train and test data
    train, validate, test = np.split(data.sample(frac=1), [int(.7 * len(df)), int(.5 * len(df))])
    return train, validate, test


def split_x_y(data):
    y = data['Primary Type']
    X = data.drop('Primary Type', axis=1)

    return X, y


def clean_data(data):  # receives X m*d
    # delete null values
    data = data.dropna()

    # delete irrelevant fetcher
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Year', axis=1)
    data = data.drop('Location', axis=1)
    data = data.drop('Case Number', axis=1)
    data = data.drop('Location Description', axis=1)
    data = data.drop('Block', axis=1)

    data = data.drop('Description', axis=1)
    data = data.drop('IUCR', axis=1)
    data = data.drop('FBI Code', axis=1)

    # changes boll to int
    data['Arrest'] = data['Arrest'].astype(int)
    data['Domestic'] = data['Domestic'].astype(int)

    # Splitting the Date to Day, Month, Year, Hour, Minute, Second
    data['date2'] = pd.to_datetime(df['Date'])
    data['Year'] = data['date2'].dt.year
    data['Month'] = data['date2'].dt.month
    data['Day'] = data['date2'].dt.day
    data['Hour'] = data['date2'].dt.hour
    data['Minute'] = data['date2'].dt.minute
    # data['Second'] = data['date2'].dt.second
    data = data.drop(['Date'], axis=1)
    data = data.drop(['date2'], axis=1)
    data = data.drop(['Updated On'], axis=1)

    # data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M:%S %p', errors='coerce')
    data = data.drop_duplicates()

    # data['Primary Type'] = LabelEncoder().fit_transform(data['Primary Type'])
    c = pd.Categorical(data['Primary Type'])
    data['Primary Type'] = c.rename_categories(labels)

    #delete outliers TODO
    # z_score = sp.zscore(data)
    # abs_z_score = np.abs(z_score)
    # filter_entire = (abs_z_score < 3).all(axis=1)
    # data = data[filter_entire]

    # print(data)
    return data


if __name__ == '__main__':
    full_data = pd.read_csv("dataset_crimes.csv")
    train, validate, test = split_data(full_data)
    print(train[target])
    train = clean_data(train)
    print(train[target])





