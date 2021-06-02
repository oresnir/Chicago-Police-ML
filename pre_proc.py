import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pdpbox.pdp import pdp_isolate, pdp_plot
import sns as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Chicago-Police-ML/Dataset_crimes.csv', sep=",")

labels = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
          3: 'DECEPTIVE RACTICE', 4: 'ASSAULT'}


def split_data(data):
    # Split Data Train/ Validate/Test
    target = 'Arrest'  # our labels
    features = data.columns.drop([target, 'ID'])

    X = data[features]
    y = data[target]

    # split to train and test data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.20, random_state=42)

    # create validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


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

    # print(data)

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

    plt.show()

    return data


if __name__ == '__main__':
    full_data = pd.read_csv("dataset_crimes.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(full_data)
    msk = np.random.rand(len(full_data)) < 0.75
    train = full_data[msk]
    test = full_data[~msk]
    print(train)
    clean_data(train)


