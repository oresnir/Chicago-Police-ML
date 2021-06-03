import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt
import pre_proc as pp
import numpy as np
import matplotlib.pyplot as plt
import math

labels = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
          'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def split_x_y(data):
    y = data['Date']

    return data, y


def clean_data(data):  # receives X m*d
    # delete null values
    data = data.dropna()

    # delete irrelevant fetcher
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('Unnamed: 0.1', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Year', axis=1)
    data = data.drop('Case Number', axis=1)
    data = data.drop('Location', axis=1)
    data['Location Description'] = pd.factorize(data["Location Description"])[0]
    data['Block'] = pd.factorize(data["Block"])[0]
    data = data.drop('Updated On', axis=1)

    data['FBI Code'] = pd.factorize(data['FBI Code'])[0]
    data['Description'] = pd.factorize(data['Description'])[0]
    data['IUCR'] = pd.factorize(data['IUCR'])[0]

    # changes boll to int
    data['Arrest'] = data['Arrest'].astype(int)
    data['Domestic'] = data['Domestic'].astype(int)

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna()

    data['Date'] = pd.to_datetime(data['Date'])

    c = pd.Categorical(data['Primary Type'])
    data['Primary Type'] = c.rename_categories(labels)

    # normalize
    lst = ['Beat', 'District', 'Ward', 'Community Area']
    for i in lst:
        data[i] = data[i] / data[i].abs().max()
    return data


def get_valid_points_per_date(date, data):
    time_change = dt.timedelta(minutes=30)
    upper = date + time_change
    data['Date'] = data[data['Date'] <= upper]
    lower = date - time_change
    data['Date'] = data[data['Date'] >= lower]
    data = data.dropna()

    print("returning new filtered data")
    return data


def get_dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    X, y = split_x_y(clean_data(data))
    X = X.reindex(range(X.shape[0]))
    y = y.reindex(range(X.shape[0]))

    points = get_valid_points_per_date(dt.datetime(2021, 1, 7, 11, 30, 0), X)
    print(points)
    # Creating figure
    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter(X['X Coordinate'], X['Y Coordinate'], c=y, cmap='rainbow')
    plt.gray()
    # plt.title("simple 3D scatter plot")

    # show plot
    # plt.figure()
    # plt.scatter3d([get_dist(X['X Coordinate'][i], X['Y Coordinate'][i]) for i in range(X.shape[0])], range(X.shape[0]))
    plt.show()

    #
    print(X)
    print()
    print(y)
    X = X.drop('Date', axis=1)
    X = X.dropna()
    kmeans = KMeans(
        init="random",
        n_clusters=30,
        n_init=10,
        max_iter=300,
        random_state=42)
    kmeans.fit(X)
    print(kmeans.get_params())
    print(kmeans.score(X, y))
