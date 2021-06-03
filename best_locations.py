import matplotlib
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt
import pre_proc as pp
import numpy as np
import matplotlib.pyplot as plt
import math

week_days = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
             "Friday": 4, "Saturday": 5, "Sunday": 6}
labels = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
          'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}

cluster_dict = {}


def split_x_y(data):
    y = data['Hour']

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
    data['date2'] = pd.to_datetime(data['Date'])
    data['Hour'] = data['date2'].dt.hour
    data['Minute'] = data['date2'].dt.minute

    data['day_of_week'] = data['date2'].dt.day_name()
    c = pd.Categorical(data['day_of_week'])
    data['day_of_week'] = c.rename_categories(week_days)

    data = data.drop('date2', axis=1)

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


class Cluster:

    def __init__(self, df, type):
        self.h = None
        self.df = df
        self.centers = None
        self.type = type

    def create_h(self):
        k_means = KMeans(
            init="random",
            n_clusters=30,
            n_init=10,
            max_iter=300,
            random_state=42)

        k_means.fit(self.df)
        self.h = k_means
        self.centers = k_means.cluster_centers_

    def plot_centers(self):
        plt.figure(figsize=(20, 13))
        ax = plt.axes(projection="3d")
        ax.scatter(self.centers[:, 0], self.centers[:, 1], c=self.centers[:, 2],
                   cmap='rainbow')
        plt.title(f"location vs {self.type}")
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    X, y = split_x_y(clean_data(data))
    X = X.reindex(range(X.shape[0]))
    y = y.reindex(range(X.shape[0]))

    # points = get_valid_points_per_date(dt.datetime(2021, 1, 7, 11, 30, 0), X)

    X = X.drop('Date', axis=1)
    X = X.dropna()
    print(X.dtypes)

    train_hour = X[['X Coordinate', 'Y Coordinate', 'Hour']]
    hour_cluster = Cluster(train_hour, 'hour')
    hour_cluster.create_h()
    hour_cluster.plot_centers()

    train_day_of_week = X[['X Coordinate', 'Y Coordinate', 'day_of_week']]
    for key, value in week_days.items():
        #
        # rslt_df = dataframe[dataframe['Percentage'] > 80]

        train_day_of_week = train_day_of_week[train_day_of_week['day_of_week'] == value]
        print(train_day_of_week)
        week_cluster = Cluster(train_day_of_week, key)
        cluster_dict[key] = week_cluster
        week_cluster.create_h()
        week_cluster.plot_centers()
