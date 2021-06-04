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

days_dict = {}
month_dict = {}
time_dict = {}


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

    # data['FBI Code'] = pd.factorize(data['FBI Code'])[0]
    # data['Description'] = pd.factorize(data['Description'])[0]
    # data['IUCR'] = pd.factorize(data['IUCR'])[0]
    data = data.drop('Description', axis=1)
    data = data.drop('IUCR', axis=1)
    data = data.drop('FBI Code', axis=1)

    # changes boll to int
    data['Arrest'] = data['Arrest'].astype(int)
    data['Domestic'] = data['Domestic'].astype(int)

    # data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    # data = data.dropna()
    data['date2'] = pd.to_datetime(data['Date'])
    # print(data['date2'])
    data['Hour'] = data['date2'].dt.hour.astype(int)
    # data['Month'] = data['date2'].dt.month
    data['Minute'] = data['date2'].dt.minute.astype(int)
    # print(data['Hour'], data['Minute'])

    data['day_of_week'] = data['date2'].dt.day_name()
    c = pd.Categorical(data['day_of_week'])
    data['day_of_week'] = c.rename_categories(week_days)

    # data['Date'] = dt.datetime.strptime(data['Date'][0], "%H:%M")
    # print("this is 0: ", data['Date'][0])
    data = data.drop('date2', axis=1)

    data['Date'] = pd.to_datetime(data['Date'])

    c = pd.Categorical(data['Primary Type'])
    data['Primary Type'] = c.rename_categories(labels)

    # normalize
    lst = ['Beat', 'District', 'Ward', 'Community Area']
    for i in lst:
        data[i] = data[i] / data[i].abs().max()

    # for i in range(100):
    #     print(data['Date'][i])
    return data


def convert_cat_to_time(cat):
    reverse_time_dict = {v: k for k, v in time_dict.items()}
    cat /= 3
    cat = cat.astype(int)
    temp = 0
    counter = 0
    while temp < cat and counter < len(reverse_time_dict) - 1:
        temp += 1
    return reverse_time_dict[temp]


def fill_time_dict(cur, parts_num):
    start = dt.datetime(cur.year, cur.month, cur.day, 0, 0)
    time_change = dt.timedelta(minutes=30)
    for i in range(parts_num):
        time_dict[start] = i
        start = start + time_change
    return time_dict


def get_time_cat(time, cur, parts_num):
    time_dict = fill_time_dict(cur, parts_num)
    counter = 0
    new_time = dt.datetime(cur.year, cur.month, cur.day, time.hour, time.minute)
    temp = dt.datetime(cur.year, cur.month, cur.day, 0, 0)
    time_change = dt.timedelta(minutes=30)
    while new_time >= temp and counter < len(time_dict) - 1:
        temp = temp + time_change
        counter += 1
    return time_dict[temp]


def add_time_col(data, cur, parts_num):
    lst = [get_time_cat(data['Date'][i], cur, parts_num) for i, crime in X.iterrows()]
    data['Time Cat'] = lst


# def get_time_category(date, data):
#     start = dt.time(0, 0)
#     time_change = dt.timedelta(minutes=30)
#     upper = start + time_change
#     data['Date'] = data[data['Date'] <= upper]
#     lower = start
#     data['Date'] = data[data['Date'] >= lower]
#     data = data.dropna()


# def get_valid_points_per_date(date, data):
#     time_change = dt.timedelta(minutes=30)
#     upper = date + time_change
#     data['Date'] = data[data['Date'] <= upper]
#     lower = date - time_change
#     data['Date'] = data[data['Date'] >= lower]
#     data = data.dropna()
#
#     print("returning new filtered data")
#     return data


def get_dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


class Cluster:

    def __init__(self, df, type, n_clusters):
        self.h = None
        self.df = df
        self.centers = None
        self.type = type
        self.n_clusters = n_clusters

    def create_h(self):
        k_means = KMeans(
            init="random",
            n_clusters=self.n_clusters,
            n_init=10,
            max_iter=100,
            random_state=42)

        k_means.fit(self.df)
        self.h = k_means
        self.centers = k_means.cluster_centers_
        # print("this is", type, self.centers)

    def plot_centers(self):
        plt.figure(figsize=(20, 13))
        ax = plt.axes(projection="3d")
        ax.scatter(self.centers[:, 0], self.centers[:, 1], self.centers[:, 2],
                   cmap='rainbow')
        plt.title(f"location vs {self.type}")
        plt.xlim((min(self.centers[:, 0]), max(self.centers[:, 0])))
        plt.ylim((min(self.centers[:, 1]), max(self.centers[:, 1])))
        plt.show()


# day = string(Sunday-Monday) , month = int (1-12)
def master_clusters(time):
    day = time.weekday()
    hour_centers = pd.DataFrame(hour_cluster.centers)
    day_centers = pd.DataFrame(days_dict[day].centers)
    frames = [hour_centers, day_centers]
    result = pd.concat(frames)
    # print(result)
    final_cluster = Cluster(result, 'final', 30)
    final_cluster.create_h()
    # final_cluster.plot_centers()
    # print("this is the cate:", final_cluster.centers[:, 2])
    temp = []
    for i in range(final_cluster.centers.shape[0]):
        temp.append(convert_cat_to_time(final_cluster.centers[i, 2]))
    # a = np.delete(final_cluster.centers, final_cluster.centers[:, 2])
    final = np.array((30, 3))
    a = final_cluster.centers[:, 0]
    b = final_cluster.centers[:, 1]
    l = np.hstack((a,b))
    ll = np.hstack((l, temp))
    # print(ll)
    # print("this is the centers:", final_cluster.centers[:, 2].astype(int).unique().size)
    return ll


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    data = clean_data(data)
    X, y = split_x_y(data)
    X = X.reindex(range(X.shape[0]))
    y = y.reindex(range(X.shape[0]))
    X = X.dropna()

    # points = get_valid_points_per_date(dt.datetime(2021, 1, 7, 11, 30, 0), X)
    # X = X.drop('Date', axis=1)
    # print(X.dtypes)
    # data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    # data = data.dropna(subset=['Date'])
    # X = X.reindex(range(X.shape[0]))
    # print(X)
    # X = X.drop(X[70], axis=0)
    # y = y.reindex(range(X.shape[0]))
    # X['Hour'] = X['Hour'].astype(int)
    # X['Minute'] = X['Minute'].astype(int)
    ts = []

    # print("X size:", X.shape[0])
    for index, crime in X.iterrows():
        # data.get('your_column', default=value_if_no_column)
        h = X['Date'][index].hour
        m = X['Date'][index].minute
        t = dt.time(h, m)
        ts.append(t)
    X['Time'] = ts
    add_time_col(X, dt.datetime(2021, 3, 7, 11, 30, 0), 48)

    # print(X['Time Cat'])
    train_hour = X[['X Coordinate', 'Y Coordinate', 'Time Cat']]
    train_hour['Time Cat'] *= 3 #* train_hour['Time Cat']
    # print(train_hour)
    # train_hour['Time'] = ts
    # train_hour = train_hour.reindex(range(train_hour.shape[0]))
    hour_cluster = Cluster(train_hour, 'hour', 30)
    hour_cluster.create_h()
    # print("this is hour:", hour_cluster.centers)
    hour_cluster.plot_centers()

    # X['Hour'] = X['Hour'].astype(int)
    # X['Minute'] = X['Minute'].astype(int)
    # for i in range(X.shape[0]):
    #     h = X['Hour'][i]
    #     m = X['Minute'][i]
    #     # t.append(dt.time(h, m))
    #     print(h,m)

    # train_hour = X[['X Coordinate', 'Y Coordinate', 'Hour']]
    # hour_cluster = Cluster(train_hour, 'hour')
    # hour_cluster.create_h()
    # hour_cluster.plot_centers()

    for key, value in week_days.items():
        train_day_of_week = X[['X Coordinate', 'Y Coordinate', 'day_of_week', 'Time Cat']]
        train_day_of_week = train_day_of_week[train_day_of_week['day_of_week'] == value]
        train_day_of_week = train_day_of_week.drop('day_of_week', axis=1)
        # print("the day: ", key, train_day_of_week)
        week_cluster = Cluster(train_day_of_week, key, 30)
        days_dict[value] = week_cluster
        week_cluster.create_h()

    # for month in range(1, 13):
    #     train_month = X[['X Coordinate', 'Y Coordinate', 'Month']]
    #     train_month = train_month[train_month['Month'] == month]
    #     print("the month: ", month, train_month)
    #     month_cluster = Cluster(train_month, month)
    #     month_dict[month] = month_cluster
    #     month_cluster.create_h()
    #     month_cluster.plot_centers()

    # master_clusters(dt.datetime(2022, 1, 7, 11, 30, 0))
