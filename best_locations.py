import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt
import pre_proc as pp
import numpy as np
import matplotlib.pyplot as plt

labels = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
          'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def split_x_y(data):
    y = data['Date']
    # data = data.drop('Date', axis=1)
    # data = data.drop('Month', axis=1)
    # data = data.drop('Day', axis=1)
    # data = data.drop('Hour', axis=1)
    # data = data.drop('Minute', axis=1)

    return data, y


def clean_data(data):  # receives X m*d
    # delete null values
    data = data.dropna()

    # delete irrelevant fetcher
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('Unnamed: 0.1', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Year', axis=1)
    data['Location'] = pd.factorize(data["Location"])[0]
    data = data.drop('Case Number', axis=1)
    data['Location Description'] = pd.factorize(data["Location Description"])[0]
    data['Block'] = pd.factorize(data["Block"])[0]
    data = data.drop('Updated On', axis=1)

    data['FBI Code'] = pd.factorize(data['FBI Code'])[0]
    data['Description'] = pd.factorize(data['Description'])[0]
    data['IUCR'] = pd.factorize(data['IUCR'])[0]

    # changes boll to int
    data['Arrest'] = data['Arrest'].astype(int)
    data['Domestic'] = data['Domestic'].astype(int)

    # Splitting the Date to Day, Month, Year, Hour, Minute, Second
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna()
    #
    # data['date2'] = pd.to_datetime(data['Date'])
    # print(data['date2'])
    # data['Year'] = data['date2'].dt.year
    # data['Month'] = data['date2'].dt.month
    # data['Day'] = data['date2'].dt.day
    # data['Hour'] = data['date2'].dt.hour
    # data['Minute'] = data['date2'].dt.minute
    # # vec = []
    # for i in range(data.shape[0]):
    #     vec.append(dt.datetime(data['Year'][i], data['Month'][i], data['Month'][i], data['Hour'][i], data['Minute'][i]))
    # data['Time'] = vec
    # data = data.drop(['Date'], axis=1)  # TODO return this
    data['Date'] = pd.to_datetime(data['Date'])

    # data = data.drop(['date2'], axis=1)

    c = pd.Categorical(data['Primary Type'])
    data['Primary Type'] = c.rename_categories(labels)

    # normalize
    lst = ['Beat', 'District', 'Ward', 'Community Area']
    for i in lst:
        data[i] = data[i] / data[i].abs().max()
    return data


def get_valid_points_per_date(date, y, data):
    # for i in range(data.shape[0]):
    #     data['Date'][i] = dt.datetime(date)
    # data['Date'] = pd.to_datetime(data['Date'])
    time_change = dt.timedelta(minutes=30)
    upper = date + time_change
    data['Date'] = data[data['Date'] <= upper]
    # data = data.drop(data[data['Date'] <= upper])
    lower = date - time_change
    # data = data.drop(data[data['Date'] >= lower])
    data['Date'] = data[data['Date'] >= lower]
    data = data.dropna()

    print("returning new filtered data")
    return data



if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    X, y = split_x_y(clean_data(data))
    X = X.reindex(range(X.shape[0]))
    y = y.reindex(range(X.shape[0]))

    points = get_valid_points_per_date(dt.datetime(2021, 1, 7, 11, 30, 0), y, X)
    print(points)
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    # dates = matplotlib.dates.date2num(y)
    ax.scatter3D(X['X Coordinate'], X['Y Coordinate'], y, color="green")
    plt.title("simple 3D scatter plot")

    # show plot
    plt.show()

    #
    # print(X_train)
    # print()
    # print(y_train)
    # kmeans = KMeans(
    #     init = "random",
    #     n_clusters = 30,
    #     n_init = 10,
    #     max_iter = 300,
    #     random_state = 42)
    # kmeans.fit(X_train, y_train)
    # print(kmeans.score(X_train, y_train))
