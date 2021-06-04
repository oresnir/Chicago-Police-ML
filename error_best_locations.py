import datetime as dt
import math
import pandas as pd

from best_locations import split_x_y, clean_data, Cluster


# crime: one true label- (x,y,time)
# y_hat: list with 30 tuples in the form- (x,y,time)
def exist_police(crime, y_hat):
    print(crime)
    for i in y_hat:
        crime_loc = (crime['X Coordinate'], crime['Y Coordinate'])
        y_loc = (i[0], i[1])
        if math.dist(crime_loc, y_loc) <= 500:
            if abs(crime['Date'] - i[2]) <= 30:
                return True
    return False


# date: date in the form-
# y_hat: list of our predictions- with 30 tuples in the form- (x,y,time)
def measure_best_locations(data, date, y_hat):
    # time_change = data.timedelta(days=3)
    # upper = date + time_change
    # lower = date - time_change
    # date_data = data.loc[data['Date'] <= upper]
    # date_data = date_data.loc[data['Date'] >= lower]
    prevented = 0
    for index, crime in data.iterrows():
        if exist_police(crime, y_hat):
            prevented += 1
    return prevented


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    X, y = split_x_y(clean_data(data))
    X = X.reindex(range(X.shape[0]))
    y = y.reindex(range(X.shape[0]))

    date1 = dt.datetime(2021, 1, 7, 11, 30, 0)

    X = X.drop('Date', axis=1)
    X = X.dropna()

    train_hour = X[['X Coordinate', 'Y Coordinate', 'Hour']]
    hour_cluster = Cluster(train_hour, 'hour')
    hour_cluster.create_h()
    y_hat = hour_cluster.centers
    measure_best_locations(data, date1, y_hat)
