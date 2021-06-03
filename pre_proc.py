import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.stats as sp
# from pdpbox.pdp import pdp_isolate, pdp_plot
# import sns as sns
from sklearn.model_selection import train_test_split
import calendar

week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]




# df = pd.read_csv('../Chicago-Police-ML/Dataset_crimes.csv', sep=",")
# Visualization of the Longitude and Latitude.
# plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)
# plt.show()

TARGET = 'Primary Type'
labels = {'BATTERY': 0, 'THEFT': 1,  'CRIMINAL DAMAGE': 2,
          'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


# def split_data(data):
#     # split to train and test data
#     train, validate, test = np.split(data.sample(frac=1), [int(.7 * len(df)), int(.5 * len(df))])
#     train.to_csv("train.csv")
#     validate.to_csv("validate.csv")
#     test.to_csv("test.csv")
#     return train, validate, test


def split_x_y(data):
    y = data[TARGET]
    X = data.drop(TARGET, axis=1)

    return X, y


def clean_data(data):  # receives X m*d
    # delete null values
    data = data.dropna()

    # delete irrelevant fetcher
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('Unnamed: 0.1', axis=1)
    data = data.drop('ID', axis=1)
    data = data.drop('Year', axis=1)
    data = data.drop('Location', axis=1)
    data = data.drop('Case Number', axis=1)
    data = data.drop('Location Description', axis=1)
    data = data.drop('Block', axis=1)
    ata = data.drop('Updated On', axis=1)

    data = data.drop('Description', axis=1)
    data = data.drop('IUCR', axis=1)
    data = data.drop('FBI Code', axis=1)

    # changes boll to int
    data['Arrest'] = data['Arrest'].astype(int)
    data['Domestic'] = data['Domestic'].astype(int)

    # Splitting the Date to Day, Month, Year, Hour, Minute, Second
    data['date2'] = pd.to_datetime(data['Date'])
    data['Year'] = data['date2'].dt.year
    data['Month'] = data['date2'].dt.month
    data['Day'] = data['date2'].dt.day
    data['Hour'] = data['date2'].dt.hour
    data['Minute'] = data['date2'].dt.minute
    # data['Second'] = data['date2'].dt.second
    # data = data.drop(['Date'], axis=1)
    # data = data.drop(['date2'], axis=1)
    data = data.drop(['Updated On'], axis=1)
    data['day_of_week'] = data['date2'].dt.day_name()
    data = pd.get_dummies(data,columns=['day_of_week'])

    data = data.drop_duplicates()
    # changes string labels to ints
    c = pd.Categorical(data[TARGET])
    data[TARGET] = c.rename_categories(labels)

    # normalize
    lst = ['Beat', 'District', 'Ward', 'Community Area']
    for i in lst:
        data[i] = data[i] / data[i].abs().max()

    # data['day-of-week'] = data[calendar.weekday(data['Year'],data['Month'],data['Day'])]
    # weekday = calendar.weekday(2020, 7, 24)
    # print(week_days[weekday])

    #delete outliers TODO
    # z_score = sp.zscore(data)
    # abs_z_score = np.abs(z_score)
    # filter_entire = (abs_z_score < 3).all(axis=1)
    # data = data[filter_entire]

    return data


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    print(data)
    # train, validate, test = split_data(full_data)
    print(data['Beat'])
    data = clean_data(data)
    print(data['date2'])






