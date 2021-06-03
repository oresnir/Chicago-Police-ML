import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.stats as sp
from scipy import stats
# from pdpbox.pdp import pdp_isolate, pdp_plot
# import sns as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# df = pd.read_csv('../Chicago-Police-ML/Dataset_crimes.csv', sep=",")
# Visualization of the Longitude and Latitude.
# plt.scatter('Longitude', 'Latitude', c='gray', data=df, s=20)
# plt.show()

TARGET = 'Primary Type'
labels = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
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
    data = data.drop('Updated On', axis=1)

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
    data = data.drop(['Date'], axis=1)
    data = data.drop(['date2'], axis=1)

    # changes string labels to ints
    c = pd.Categorical(data[TARGET])
    data[TARGET] = c.rename_categories(labels)

    # delete duplicates & outliers
    data = data.drop_duplicates()
    # data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

    # normalize
    lst = ['Beat', 'District', 'Ward', 'Community Area']
    for i in lst:
        data[i] = data[i] / data[i].abs().max()
    return data


if __name__ == '__main__':
    data = pd.read_csv("train.csv")
    print(data)
    print(data.dtypes)
    # train, validate, test = split_data(full_data)
    # print(data['Beat'])
    data = clean_data(data)

    # print(data['Beat'])

    X_train, y_train = split_x_y(data)
    print(y_train)
    # print(y_train.value_counts())
    print(y_train.value_counts() / y_train.shape[0])
    print(y_train.shape[0])
    # print("THEFT", y_train(1).count())
    # print("CRIMINAL DAMAGE", y_train(2).count())
    # print("DECEPTIVE PRACTICE", y_train(3).count())
    # print("ASSAULT", y_train(4).count())

    sm = SMOTE()
    resampled_training_inputs, resampled_training_outputs_labels = sm.fit_resample(X_train, y_train)
    print(resampled_training_outputs_labels.value_counts())
    print("X:", resampled_training_outputs_labels.shape)
    print("y:", resampled_training_outputs_labels.shape)

    # labels = {'BATTERY': 0, 'THEFT': 1,  'CRIMINAL DAMAGE': 2,
    # 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
    #
    # svm_classifier = SVC(decision_function_shape='ovr')
    # svm_classifier.fit(resampled_training_inputs, resampled_training_outputs_labels)
    # svm_predictions_labels = svm_classifier.predict(resampled_training_inputs)
    # print('I learned it allll')
    # b = svm_classifier.score(resampled_training_inputs, resampled_training_outputs_labels)
    # print(b)

    # a = svm_predictions_labels - resampled_training_outputs_labels
    # print(a.shape[0])
    # print(np.count_nonzero(a))
    clf = DecisionTreeClassifier(max_depth=50)

    # Train Decision Tree Classifer
    clf = clf.fit(resampled_training_inputs, resampled_training_outputs_labels)

    # Predict the response for test dataset
    # y_pred = clf.predict(resampled_training_inputs)
    print(clf.score(resampled_training_inputs, resampled_training_outputs_labels))

    data_validation = pd.read_csv("train.csv")

    data_validation = clean_data(data_validation)
    X_valid, y_valid = split_x_y(data_validation)
    print(clf.score(X_valid, y_valid))