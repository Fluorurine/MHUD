import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("train.csv")

#
#          Xem trong Readme.md de biet them ve du lieu
#
print("So luong hang, cot trong data (Ke ca nhan):")
print(data.shape)

# in ra mảng chứa số lượng nhãn đặc trưng
print("So luong nhan dac trung la :", np.unique(data.price_range))

print("So luong cac nhan :")
print(data.price_range.value_counts())
# 0: 500 1:500 2:500 3:500

# Su dung nghi thức K-Fold

kf = KFold(n_splits=100, shuffle=True)
total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=15, min_samples_leaf=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100



print("Do chinh xac trung binh cua Decision Tree (gini, maxdepth 15, min_leaf 4) :", total_acc/100, "%")

total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    RFC = RandomForestClassifier(criterion="gini", random_state=100, max_depth=15, min_samples_leaf=4)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100
print("Do chinh xac trung binh cua RandomForest (gini, maxdepth 15, min_leaf 4) :", total_acc/100, "%")

total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    y_pred = GNB.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100


print("Do chinh xac trung binh cua Naive Bayes theo phan phoi Gaussian :", total_acc/100, "%")

total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    KNN = KNeighborsClassifier(n_neighbors=20)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100
print("Do chinh xac trung binh cua KNN voi n =20 :", total_acc/100, "%")

