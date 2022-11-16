import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./train.csv")

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
print("Dataset co cot trong (NaN) khong ?", data.isnull().any().any())
# Su dung nghi thức K-Fold
from sklearn.metrics import confusion_matrix

#Kiem thu Decision Tree bằng K-Fold (K=10) 10 lần
print("Bat dau ap dung K-Fold(K=10) 10 lan")
total_acc_10 = 0
for i in range(1, 11):
    print("Lan thu", i)
    #Khoi tao theo KFold = 10
    kf = KFold(n_splits=10, shuffle=True)
    total_acc = 0
    for train_index, test_index in kf.split(data):

        # Lay index theo mang tra ve
        X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
        y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
        clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=9, min_samples_leaf=8)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        total_acc += accuracy_score(y_test, y_pred) * 100



    print("Do chinh xac cua Decision Tree (gini, maxdepth 9, min_leaf 8) : qua 10 lan lap:", accuracy_score(y_test, y_pred) * 100, "%")
    total_acc_10+=total_acc/10





print("Do chinh xac trung binh cua Decision Tree (gini, maxdepth 9, min_leaf 8) :", total_acc/10   , "%")


total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    y_pred = GNB.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100


print("Do chinh xac trung binh cua Naive Bayes theo phan phoi Gaussian :", total_acc/10, "%")

total_acc = 0
for train_index, test_index in kf.split(data):

# Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    KNN = KNeighborsClassifier(n_neighbors=10)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100
print("Do chinh xac trung binh cua KNN voi n =10 :", total_acc/10, "%")
total_acc = 0
for train_index, test_index in kf.split(data):

    # Lay index theo mang tra ve
    X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
    y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
    RFC = RandomForestClassifier(criterion="gini", random_state=100, max_depth=15, min_samples_leaf=4)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    total_acc += accuracy_score(y_test, y_pred) * 100
print("Do chinh xac trung binh cua RandomForest (gini, maxdepth 15, min_leaf 4) :", total_acc/10, "%")

#Phan xu ly file csv
# từ model đã xây dựng lấy kết quả từ test.csv dự đoán và xuất ra file
X_train= data.iloc[train_index, 0:20]
clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=9, min_samples_leaf=8)
clf.fit(X_train, y_train)
X_test= pd.read_csv("./test.csv", index_col=0)
y_pred = clf.predict(X_test)


#Nếu cần lưu kết quả vào file dự đoán VD:ketqua.csv thì sử dụng hàm này
total_acc = 0
print("Da xuat ket qua tu test.csv ra file")

# df = pd.DataFrame(y_pred, columns=['Ketqua'])
# df.index.names = ['Id']
# df.to_csv('./Ketqua.csv')

#Do đã có file Ketqua.csv rồi nên không chạy đoạn chương trình trên