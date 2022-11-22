#Đây là file phụ dùng để kiểm tra các tham số đối với Decision Tree xem tham số nào hiệu quả nhất
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
data = pd.read_csv("./train.csv")

# Kiểm tra bằng K-Fold (Ngẫu nhiên)
kf = KFold(n_splits=10, shuffle=True,random_state=100)

max = 0
maxa = 0
maxb = 0
for a in range(4, 12):
    for b in range(3, 12):
        total_acc = 0

        for train_index, test_index in kf.split(data):
            # Lay index theo mang tra ve
            X_train, X_test = data.iloc[train_index, 0:20], data.iloc[test_index, 0:20]
            y_train, y_test = data.price_range.iloc[train_index], data.price_range[test_index]
            clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=a, min_samples_leaf=b)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc=accuracy_score(y_test, y_pred)
            total_acc += acc * 100


        print("Decision Tree(", a, b, "):", total_acc / 10, "%")
        if (max < total_acc/10):

            max = total_acc/10
            print(max)
            maxa = a
            maxb = b
print("Nhu vay Decision Tree K-Fold lon nhat la",max,"Voi a=",maxa,"b=",maxb)
# Kiểm tra bằng Hold-Out đã có random_state=100 nên cố định

from sklearn.model_selection import train_test_split
max = 0
maxa = 0
maxb = 0
#Tuy chinh a,b (Nhập tham số cho chương trình)
a = int(input("Nhập a(Chạy từ 5->a)"))
b = int(input("Nhập b(Chạy từ 5->b)"))
for i in range(5, a+1):
    for j in range(5, b+1):
        X_train,X_test,y_train,y_test = train_test_split(data.iloc[:, 0:20],data.price_range,test_size=3/10.0,random_state=100)
        clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=i, min_samples_leaf=j)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Decision Tree(", i, j, "):", round(accuracy_score(y_test, y_pred) * 100,2), "%")
        if (max < round(accuracy_score(y_test, y_pred) * 100,2)):
            print(max)
            max = accuracy_score(y_test, y_pred) * 100
            maxa = i
            maxb = j
print("Nhu vay Decision Tree Hold_Out lon nhat la",round(max,2),"Voi a=",maxa,"b=",maxb)