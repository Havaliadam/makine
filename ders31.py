import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data=pd.read_csv("kanser.csv")
veri=data.copy()

veri=veri.drop(columns=["id","Unnamed: 32"],axis=1)

veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]

y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=
                         9)
knn.fit(X_train,y_train)
tahmin=knn.predict(X_test)

acs=accuracy_score(y_test,tahmin)
print(acs)

# basari=[]

# for k in range(1,20):
#     knn=KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train,y_train)
#     tahmin2=knn.predict(X_test)
#     basari.append(accuracy_score(y_test,tahmin2))

# print(max(basari))


