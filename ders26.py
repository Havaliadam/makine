import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("kanser.csv")
veri=data.copy()


veri=veri.drop(columns=["id","Unnamed: 32"]
               )

y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


lc=LogisticRegression(random_state=0)
lc.fit(X_train,y_train)
tahmin=lc.predict(X_test)
print(tahmin)


