import pandas as pd 
from sklearn.model_selection import train_test_split

veri=pd.read_excel("C:/Users/Acer/OneDrive/Masaüstü/MAKNEDERS/ornek.xlsx")
#print(veri)

y=veri["y"]
X=veri[["X1","X2"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.sum())
# print(X_test)
