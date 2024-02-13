import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt 
data=sns.load_dataset("tips")
veri=data.copy()


kategori=[]
kategorik=veri.select_dtypes(include=["category"])

for i in kategorik.columns:
    kategori.append(i)


veri=pd.get_dummies(veri,columns=kategori,drop_first=True)

y=veri["tip"]
x=veri.drop(columns="tip",axis=1)#atma tip  yapar 




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

y_test=y_test.sort_index()
df=pd.DataFrame({"Gerçek":y_test,"tahmin":tahmin})
df.plot(kind="line")
print(mt.r2_score(y_test,tahmin))



