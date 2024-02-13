import pandas as pd 
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt 
data=pd.read_csv("reklam.csv")
veri=data.copy()

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(veri)
veri.iloc[:,:]=imputer.transform(veri)

y=veri["Sales"]

x=veri[["TV","Radio"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=150)



lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)
r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=False)
mae=mt.mean_absolute_error(y_test,tahmin)
print(r2,mse,rmse,mae)
