import pandas as pd 
import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error

veri=pd.read_excel("ornek.xlsx")

y=veri["y"]
X=veri[["X1","X2"]]
sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit()
# print(model.summary())

tahmin=model.predict(sabit)
#print(tahmin)

# rmse=mean_squared_error(y,tahmin,squared=False)

# print(rmse)

mae=mean_absolute_error(y,tahmin)

print(mae)

