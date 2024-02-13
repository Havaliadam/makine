import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
#mport statsmodels.api as sm 
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,cross_val_score
import sklearn.metrics as mt 
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet


data=pd.read_csv("ev1.csv")
veri=data.copy()

veri=veri.drop(columns="Address",axis=1)

# sns.pairplot(veri)
# plt.show()

# kor=veri.corr()
# sns.heatmap(kor,annot=True)
# plt.show()

y=veri["Price"]
X=veri.drop(columns="Price",axis=1)

# sabit=sm.add_constant(X)
# vif=pd.DataFrame()

# vif["Değişkeler"]=X.columns
# vif["VIF"]=[variance_inflation_factor(sabit,i+1)for i in range(X.shape[1])]
# print(vif)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

def caprazdog(model):
    dogruluk=cross_val_score(model,X,y,cv=10)
    return dogruluk.mean()


def basari(gercek,tahmin):
    rmse=mt.mean_squared_error(gercek,tahmin,squared=True)
    r2=mt.r2_score(gercek,tahmin)
    return [rmse,r2]



lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
lin_tahmin=lin_model.predict(X_test)

rid_model=Ridge(alpha=0.1)
rid_model.fit(X_train,y_train)
rid_tahmin=rid_model.predict(X_test)

las_model=Lasso(alpha=0.1)
las_model.fit(X_train,y_train)
las_tahmin=las_model.predict(X_test)

elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)
elas_tahmin=elas_model.predict(X_test)


sonuclar=[["Linear Model",basari(y_test,lin_tahmin)[0],basari(y_test,lin_tahmin)[1],caprazdog(lin_model)],
         ["Ridge Model",basari(y_test,rid_tahmin)[0],basari(y_test,rid_tahmin)[1],caprazdog(rid_model)],
          ["Lasso Model",basari(y_test,las_tahmin)[0],basari(y_test,las_tahmin)[1],caprazdog(las_model)],
           ["Elastic Model",basari(y_test,elas_tahmin)[0],basari(y_test,elas_tahmin)[1],caprazdog(elas_model)]  
           
    
           ]
pd.options.display.float_format='{:,.4f}'.format
sonuclar=pd.DataFrame(sonuclar,columns=["Model","RMSE","R2","DOGRULAMA"])
print(sonuclar)



