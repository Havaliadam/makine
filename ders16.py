import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np 

data=pd.read_csv("sarap.csv")
veri=data.copy()


y=veri["quality"]

X=veri.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

ld=LinearDiscriminantAnalysis(n_components=4)

X_train2=ld.fit_transform(X_train,y_train)
X_test2=ld.transform(X_test)

print(np.cumsum(ld.explained_variance_ratio_)*100)
print(len(np.unique(y_train)))

