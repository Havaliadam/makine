import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
data=pd.read_csv("sarap.csv")
veri=data.copy()

# kor=veri.corr()
# sns.heatmap(kor,annot=True,cbar=True)
# plt.show()
y=veri["quality"]

X=veri.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


pca=PCA()

X_train2=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


print(np.cumsum(pca.explained_variance_ratio_)*100)#kümeletik olakar
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("bileşen sayisi")
plt.ylabel("açıklanan varyans")
plt.show()