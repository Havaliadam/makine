import pandas as pd 
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import AgglomerativeClustering

data=pd.read_csv("iris.csv")
veri=data.copy()

X=veri.drop(columns=["Id","Species"],axis=1)

hcsingle=linkage(X,method="single")
hccomplete=linkage(X,method="complete")
hcavarage=linkage(X,method="average")
hccentroid=linkage(X,method="centroid")
hcmedian=linkage(X,method="median")
hcward=linkage(X,method="ward")

fig,axes=plt.subplots(2,3)
dendrogram(hcsingle,ax=axes[0,0])
axes[0,0].set_title("Single")

dendrogram(hccomplete,ax=axes[0,1])
axes[0,1].set_title("complete")

dendrogram(hcavarage,ax=axes[0,2])
axes[0,2].set_title("avaragee")

dendrogram(hccentroid,ax=axes[1,0])
axes[1,0].set_title("centroid")

dendrogram(hcmedian,ax=axes[1,1])
axes[1,1].set_title("median")

dendrogram(hcward,ax=axes[1,2])
axes[1,2].set_title("ward")

model=AgglomerativeClustering(n_clusters=2,linkage="average")
tahmin=model.fit_predict(X)

labels=model.labels_

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",data=X,hue=labels,palette="deep")
plt.show()








