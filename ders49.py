import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
data=pd.read_csv("Müsteriler1.csv")

veri=data.copy()
veri=veri.dropna()#tekrar eden değerleri silindi

veri["Total"]=veri["Quantity"]*veri["UnitPrice"]
veri=veri.drop(veri[veri["Total"]<=0].index)


Q1=veri["Total"].quantile(0.25)
Q3=veri["Total"].quantile(0.75)


IQR=Q3-Q1

altsinir=Q1-1.5*IQR
ustsinir=Q3+1.5*IQR

veri=veri[~~((veri["Total"]>ustsinir) |(veri["Total"]<altsinir))]
veri=veri.reset_index(drop=True)

veri["CustomerID"]=veri["CustomerID"].astype("int")
veri["InvoiceDate"]=pd.to_datetime(veri["InvoiceDate"])

bugün=veri["InvoiceDate"].max()
bugün=dt.datetime(2011,12,9,12,49,0)

r=(bugün-veri.groupby("CustomerID").agg({"InvoiceDate":"max"})).apply(lambda x:x.dt.days)

f=veri.groupby(["CustomerID","InvoiceNo"]).agg({"InvoiceNo":"count"})
f=f.groupby("CustomerID").agg({"InvoiceNo":"count"})

m=veri.groupby("CustomerID").agg({"Total":"sum"})
RFM=r.merge(f,on="CustomerID").merge(m,on="CustomerID")
RFM=RFM.reset_index()
RFM=RFM.rename(columns={"CustomerID":"Customer","InvoiceDate":"Recency","InvoiceNo":"Frequency","Total":"Monetary"})

df=RFM.iloc[:,1:]


sc=MinMaxScaler()
dfnorm=sc.fit_transform(df)
dfnorm=pd.DataFrame(dfnorm,columns=df.columns)

kmodel=KMeans(random_state=0,n_clusters=4,init="k-means++")
kfit=kmodel.fit(dfnorm)
labels=kfit.labels_


RFM["Labels"]=labels
print(RFM.groupby("Labels").mean().iloc[:,1:])





# sns.scatterplot(x="Labels",y="Customer",data=RFM,hue=labels,palette="deep")
# plt.xlim([-1,5])
# plt.show()



# grafik=KElbowVisualizer(kmodel,k=(2,10))
# grafik.fit(dfnorm)
# grafik.poof()









# sns.boxplot(veri["Total"])
# plt.show()


     