import pandas as pd 
from pyECLAT import ECLAT


data=pd.read_csv("Bakkal2.csv",header=None)
veri=data.copy()
veri.columns=["Ürün"]
veri=list(veri["Ürün"].apply(lambda x:x.split(",")))


veri2=pd.DataFrame(veri)
print(veri)
minürün=2
mindeste=0.02
maxürün=max([len(x) for x in veri ])

ec=ECLAT(veri2,verbose=True)

a,b=ec.fit(min_support=mindeste,min_combination=minürün,max_combination=maxürün)
print(b)





