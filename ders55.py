import pandas as pd 
import re 
data=pd.read_csv("spam1.csv",encoding="ISO-8859-9")
veri=data.copy()

veri=veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
veri=veri.rename(columns={"v1":"etiket","v2":"sms"})

veri2=veri["sms"].str.replace("[^\w\s]","")
veri3=veri2.str.lower()
veri4=veri2.str.replace("[\d]","")

print(veri3)
print(veri4)





