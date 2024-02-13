import pandas as pd 
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

data=pd.read_csv("spam1.csv",encoding="ISO-8859-9")
veri=data.copy()

veri=veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
veri=veri.rename(columns={"v1":"etiket","v2":"sms"})

veri2=veri["sms"].str.replace("[^\w\s]","")
veri3=veri2.str.lower()
veri4=veri2.str.replace("[\d]","")


etkisiz=stopwords.words("english")

ayır=veri4.str.split()

veri5=veri4.apply(lambda x:" ".join(x for x in x.split() if x not in etkisiz  ))

print(etkisiz)
print(veri4)
print(veri5)





