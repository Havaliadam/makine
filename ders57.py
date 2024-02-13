import pandas as pd 
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re 
data=pd.read_csv("spam1.csv",encoding="ISO-8859-9")
veri=data.copy()

veri=veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
veri=veri.rename(columns={"v1":"etiket","v2":"sms"})

ps=PorterStemmer()

temiz=[]



for i in range(len(veri)):
    duzenle=re.sub('[^a-zA-Z]',' ',veri["sms"][i])
    duzenle=duzenle.lower()
    duzenle=duzenle.split()
    duzenle=[ps.stem(kelime) for kelime in duzenle if not kelime in set(stopwords.words("english"))]
    duzenle=' '.join(duzenle)
    temiz.append(duzenle)

print(veri["sms"][20])
print(temiz[20])


df=pd.DataFrame(list(zip(veri["sms"],temiz)),columns=["orjinal sms","temiz sms"])
print(df)










# veri2=veri["sms"].str.replace("[^\w\s]","")
# veri3=veri2.str.lower()
# veri4=veri2.str.replace("[\d]","",regex=True)


# etkisiz=stopwords.words("english")

# ayır=veri4.str.split()

# veri5=veri4.apply(lambda x:" ".join(x for x in x.split() if x not in etkisiz  ))

# ps=PorterStemmer()
# print(ps.stem("waiting"))

