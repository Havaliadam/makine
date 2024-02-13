import pandas as pd 
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB# metin analistte 
from sklearn.metrics import accuracy_score
import numpy as np 
data=pd.read_csv("spam1.csv",encoding="Windows-1252")
veri=data.copy()

veri=veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)


veri=veri.rename(columns={"v1":"Etiket","v2":"Sms"})
#print(veri.groupby("Etiket").count())

veri=veri.drop_duplicates()#tekrar verileri kaldırma 

veri["Karakter Sayisi"]=veri["Sms"].apply(len)
veri.Etiket=[1 if kod=="spam" else 0 for kod in veri.Etiket]
def harfler(cumle):
    yer=re.compile("[^a-zA-Z]")
    return re.sub(yer," ",cumle)



durdurma=stopwords.words("english")
#print(durdurma)

spam=[]
ham=[]
tumcumleler=[]
for i in range(len(veri["Sms"].values)):
    r1=veri["Sms"].values[i]
    r2=veri["Etiket"].values[i]

    temizcumle=[]
    cumleler=harfler(r1)

    cumleler=cumleler.lower()
      
    for kelimeler in cumleler.split():
         temizcumle.append(kelimeler)

         if r2==1:
            spam.append(cumleler)
         else:
             ham.append(cumleler)
    tumcumleler.append(" ".join(temizcumle))


veri["Yeni Sms"]=tumcumleler



veri=veri.drop(columns=["Sms","Karakter Sayisi"],axis=1)

cv=CountVectorizer()
x=cv.fit_transform(veri["Yeni Sms"]).toarray()

y=veri["Etiket"]
X=x

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
for i in np.arange(0.0,1.1,0.1):
    model=MultinomialNB(alpha=i)
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    skor=accuracy_score(y_test,tahmin)
    print("ALFA :{} değeri için skor :{}".format(round(i,1),round(skor*100,2)))




# print(harfler("tuy,,,?RTR11"))

# mesaj=re.sub("[^a-zA-Z]"," ",veri["Sms"][0])
# print(veri["Sms"][0])
# print(mesaj)


# veri.hist(column="Karakter Sayisi",by="Etiket",bins=50)
# plt.show()