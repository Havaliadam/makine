import pandas as pd 
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re 
nltk.download("wordnet")
lema=nltk.WordNetLemmatizer()
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
import numpy as np 
from PIL import Image

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
    duzenle=[lema.lemmatize(kelime) for kelime in duzenle if not kelime in set(stopwords.words("english"))]
    duzenle=' '.join(duzenle)
    temiz.append(duzenle)




df=pd.DataFrame(list(zip(veri["sms"],temiz)),columns=["orjinal sms","temiz sms"])

frekans=(df["temiz sms"]).apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
frekans.columns=["kelimeler","frekans"]

kelimeler=dict(frekans.values)


resim=np.array(Image.open("indir.png"))

plt.figure(figsize=(5,5))
bulut=WordCloud(background_color="Black",mask=resim,contour_width=3,contour_color="white",max_words=250).generate_from_frequencies(kelimeler)

plt.imshow(bulut)
plt.axis("off")

plt.show()





