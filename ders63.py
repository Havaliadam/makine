import pandas as pd 
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# filmler öneri sistenler
data=pd.read_csv("filmler.csv",low_memory=False)
veri=data.copy()

df=veri[["id","title","overview"]]

tdidf=TfidfVectorizer(stop_words="english")
df["overview"]=df["overview"].fillna(" ")
tdidf_matris=tdidf.fit_transform(df["overview"])


benzerlik_matrisi=linear_kernel(tdidf_matris,tdidf_matris)



indesler=pd.Series(df.index,index=df["title"]).drop_duplicates()

film_indeks=indesler["Toy Story"]


filmbenzerlik=list(enumerate(benzerlik_matrisi[film_indeks]))
sıralama=sorted(filmbenzerlik,key=lambda x:x[1],reverse=True)
sıralamapuan=sıralama[1:6]
sıralamaindeks=[i[0] for i in sıralamapuan]
sonuc=df["title"].iloc[sıralamaindeks]
print(sonuc)






