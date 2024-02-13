import pandas as pd 
from sklearn.metrics import pairwise_distances


# links=pd.read_csv("links.csv")
movies=pd.read_csv("movies.csv")
rating=pd.read_csv("ratings.csv")
#tags=pd.read_csv("tags.csv")




df=pd.merge(movies,rating,on="movieId")

df=df[["userId","title","rating"]]

pivot=df.pivot_table(index="title",columns="userId",values="rating",fill_value=0)

benzerlik_matirsi=pairwise_distances(pivot,metric="correlation")

indeksler=list(pivot.index)
film_indeks=indeksler.index("Godfather, The (1972)")

benzer_indeksler=(benzerlik_matirsi[film_indeks].argsort()[1:6])
# print(benzer_indeksler)

for i in benzer_indeksler:
    print(indeksler[i])


