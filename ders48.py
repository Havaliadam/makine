import requests

from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np

url="https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=03#page-1"
r=requests.get(url)
s=BeautifulSoup(r.text,"html.parser")

tablo=s.find("table",{"id":"summaryBasicData"})
tablo=pd.read_html(str(tablo),flavor="bs4")[0]


hisseler=[]


for i in tablo["Kod"]:
    hisseler.append(i)


parametreler=(
    ("hisse",hisseler[0]),
    ("startdate","28-11-2020"),
    ("enddate","28-11-2022"))


url2="https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
r2=requests.get(url2,params=parametreler).json()["value"]
veri=pd.DataFrame.from_dict(r2)
veri=veri.iloc[:,0:3]


veri=veri.rename({"HGDG_HS_KODU":"HİSSE","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

data = {"Tarih":veri["Tarih"],veri["HİSSE"][0]:veri["Fiyat"]}
veri=pd.DataFrame(data)
print(veri)




tumveri=[veri]
for j in hisseler:
    parametreler=(
    ("hisse",j),
    ("startdate","28-11-2020"),
    ("enddate","28-11-2022"))


    url2="https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
    r2=requests.get(url2,params=parametreler).json()["value"]
    veri=pd.DataFrame.from_dict(r2)
    veri=veri.iloc[:,0:3]
    veri=veri.rename({"HGDG_HS_KODU":"HİSSE","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

    data = {"Tarih":veri["Tarih"],veri["HİSSE"][0]:veri["Fiyat"]}
    veri=pd.DataFrame(data)
    tumveri.append(veri) 

df=tumveri[0]


for son in tumveri[:1]:
    df=df.merge(son,on="Tarih")

veri=df.drop(columns="Tarih",axis=1)

gelir=veri.pct_change().mean()*252
sonuc=pd.DataFrame(gelir)
sonuc.columns=["gelir"]

sonuc["oynaklık"]=veri.pct_change().std()*np.sqrt(252)
sonuc=sonuc.reset_index()
sonuc=sonuc.rename({"index":"HİSSE"},axis=1)
sonuc.to_csv("C:/Users/Acer/OneDrive/Masaüstü/MAKNEDERS/veri.csv",index=False)




















