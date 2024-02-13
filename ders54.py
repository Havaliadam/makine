import pdfplumber
import pandas as pd 

with pdfplumber.open("C:/Users/Acer/OneDrive/Masaüstü/MAKNEDERS/Gençliğe-hitabe.pdf")as pdf :
    sayfa=pdf.pages[0]
    metin=sayfa.extract_text()



metin=metin.replace("!",".")
d1=metin.split(".")


d2=[]

for i in d1:
    d2.append(i.replace("\n","").strip(" "))

df=pd.DataFrame(d2,columns=["Cümleler"])
print(df)


