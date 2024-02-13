import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("müsteri.csv")
veri=data.copy()
veri=veri.drop(columns="customerID",axis=1)

#shape boyut gösteri
#info içerik
#customerid veriden atma  işlemi yaptık 

veri=veri.rename({"gender":"cinsiyet","SeniorCitizen":"65 yaş üstü","Partner":"Medeni Durum","Dependents":"Bakma sorumluğunu","tenure":"müşteri olma süresi (ay)","PhoneService":"ev telefonu aboneliği","MultipleLines":"Birden fazla abonelik durumu","OnlineSecurity":"intenet aboneği","OnlineSecurity":"Güvenlik hizmeti aboneliği","OnlineBackup":"yedekleme hizmeti aboneliği","DeviceProtection":"ekipman güvenlik aboneliği","TechSupport":"teknik destek aboneliği","StreamingTV":"Ip tv aboneliği","StreamingMovies":"film aboneliği"
,"Contract":"Sözleşme süresi","PaperlessBilling":"e-fatura","PaymentMethod":"ödeme sekli","MonthlyCharges":"Aylik ücret",
"TotalCharges":"toplam ücret","Churn":"kayıp durumu"},axis=1)

veri["cinsiyet"]=["erkek" if kod=="Male" else "kadin" for kod in veri["cinsiyet"]]
veri["65 yaş üstü"]=["evet" if kod==1 else "hayir" for kod in veri["65 yaş üstü"]]
veri["Medeni Durum"]=["evli" if kod=="Yes" else "bekar" for kod in veri["Medeni Durum"]]
veri["Bakma sorumluğunu"]=["var" if kod=="Yes" else "yok" for kod in veri["Bakma sorumluğunu"]]
veri["ev telefonu aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["ev telefonu aboneliği"]]
veri["Birden fazla abonelik durumu"]=["var" if kod=="Yes" else "yok" for kod in veri["Birden fazla abonelik durumu"]]
veri["InternetService"]=["yok" if kod=="No" else "var" for kod in veri["InternetService"]]
veri["Güvenlik hizmeti aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["Güvenlik hizmeti aboneliği"]]
veri["yedekleme hizmeti aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["yedekleme hizmeti aboneliği"]]
veri["ekipman güvenlik aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["ekipman güvenlik aboneliği"]]
veri["teknik destek aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["teknik destek aboneliği"]]
veri["Ip tv aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["Ip tv aboneliği"]]
veri["film aboneliği"]=["var" if kod=="Yes" else "yok" for kod in veri["film aboneliği"]]
veri["Sözleşme süresi"]=["1 aylik" if kod=="Month-to-month" else "1 yillik" if kod=="One year" else "2 yillik" for kod in veri["Sözleşme süresi"]]
veri["e-fatura"]=["evet" if kod=="Yes" else "yok" for kod in veri["e-fatura"]]
veri["ödeme sekli"]=["elektronik ödeme" if kod=="Electronic check" else "mail" if kod=="Mailed check" else "banka transfer" if kod=="Bank transfer (automatic)" else "kredi karti"  for kod in veri["ödeme sekli"]]
veri["kayıp durumu"]=["evet" if kod=="Yes" else "yok" for kod in veri["kayıp durumu"]]
veri["toplam ücret"]=pd.to_numeric(veri["toplam ücret"],errors="coerce")



veri=veri.dropna()
print(veri)

le=LabelEncoder()
degisken=veri.select_dtypes(include="object").columns
veri.update(veri[degisken].apply(le.fit_transform))

veri["kayıp durumu"] = le.fit_transform(veri["kayıp durumu"])

y=veri["kayıp durumu"]
X=veri.drop(columns="kayıp durumu",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

models=["LinearSVC","SVC","Ridge","Logistic","RandomForest","LGBM","XGBM"]

siniflar=[LinearSVC(random_state=0,C=0.1
,penalty="l1",dual=False)
          ,SVC(random_state=0,C=1,gamma=0.01),
          RidgeClassifier(random_state=0,alpha=0.1),
          LogisticRegression(random_state=0,C=0.1,penalty="l2",dual=False),
          RandomForestClassifier(random_state=0,max_depth=10,min_samples_split=2,n_estimators=2000),
          LGBMClassifier(random_state=0,learning_rate=0.01,max_depth=4,n_estimators=1000,subsample=0.6),
          XGBClassifier(learning_rate=0.001,max_depth=4,n_estimators=2000,subsample=0.8)
          ]

# parametreler={
#     models[0]:{"C":[0.1,1,10,100],"penalty":["l1","l2"]},
#     models[1]:{"kernel":["linear","rbf"],"C":[0.1,1],"gamma":[0.01,0.001]},
#     models[2]:{"alpha":[0.1,1.0]},
#     models[3]:{"C":[0.1,1],"penalty":["l1","l2"]},
#     models[4]:{"n_estimators":[1000,2000],"max_depth":[4,10],"min_samples_split":[2,5]},
#     models[5]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],"subsample":[0.6,0.8]},
#     models[6]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],"subsample":[0.6,0.8]}
# }


def cozum(model):
    model.fit(X_train,y_train)
    return model


def skor(model2):
    tahmin=cozum(model2).predict(X_test)
    acs=accuracy_score(y_test,tahmin)
    return acs*100

basari=[]

for i in siniflar:
    basari.append(skor(i))

a=list(zip(models,basari))
sonuc=pd.DataFrame(a,columns=["Model","Basari"])

print(sonuc.sort_values("Basari",ascending=False))

# for i,j in zip(models,siniflar):
#         print(i)
#         grid=GridSearchCV(cozum(j),parametreler[i],cv=10,n_jobs=-1)
#         grid.fit(X_train,y_train)
#         print(grid.best_params_)












# clf=LazyClassifier()
# modeller,tahmin=clf.fit(X_train,X_test,y_train,y_test)
# sira=modeller.sort_values(by="Accuracy",ascending=True)
# plt.barh(sira.index,sira["Accuracy"])

# plt.show()







