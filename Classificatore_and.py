from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,KFold,RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2, f_classif,SelectFromModel,SelectPercentile
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics,linear_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LassoCV,LinearRegression,Lasso
import numpy as np
from numpy import where,mean,std,absolute
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import _random_over_sampler,BorderlineSMOTE,SMOTENC,SMOTE,RandomOverSampler
from sklearn.linear_model import LogisticRegression,LassoCV
from imblearn.pipeline import make_pipeline,Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial.distance import cdist
from collections import Counter
import warnings
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
warnings.simplefilter('ignore')

#prepara training e test set del dataset
Filename='/media/andrea/DATA/STAS/rachele/codice_tesi/Tesi//10_STAS.csv' ###################### Enter the full path of csv dataset
data=np.loadtxt(Filename,delimiter=';',skiprows=1)
data2=np.loadtxt(Filename,delimiter=';',skiprows=0,dtype=str)
features=data2[0,1:-16]
X=data[:,1:-16] #Data=Features
y=data[:,-1] #Target=STAS

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,stratify=y)#con stratify=y abbiamo che test size=25%
# Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% #..................................................................................................................


#             Univariate Features Selection
#................................................#
#standardizzo
X_uni = X_train
y_uni=y_train
#y_uni=np.random.permutation(y_uni)
T=5 #iNSERISCI IL NUMERO DI FEATURES MESSE IN GIOCO
plt.figure(1)


selector = SelectKBest(f_classif, k=T)
#selector = SelectPercentile(f_classif, percentile=10)
X_uni =selector.fit_transform(X_uni, y_uni)
X_test=selector.transform(X_test)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
sorted_features=np.argsort(scores)
print(features[sorted_features[-T::-1]])


#istogramma tutte
f,ax=plt.subplots(1)
ax.hist(selector.pvalues_,25,density=True)
ax.set_title('p-value, full dataset')
ax.set_xlabel('p-value')
ax.set_ylabel('Probability density')
#istogramma best
f,ax=plt.subplots(1)
ax.hist(selector.pvalues_[sorted_features[-T:]],15,density=True)
ax.set_title('p-value, best 30')
ax.set_xlabel('p-value')
ax.set_ylabel('Probability density')

# %%histogram best 4 features
f,axex=plt.subplots(4,2,figsize=(15,10))
for i, ax in enumerate(axex.flat):
    sel=sorted_features[-i-1]
    ax.hist(X[y==1,sel],density=True,histtype='bar',alpha=0.4,label='STAS')
    ax.hist(X[y==0,sel],density=True,histtype='bar',alpha=0.4,label='NOSTAS')
    # "{:.2e}".format(12300000)
    ax.set_title("{:.2e}".format(selector.pvalues_[sel],3))
    ax.set_xlabel(features[sel])
    ax.set_ylabel('Probability density')
    ax.legend()
f.tight_layout()

# %% matrice di correlazione
correlation_p=np.corrcoef(X_uni,rowvar=False)
names = features[sorted_features[-T:]]
f = plt.figure(figsize=(2*T, 2*T))
plt.matshow(correlation_p, fignum=f.number)
plt.xticks(range(len(names)),names, fontsize=14, rotation=45)
plt.yticks(range(len(names)),names, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);


# %%
if 1:
    oversample = SMOTE(k_neighbors=3)
    X_uni, y_uni = oversample.fit_resample(X_uni, y_uni)

# %%



KNN=KNeighborsClassifier(n_neighbors=3, weights='uniform',algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(X_uni)})
KNN = KNN.fit(X_uni, y_uni)

K_pred_train = KNN.predict(np.array(X_uni))

print('TRAIN')
print('Accuracy model on train: %.2f ' % (metrics.accuracy_score(K_pred_train, y_uni)))
print('Precision and Sensibility model on train: %.2f' % (metrics.f1_score(K_pred_train, y_uni)))


K_pred_test = KNN.predict(X_test)
print('TEST')
print('Accuracy model on test: %.2f ' % (metrics.accuracy_score(K_pred_test, y_test)))
print('Precision and Sensibility model on test: %.2f' % (metrics.f1_score(K_pred_test, y_test)))

K_pred_base = np.ones(y_test.shape)
print('Stupido classificatore con tutti 1 on TEST')
print('Accuracy base on test: %.2f ' % (metrics.accuracy_score(K_pred_base, y_test)))
print('Precision and Sensibility base on test: %.2f' % (metrics.f1_score(K_pred_base, y_test)))


#matrici di confusione

confusion_matrix( y_test,K_pred_base)/len( y_test)
#questo è con il pacchetto
#disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 #display_labels=class_names,
                                 #cmap=plt.cm.Blues,
                                 #normalize=normalize)
#disp.ax_.set_title(title)

# %% matrice di correlazione
confusion_m=confusion_matrix( y_test,K_pred_base)/len( y_test)
df_cm = df_cm = pd.DataFrame(confusion_m, index=["NOSTAS","STAS"], columns=["NOSTAS","STAS"])
plt.figure(figsize=(5,5))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # 

# =============================================================================
# # %%
# 
# #          K-Fold split Data with K-NN Classificator After Univariate features selection
# #....................................#
# kf=KFold(n_splits=10)
# kf.get_n_splits(X)
# for train_index,test_index in kf.split(X):
#     X_train,X_test=X[train_index],X[test_index]
#     y_train,y_test=y[train_index],y[test_index]
# 
# # Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# 
# KNN=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(X)})
# KNN = KNN.fit(X_train, y_train)
# 
# #y_pred = KNN.predict(X_test)
# #print(classification_report(y_test, y_pred))
# #print(confusion_matrix(y_test, y_pred))
# 
# 
# #print('TRAIN')
# #print('Accuracy model: %.3f' % str(metrics.accuracy_score(KNN.predict(np.array(X_train)), y_train)*100)+'%')
# #print('Precision and Sensibility model: %.3f' % str(metrics.f1_score(KNN.predict(np.array(X_train)), y_train)*100)+'%')#SENSIBILITà + PRECISIONE
# 
# print('TEST')
# print(str(metrics.accuracy_score(KNN.predict(np.array(X_test)), y_test)*100)+'%')
# print(str(metrics.f1_score(KNN.predict(np.array(X_test)), y_test)*100)+'%')
# 
# 
# del(train_index,test_index,KNN,kf,scaler)
# 
# 
# #...............................................................................................................
# 
# #            LASSO
# #...................................#
# model = Lasso(alpha=1,max_iter=1000)
# model.fit(X, y)
# 
# Coef_Lasso=model.coef_
# New_Indice=[]
# for i in range(0,T):
#     if Coef_Lasso[i]!=0:
#         Indice=i
#         New_Indice.append(Indice)
# Indices=np.asarray(New_Indice).transpose()
# 
# U=[]
# feature_name=[]
# for i in Indices:
#     x=X[:,i]
#     new_feature_name=features[i]
#     U.append(x)
#     feature_name.append(new_feature_name)
# del(X,features)
# X=np.asarray(U).transpose()
# features=np.asarray(feature_name).transpose()
# 
# del(X_train,X_test,y_train,y_test,New_Indice,U,Indices,x,model,feature_name,new_feature_name,Indice)
# 
# 
# 
# #          K-Fold split Data with Linear Regression Classificator After Univariate features selection and Lasso
# #............................................................................................#
# kf=KFold(n_splits=10)
# kf.get_n_splits(X)
# for train_index,test_index in kf.split(X):
#     X_train,X_test=X[train_index],X[test_index]
#     y_train,y_test=y[train_index],y[test_index]
# 
# # Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# 
# model=LinearRegression()
# model.fit(X_train,y_train)
# y_predict=model.predict(X_test)
# #score=model.score(X_test,y_test)
# print('Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,y_predict))
# print('Coefficient Of Determination: %.3f' % metrics.r2_score(y_test,y_predict))
# 
# for i in range(len(model.coef_)):
#     print(features[i],'\t',model.coef_[i])
# 
# =============================================================================
