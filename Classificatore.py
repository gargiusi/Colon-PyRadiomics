from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,KFold,RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2, f_classif,SelectFromModel,SelectPercentile
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics,linear_model
from sklearn.metrics import classification_report, confusion_matrix
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
warnings.simplefilter('ignore')

#prepara training e test set del dataset
Filename="C:\\Users\\rac-g\\Desktop\\work\\10_STAS.csv" ###################### Enter the full path of csv dataset
data=np.loadtxt(Filename,delimiter=';',skiprows=1)
data2=np.loadtxt(Filename,delimiter=';',skiprows=0,dtype=str)
features=data2[0,1:-1]
X=data[:,1:-1] #Data=Features
y=data[:,-1] #Target=STAS

X_train, X_test, y_train, y_test=train_test_split(X,y,stratify=y,random_state=0)#con stratify=y abbiamo che test size=25%
# Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#..................................................................................................................


#             Univariate Features Selection
#................................................#
T=30 #iNSERISCI IL NUMERO DI FEATURES MESSE IN GIOCO
plt.figure(1)
plt.clf()
X_indices = np.arange(X.shape[-1])
selector = SelectKBest(f_classif, k=T)
#selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices, scores,width=.2,
        label=r'Univariate score',color = 'b')#IN BLU:FEATURES DI PARTENZA(TOTALI)


del(X,features)
U=[]
Scores=[]
New_Indice=[]
feature_name=[]
for i in X_indices[selector.get_support()]:
    Indice=i
    New_Indice.append(Indice)
    x=data[:,i]
    new_feature_name = data2[0, i]
    Score=scores[i]
    U.append(x)
    feature_name.append(new_feature_name)
    Scores.append(Score)
X=np.asarray(U).transpose()
Indices=np.asarray(New_Indice).transpose()
NewScores=np.asarray(Scores).transpose()
features=np.asarray(feature_name).transpose()
#print(features)

plt.bar(Indices, NewScores,width=.2,
        label=r'Univariate score Selected',color = 'g' )#IN VERDE: FEATURES SELEZIONATE CON ANOVA

plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.ylabel('$-Log(p_{value})$')
plt.axis('tight')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()#GRAFICO

del(X_train,X_test,y_train,y_test,Score,Scores,NewScores,scaler,New_Indice,U,Indices,Indice,x,feature_name,new_feature_name)


plt.figure(2)
plt.scatter(X[:,5],X[:,25],c=y)
plt.colorbar()
plt.show()




#          K-Fold split Data with K-NN Classificator After Univariate features selection
#....................................#
kf=KFold(n_splits=10)
kf.get_n_splits(X)
for train_index,test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

# Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

KNN=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(X)})
KNN = KNN.fit(X_train, y_train)

#y_pred = KNN.predict(X_test)
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))


#print('TRAIN')
#print('Accuracy model: %.3f' % str(metrics.accuracy_score(KNN.predict(np.array(X_train)), y_train)*100)+'%')
#print('Precision and Sensibility model: %.3f' % str(metrics.f1_score(KNN.predict(np.array(X_train)), y_train)*100)+'%')#SENSIBILITà + PRECISIONE

print('TEST')
print(str(metrics.accuracy_score(KNN.predict(np.array(X_test)), y_test)*100)+'%')
print(str(metrics.f1_score(KNN.predict(np.array(X_test)), y_test)*100)+'%')


del(train_index,test_index,KNN,kf,scaler)


#...............................................................................................................

#            LASSO
#...................................#
model = Lasso(alpha=1,max_iter=1000)
model.fit(X, y)

Coef_Lasso=model.coef_
New_Indice=[]
for i in range(0,T):
    if Coef_Lasso[i]!=0:
        Indice=i
        New_Indice.append(Indice)
Indices=np.asarray(New_Indice).transpose()

U=[]
feature_name=[]
for i in Indices:
    x=X[:,i]
    new_feature_name=features[i]
    U.append(x)
    feature_name.append(new_feature_name)
del(X,features)
X=np.asarray(U).transpose()
features=np.asarray(feature_name).transpose()

del(X_train,X_test,y_train,y_test,New_Indice,U,Indices,x,model,feature_name,new_feature_name,Indice)



#          K-Fold split Data with Linear Regression Classificator After Univariate features selection and Lasso
#............................................................................................#
kf=KFold(n_splits=10)
kf.get_n_splits(X)
for train_index,test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

# Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
#score=model.score(X_test,y_test)
print('Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,y_predict))
print('Coefficient Of Determination: %.3f' % metrics.r2_score(y_test,y_predict))

for i in range(len(model.coef_)):
    print(features[i],'\t',model.coef_[i])
