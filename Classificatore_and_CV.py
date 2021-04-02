from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,KFold,RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2, f_classif,SelectFromModel,SelectPercentile
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics,linear_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LassoCV,LinearRegression,Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import _random_over_sampler,BorderlineSMOTE,SMOTENC,SMOTE,RandomOverSampler,ADASYN
from sklearn.linear_model import LogisticRegression,LassoCV
from collections import Counter
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold


#parametri (in futuro da riga di comando)
param = {
    'test_hold':0.25,
    'stratify':True,
    'N_features': 25, 
    'corr_cut':0.85, 
    'do_scaler': True,
    'do_corr_cut': True,
    'do_SMOTE': False,
    'shuffle_labels': False,
    'lasso_cut': True
    
    }  





#prepara training e test set del dataset
Filename='/media/andrea/DATA/STAS/rachele/codice_tesi/Tesi//10_STAS.csv' ###################### Enter the full path of csv dataset
data=np.loadtxt(Filename,delimiter=';',skiprows=1)
data2=np.loadtxt(Filename,delimiter=';',skiprows=0,dtype=str)
features=data2[0,1:-16]
X=data[:,1:-16] #Data=Features
y=data[:,-1] #Target=STAS
if param['shuffle_labels']:
    y=np.random.permutation(y)

skf = StratifiedKFold(n_splits=4,shuffle=True)
#auc_total=np.zeros([4])
acc_before_lasso=np.zeros([4])
acc_after_lasso=np.zeros([4])
selected_feature_out=[]

for k, [train_out, test_out] in enumerate(skf.split(X, y)):
    print('\n\tFOLD = {}\n'.format(k))
    #X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=param['test_hold'],shuffle=True,stratify=(y if param['stratify']  else None))#con stratify=y abbiamo che test size=25%
    # Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
    X_train=X[train_out,:]
    y_train=y[train_out]
    X_test=X[test_out,:]
    y_test=y[test_out]
    
    if param['do_scaler']:
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
    T=param['N_features'] #iNSERISCI IL NUMERO DI FEATURES MESSE IN GIOCO
    plt.figure(1)
    
    
    selector = SelectKBest(f_classif, k=T)
    #selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X_uni, y_uni)
    
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    sorted_features=np.argsort(selector.pvalues_)
    print(features[sorted_features[:T]])
    
    # %%trasformo e ordino - lo faccio a mano perchè così sono ordinati
    print(''.join(45*['-']))
    print('SELEZIONO {} Features'.format(T))
    X_uni=X_uni[:,sorted_features[:T]]
    
    X_test=X_test[:,sorted_features[:T]]
    
    names = features[sorted_features[:T]]
    # %%istogramma tutte
    
    
    f,ax=plt.subplots(1)
    ax.hist(selector.pvalues_,25,density=True)
    ax.set_title('p-value, full dataset')
    ax.set_xlabel('p-value')
    ax.set_ylabel('Probability density')
    #istogramma best
    f,ax=plt.subplots(1)
    ax.hist(selector.pvalues_[sorted_features[:T]],15,density=True)
    ax.set_title('p-value, best {}'.format(T))
    ax.set_xlabel('p-value')
    ax.set_ylabel('Probability density')
    
    # %%histogram best 8 features
    f,axex=plt.subplots(4,2,figsize=(15,10))
    for i, ax in enumerate(axex.flat):
        sel=sorted_features[i]
        ax.hist(X[y==1,sel],density=True,histtype='bar',alpha=0.4,label='STAS')
        ax.hist(X[y==0,sel],density=True,histtype='bar',alpha=0.4,label='NOSTAS')
        # "{:.2e}".format(12300000)
        ax.set_title("p-value = {:.2e}".format(selector.pvalues_[sel],3))
        ax.set_xlabel(features[sel])
        ax.set_ylabel('Probability density')
        ax.legend()
    f.tight_layout()
    
    # %% matrice di correlazione
    correlation_p=np.corrcoef(X_uni,rowvar=False)
    #names = features[sorted_features[:T]]
    f = plt.figure(figsize=(2*T, 2*T))
    plt.matshow(correlation_p, fignum=f.number)
    plt.xticks(range(len(names)),names, fontsize=4*T, rotation=90)
    plt.yticks(range(len(names)),names, fontsize=4*T)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=6*T)
    ax.xaxis.set_ticks_position('top')
    #plt.title('Correlation Matrix', fontsize=16);
    # %%
    X_tmp = X_uni
    #taglio sulla correlazione e replot
    columns = np.full((correlation_p.shape[0],), True, dtype=bool)
    for i in range(correlation_p.shape[0]):
        for j in range(i+1, correlation_p.shape[0]):# dovrebbe contare a partire da destra
            if np.abs(correlation_p[i,j]) >= param['corr_cut']: #taglio anche le corr negative
                if columns[j]:
                   columns[j] = False
                   
    X_tmp = X_tmp[:,columns]
    names = names[columns]# possibile bug taglia  sempre
    
    correlation_p=np.corrcoef(X_tmp,rowvar=False)
    
    f = plt.figure(figsize=(2*T, 2*T))
    plt.matshow(correlation_p, fignum=f.number)
    plt.xticks(range(len(names)),names, fontsize=14, rotation=45)
    plt.yticks(range(len(names)),names, fontsize=14)
    ax.xaxis.set_ticks_position('top')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    #plt.title('Correlation Matrix Post Cut', fontsize=16);
    if param['do_corr_cut']:
        print(''.join(45*['-']))
        print('TAGLIO SULLA CORRELAZIONE\n |corr| > {:.2f} --> Tengo {} Features'.format(param['corr_cut'],np.sum(columns)))
        X_uni = X_tmp
        X_test = X_test[:,columns]
        print(names)
    # %%
    if param['do_SMOTE']:
        print(''.join(45*['-']))
        print('OVERSAMPLING SMOTE\n')
        oversample = ADASYN(n_neighbors=2)
        X_uni, y_uni = oversample.fit_resample(X_uni, y_uni)
    
    # %%
    
    print(''.join(45*['-']))
    print('K-NN\n')
    
    #KNN=KNeighborsClassifier(n_neighbors=3, weights='distance',algorithm='brute',metric='mahalanobis',metric_params={'VI': np.cov(X_uni)})
    #KNN=KNeighborsClassifier(n_neighbors=5, weights='uniform',algorithm='brute')
    KNN=KNeighborsClassifier(n_neighbors=4, weights='uniform',algorithm='brute')
    KNN = KNN.fit(X_uni, y_uni)
    
    K_pred_train = KNN.predict(np.array(X_uni))
    
    print('\nTRAIN')
    print('Accuracy model on train: %.2f ' % (metrics.accuracy_score(K_pred_train, y_uni)))
    print('Precision and Sensibility model on train: %.2f' % (metrics.f1_score(K_pred_train, y_uni)))
    
    
    K_pred_test = KNN.predict(X_test)
    print('\nTEST')
    print('Accuracy model on test: %.2f ' % (metrics.accuracy_score(K_pred_test, y_test)))
    print('Precision and Sensibility model on test: %.2f' % (metrics.f1_score(K_pred_test, y_test)))
    
    K_pred_base = np.ones(y_test.shape)
    print('\nSTUPIDO classificatore con tutti 1 on TEST')
    print('Accuracy base on test: %.2f ' % (metrics.accuracy_score(K_pred_base, y_test)))
    print('Precision and Sensibility base on test: %.2f' % (metrics.f1_score(K_pred_base, y_test)))
    
    acc_before_lasso[k]=metrics.accuracy_score(K_pred_test, y_test)
    #matrici di confusione
    
    confusion_matrix( y_test,K_pred_base)/len( y_test)
    #questo è con il pacchetto
    #disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     #display_labels=class_names,
                                     #cmap=plt.cm.Blues,
                                     #normalize=normalize)
    #disp.ax_.set_title(title)
    
    # %% matrice di confusione
    #f,ax =subplot(3,1)
    
    for tit,cl,y_i in zip(['Banale','test','train'],[K_pred_base,K_pred_test,K_pred_train],[y_test,y_test,y_uni]):
        confusion_m=confusion_matrix( y_i,cl,normalize='pred')
        df_cm = df_cm = pd.DataFrame(confusion_m, index=["NOSTAS","STAS"], columns=["NOSTAS","STAS"])
        plt.figure(figsize=(5,5))
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}).set_title(tit) # 
      
    
    # %%Esempio di cross validation
    print('\n\nCROSS VALIDATION with lasso for feature selection\n\n')
    y_uni=y_uni.astype('int16')
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    auc=np.zeros([5])
    acc=np.zeros([5])
    pesi=np.zeros([len(names),5])
    prev=np.zeros([5])#prevalenza di stas
    for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
        
        
        model = LogisticRegression(C=5,penalty='l1',class_weight='balanced',solver='liblinear')
        model.fit(X_uni[train,:], y_uni[train])
        fpr, tpr, thresholds = metrics.roc_curve(y_uni[test],model.predict(X_uni[test,:]))
        acc[i]=metrics.accuracy_score(y_uni[test],model.predict(X_uni[test,:]))
        auc[i]=metrics.auc(fpr, tpr)
        print('K= {} |AUC -  {:.2f}   |   ACC -  {:.2f}'.format(
            i,auc[i], acc[i]))
    
        pesi[:,i]=(np.round(model.coef_/np.max(np.abs(model.coef_)),2))
        prev[i]=np.sum(y_uni[test]==1)/len(y_uni[test])
    
    print('\nMean AUC: {:.2f} +/- {:.2f}   |  Mean ACC: {:.2f} +/- {:.2f}'.format(
            auc.mean(),auc.std(), acc.mean(),acc.std()))
    # %%stampo tabella dei pesi
    list2=['ACC: '+el+'%' for el in list(map(str, np.round(acc*100).astype('int16')))]
    fig, axs =plt.subplots(1,figsize=(8, 4))
    axs.axis('tight')
    axs.axis('off')
    ytable=axs.table(cellText=pesi, cellColours=None, cellLoc='right', colWidths=None, rowLabels=names, rowColours=None, rowLoc='left', colLabels=list2, colColours=None, colLoc='center', loc='bottom', bbox=None, edges='closed')
    
    ytable.set_fontsize(34)
    ytable.scale(1, 4)
    
    # %%
    model.fit(X_uni, y_uni)
    acc_lasso_test=metrics.accuracy_score(y_test,model.predict(X_test))
    
    
    print('\n\nLASSO TEST')
    print('Accuracy LASSO on test: %.2f ' % (acc_lasso_test))
    
    print('\nLASSO dropped {} feature/s '.format(np.sum(acc<prev)))
    
    # %% selectio based on lasso scores? (non si fa così)
    if (np.sum(acc<prev)>0 & param['lasso_cut']):
        selected_over_lasso = abs(np.average(pesi,axis=1, weights=((acc>prev)*acc)+1e-3))>0.15
        X_uni=X_uni[:,selected_over_lasso]
        X_test=X_test[:,selected_over_lasso]
        names=names[selected_over_lasso]
    
        print('Selected Features for TEST :\n {}'.format(names))
    
    
        model.fit(X_uni, y_uni)
        acc_lasso_test=metrics.accuracy_score(y_test,model.predict(X_test))
    
    
        print('\n\nLASSO TEST su spazio ristretto')
        print('Accuracy LASSO on test: %.2f ' % (acc_lasso_test))
    
    
    # %%
        print('\n\nRipeto K-NN su spazio ristretto')
        #KNN=KNeighborsClassifier(n_neighbors=3, weights='uniform',algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(X_uni)})
        KNN=KNeighborsClassifier(n_neighbors=4, weights='uniform',algorithm='brute')
        
        KNN = KNN.fit(X_uni, y_uni)
        
        K_pred_train = KNN.predict(np.array(X_uni))
        
        print('\nTRAIN')
        print('Accuracy model on train: %.2f ' % (metrics.accuracy_score(K_pred_train, y_uni)))
        print('Precision and Sensibility model on train: %.2f' % (metrics.f1_score(K_pred_train, y_uni)))
        
        
        K_pred_test = KNN.predict(X_test)
        print('\nTEST')
        print('Accuracy model on test: %.2f ' % (metrics.accuracy_score(K_pred_test, y_test)))
        print('Precision and Sensibility model on test: %.2f' % (metrics.f1_score(K_pred_test, y_test)))
        
        K_pred_base = np.ones(y_test.shape)
        print('\nSTUPIDO classificatore con tutti 1 on TEST')
        print('Accuracy base on test: %.2f ' % (metrics.accuracy_score(K_pred_base, y_test)))
        print('Precision and Sensibility base on test: %.2f' % (metrics.f1_score(K_pred_base, y_test)))
        
        
        acc_after_lasso[k]=metrics.accuracy_score(K_pred_test, y_test)
        
    # %%   
        
        for tit,cl,y_i in zip(['Banale','test','train'],[K_pred_base,K_pred_test,K_pred_train],[y_test,y_test,y_uni]):
            confusion_m=confusion_matrix( y_i,cl,normalize='pred')
            df_cm = df_cm = pd.DataFrame(confusion_m, index=["NOSTAS","STAS"], columns=["NOSTAS","STAS"])
            plt.figure(figsize=(5,5))
            sns.set(font_scale=1.4) # for label size
            sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}).set_title('red. Space'+tit) # 
        
    elif np.sum(acc<prev)==0:
        print('LASSO DID NOT FIND FEATURES TO DROP')
        acc_after_lasso[k]=acc_before_lasso[j]

    plt.close()
    del(X_train,y_train,X_test,y_test)
    
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
