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
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from utils import print_confusion_matrix,print_corr_matrix,find_correlated,corr_matrix,print_hist_rack
from sklearn import svm
#parametri (in futuro da riga di comando)
# %%
param = {
    'outer_split':4,
    'stratify':True,#not in use
    'N_features': 25, 
    'corr_cut':0.80, 
    'do_scaler': True,
    'do_corr_cut': True,
    'do_SMOTE': False,
    'shuffle_labels': False,
    'lasso_cut': True,
    'print_fig': False,
    'corr_type': 'spearman', #can also be "pearson"
    'remove_out':True, # removing outliers before correlation calculation
    
    }  





# %%prepara training e test set del dataset
Filename='/media/andrea/DATA/STAS/rachele/codice_tesi/Tesi//10_STAS.csv' ###################### Enter the full path of csv dataset
data=np.loadtxt(Filename,delimiter=';',skiprows=1)
data2=np.loadtxt(Filename,delimiter=';',skiprows=0,dtype=str)
features=data2[0,1:-16]
X=data[:,1:-16] #Data=Features
y=data[:,-1] #Target=STAS
if param['shuffle_labels']:
    y=np.random.permutation(y)

out_k=param['outer_split']
skf = StratifiedKFold(n_splits=out_k,shuffle=True)
#auc_total=np.zeros([4])
acc_before_lasso=np.zeros([out_k])
acc_after_lasso=np.zeros([out_k])
selected_feature_out=[]

for k, [train_out, test_out] in enumerate(skf.split(X, y)):
    print('\n\tFOLD = {}\n'.format(k))
    
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
    if param['print_fig']:
        print_hist_rack(X,y,features,sorted_features,selector.pvalues_)
        
    
    # %% matrice di correlazione
    correlation_p=corr_matrix(X_uni,param)
   
    if param['print_fig']:
        print_corr_matrix(correlation_p,T,names)
   
    
    
    columns = find_correlated(correlation_p,param)
                   
    X_tmp = X_uni[:,columns]
    
    
    correlation_p=corr_matrix(X_tmp,param)
    if param['print_fig']:
        print_corr_matrix(correlation_p,T,names[columns])
        
    if param['do_corr_cut']:
        print(''.join(45*['-']))
        print('TAGLIO SULLA CORRELAZIONE\n |corr| > {:.2f} --> Tengo {} Features'.format(param['corr_cut'],np.sum(columns)))
        X_uni = X_tmp
        X_test = X_test[:,columns]
        names = names[columns]
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
    #KNN=KNeighborsClassifier(n_neighbors=4, weights='uniform',algorithm='brute')
    KNN=svm.SVC(class_weight = 'balanced')
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
    
   
    
    # %% matrice di confusione
    #f,ax =subplot(3,1)
    if param['print_fig']:
        print_confusion_matrix([K_pred_base,K_pred_test,K_pred_train],[y_test,y_test,y_uni],title_prefix='before lasso')
          
    
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
    if param['print_fig']:
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
        #KNN=KNeighborsClassifier(n_neighbors=3, weights='distance',algorithm='brute')
        #KNN=KNeighborsClassifier(n_neighbors=3, weights='distance',algorithm='brute',metric='mahalanobis',metric_params={'VI': np.cov(X_uni)})
        #clf = ExtraTreesClassifier(n_estimators=20, random_state=0,class_weight='balanced')
        #KNN = clf.fit(X_uni, y_uni)
        KNN=svm.SVC(class_weight = 'balanced')
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
        if param['print_fig']:
            print('entro e non printo?')
            print_confusion_matrix([K_pred_base,K_pred_test,K_pred_train],[y_test,y_test,y_uni],title_prefix='After LASSO ')

    elif np.sum(acc<prev)==0:
        print('LASSO DID NOT FIND FEATURES TO DROP')
        acc_after_lasso[k]=acc_before_lasso[k]

    plt.close()
    del(X_train,y_train,X_test,y_test)

print('\nMean ACC before lasso: {:.2f} +/- {:.2f}'.format(acc_before_lasso.mean(),acc_before_lasso.std()))
print('\nMean ACC after lasso: {:.2f} +/- {:.2f}'.format(acc_after_lasso.mean(),acc_after_lasso.std()))
t,p=stats.ttest_rel(acc_before_lasso,acc_after_lasso)
print('\n paired-t-test (ipotesis of equal average)\n if p-value:({:.2}) > 0.05, the average is the same'.format(p))


# =============================================================================