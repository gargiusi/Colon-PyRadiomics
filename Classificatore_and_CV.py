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

from utils import print_confusion_matrix,print_corr_matrix,print_hist_rack,print_model_report
from analisi import find_correlated,corr_matrix,univariate_selection
from sklearn import svm

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from collections import Counter
from sklearn.decomposition import PCA
#parametri (in futuro da riga di comando)
# %%
# param = { #funzionano
#     'outer_split':5,
#     'stratify':True,#not in use
#     'N_features': 25, 
#     'corr_cut':0.85, 
#     'do_scaler': True,
#     'do_corr_cut': True,
#     'do_SMOTE': False,
#     'shuffle_labels': False,
#     'lasso_cut': True,
#     'print_fig': False,
#     'corr_type': 'spearman', #can also be "pearson"
#     'remove_out':True, # removing outliers before correlation calculation
#     'univar':'f_classif', #or custom
#     'PCA_denoise':True,
#     }  
param = {
    'outer_split':5,
    'stratify':True,#not in use
    'N_features': 35, 
    'corr_cut':0.97, 
    'do_scaler': True,
    'do_corr_cut': True,
    'do_SMOTE': False,
    'shuffle_labels': False,
    'lasso_cut': True,
    'print_fig': False,
    'corr_type': 'pearson', #can also be "pearson"
    'remove_out':False, # removing outliers before correlation calculation
    'univar':'f_classif', #or custom
    'PCA_denoise':False,
    'cut_mean_corr_first':True
    }  

kernel = 1.5 * RBF(0.5)
#model = GaussianProcessClassifier(kernel=kernel,random_state=10)
#model=LogisticRegression(C=1,penalty='l2',class_weight='balanced',solver='liblinear')
model=svm.SVC(class_weight = 'balanced',kernel='rbf',degree=3)
#model_ssfs = svm.SVC(class_weight = 'balanced',kernel='linear')
model_ssfs = LogisticRegression(C=0.2,penalty='l1',class_weight='balanced',solver='liblinear')

#model=svm.SVC(class_weight = 'balanced',kernel='linear')
#model=KNeighborsClassifier(n_neighbors=3, weights='distance',algorithm='brute',metric='mahalanobis',metric_params={'VI': np.cov(X_uni)})
#model=KNeighborsClassifier(n_neighbors=3, weights='distance',algorithm='brute')
# %% prepara training e test set del dataset
Filename='/media/andrea/DATA/STAS/rachele/codice_tesi/Tesi//10_STAS.csv' 
data=np.loadtxt(Filename,delimiter=';',skiprows=1)
data2=np.loadtxt(Filename,delimiter=';',skiprows=0,dtype=str)
features=data2[0,1:-1]
X=data[:,1:-1] #Data=Features
y=data[:,-1] #Target=STAS






if param['shuffle_labels']:
    y=np.random.permutation(y)

out_k=param['outer_split']
out_nr=50
#skf = StratifiedKFold(n_splits=out_k,shuffle=True)
#skf = KFold(n_splits=out_k,shuffle=True)
skf = RepeatedStratifiedKFold(n_splits=out_k,n_repeats=out_nr)

#auc_total=np.zeros([4])
acc_before_lasso=np.zeros([out_k*out_nr])
acc_after_lasso=np.zeros([out_k*out_nr])
acc_lasso_test_before=np.zeros([out_k*out_nr])
acc_lasso_test_after=np.zeros([out_k*out_nr])
selected_feature_out=[]
f_used_before=np.zeros([out_k*out_nr])
f_used_after=np.zeros([out_k*out_nr])
# %%
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
    
    T=param['N_features'] #iNSERISCI IL NUMERO DI FEATURES MESSE IN GIOCO
    p_values=univariate_selection(X_uni,y_uni,param)
    sorted_features=np.argsort(p_values)
    # selector = SelectKBest(f_classif, k=T)

    # selector.fit(X_uni, y_uni)
    
    # scores = -np.log10(selector.pvalues_)
    # scores /= scores.max()
    # sorted_features=np.argsort(selector.pvalues_)
    
    
    print(features[sorted_features[:T]])
    
    # %%trasformo e ordino - lo faccio a mano perchè così sono ordinati
    print(''.join(45*['-']))
    print('SELEZIONO {} Features'.format(T))
    X_uni=X_uni[:,sorted_features[:T]]
    
    X_test=X_test[:,sorted_features[:T]]
    
    names = features[sorted_features[:T]]
   
    #denoiser
    if param['PCA_denoise']:#denoiser
    
        pca = PCA(n_components=round(0.70*T) )
        pca.fit(X_uni)
        X_uni = pca.transform(X_uni)
        X_test = pca.transform(X_test)
        
        X_uni = pca.inverse_transform(X_uni)
        X_test = pca.inverse_transform(X_test)
   
    
   
    # %%istogramma tutte
    
    if param['print_fig']:
        f,ax=plt.subplots(1)
        ax.hist(p_values,25,density=True)
        ax.set_title('p-value, full dataset')
        ax.set_xlabel('p-value')
        ax.set_ylabel('Probability density')
        #istogramma best
        f,ax=plt.subplots(1)
        ax.hist(p_values[sorted_features[:T]],15,density=True)
        ax.set_title('p-value, best {}'.format(T))
        ax.set_xlabel('p-value')
        ax.set_ylabel('Probability density')
    
    # %%histogram best 8 features
    if param['print_fig']:#param['print_fig']:
        print_hist_rack(X,y,features,sorted_features,p_values)
        
    
    # %% matrice di correlazione
    correlation_p=corr_matrix(X_uni,param)
   
    if param['print_fig']:
        print_corr_matrix(correlation_p,names)
   
    
    #find correleted and generate new correlation matrix
    columns = find_correlated(correlation_p,param)                
    X_tmp = X_uni[:,columns]
    correlation_p=corr_matrix(X_tmp,param)
    
    if param['print_fig']:
        print_corr_matrix(correlation_p,names[columns])
        
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
        oversample = SMOTE(k_neighbors=3)
        X_uni, y_uni = oversample.fit_resample(X_uni, y_uni)
    
    # %%
    
    print(''.join(45*['-']))
    print('MODEL BEFORE SECOND STEP FS\n')

    m_before = model.fit(X_uni, y_uni)
    
    acc_before_lasso[k]=print_model_report(m_before,X_uni,X_test,y_uni,y_test,param,'Before second step ')
    
    

    
    # %%Esempio di cross validation
    print('\n\nCROSS VALIDATION with lasso for feature selection\n\n')
    y_uni=y_uni.astype('int16')
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    auc=np.zeros([5])
    acc=np.zeros([5])
    pesi=np.zeros([len(names),5])
    prev=np.zeros([5])#prevalenza di stas
    
    for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
        
        
        #model = LogisticRegression(C=3,penalty='l1',class_weight='balanced',solver='liblinear')
        model_ssfs.fit(X_uni[train,:], y_uni[train])
        fpr, tpr, thresholds = metrics.roc_curve(y_uni[test],model_ssfs.predict(X_uni[test,:]))
        acc[i]=metrics.accuracy_score(y_uni[test],model_ssfs.predict(X_uni[test,:]))
        auc[i]=metrics.auc(fpr, tpr)
        print('K= {} |AUC -  {:.2f}   |   ACC -  {:.2f}'.format(
            i,auc[i], acc[i]))
    
        pesi[:,i]=(np.round(model_ssfs.coef_/np.max(np.abs(model_ssfs.coef_)),2))
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
    f_used_before[k]=len(names)
    model_ssfs.fit(X_uni, y_uni)
    acc_lasso_test_before[k]=metrics.accuracy_score(y_test,model_ssfs.predict(X_test))
    
    
    print('\n\nLASSO TEST')
    print('Accuracy LASSO on test: %.2f ' % (acc_lasso_test_before[k]))
    
    print('\nLASSO dropped {} feature/s '.format(np.sum(acc<prev)))
    
    # %% selectio based on lasso scores? (non si fa così)
    if (np.sum(acc<prev)>0 & param['lasso_cut']):
        selected_over_lasso = abs(np.average(pesi,axis=1, weights=((acc>prev)*acc)+1e-3))>0.15
        X_uni=X_uni[:,selected_over_lasso]
        X_test=X_test[:,selected_over_lasso]
        names=names[selected_over_lasso]
        f_used_after[k]=len(names)
        print('Selected Features for TEST :\n {}'.format(names))
    
    
        model_ssfs.fit(X_uni, y_uni)
        acc_lasso_test_after[k]=metrics.accuracy_score(y_test,model_ssfs.predict(X_test))
    
    
        print('\n\nLASSO TEST su spazio ristretto')
        print('Accuracy LASSO on test: %.2f ' % (acc_lasso_test_after[k]))
    
    
    # %%
        print('\n\nRipeto MODEL su spazio ristretto')
        
        m_after = model.fit(X_uni, y_uni)
        acc_after_lasso[k]=print_model_report(m_after,X_uni,X_test,y_uni,y_test,param,'After second step ')
        
        
        
    # %%   
        
    elif np.sum(acc<prev)==0:
        print('LASSO DID NOT FIND FEATURES TO DROP')
        acc_after_lasso[k]=acc_before_lasso[k]
        acc_lasso_test_after[k]=acc_lasso_test_before[k]
        f_used_after[k]=f_used_before[k]
    #plt.close()
    selected_feature_out.append(names)
    del(X_train,y_train,X_test,y_test)
# %%  
print('\nLASSO MODEL')
print('\nMean ACC before ssfs: {:.2f} +/- {:.2f}'.format(acc_lasso_test_before.mean(),acc_lasso_test_before.std()))
print('\nMean ACC after ssfs: {:.2f} +/- {:.2f}'.format(acc_lasso_test_after.mean(),acc_lasso_test_after.std()))
t,p=stats.ttest_rel(acc_lasso_test_before,acc_lasso_test_after)
print('\n paired-t-test (ipotesis of equal average)\n if p-value:({:.2}) > 0.05, the average is the same'.format(p))

print('\nCHOOSED MODEL')
print('\nMean ACC before lasso: {:.2f} +/- {:.2f}'.format(acc_before_lasso.mean(),acc_before_lasso.std()))
print('\nMean ACC after lasso: {:.2f} +/- {:.2f}'.format(acc_after_lasso.mean(),acc_after_lasso.std()))
t,p=stats.ttest_rel(acc_before_lasso,acc_after_lasso)
print('\n paired-t-test (ipotesis of equal average)\n if p-value:({:.2}) > 0.05, the average is the same'.format(p))

flat_list = [item for sublist in selected_feature_out for item in sublist]

sel=Counter(flat_list).most_common(8)
print(sel)

#per Rachele: altre analisi

# np.argmax(acc_after_lasso) il miglior classificatore

# le feature usate dal migliore selected_feature_out[np.argmax(acc_after_lasso)]

#le feature medie selezionate f_used_after.mean()



#check delle feature usate quando fuziona male rispetto a quando funziona bene?

#a=np.array(selected_feature_out, dtype=object)[acc_after_lasso>0.7]

# =============================================================================
