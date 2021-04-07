#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:31:27 2021

@author: andrea
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy import stats

def print_confusion_matrix(preds,gt,title_prefix=''):
    f,axex=plt.subplots(1,3,figsize=(15,5))
    for i,tit,cl,y in zip(range(3),['Banale','test','train'],preds,gt):
        confusion_m=confusion_matrix( y,cl,normalize='pred')
        df_cm = df_cm = pd.DataFrame(confusion_m, index=["NOSTAS","STAS"], columns=["NOSTAS","STAS"])
        
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(df_cm, ax=axex[i],annot=True, annot_kws={"size": 16},vmin=0,vmax=1).set_title(title_prefix+tit) # 
 
    
    
def print_corr_matrix(correlation_p,T,names):
        f = plt.figure(figsize=(2*T, 2*T))
        ax = f.add_subplot(1, 1, 1)
        plt.matshow(correlation_p, fignum=f.number,vmin=-1, vmax=1)
        plt.xticks(range(len(names)),names, fontsize=4*T, rotation=90)
        plt.yticks(range(len(names)),names, fontsize=4*T)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6*T)
        ax.xaxis.set_ticks_position('top')
        
        
def print_hist_rack(X,y,features,sorted_features,pval):
    f,axex=plt.subplots(4,2,figsize=(15,10))
    for i, ax in enumerate(axex.flat):
            sel=sorted_features[i]
            (n, bins, patches) = ax.hist(X[y==1,sel],density=True,histtype='bar',alpha=0.4,label='STAS')
            ax.hist(X[y==0,sel],bins=bins,density=True,histtype='bar',alpha=0.4,label='NOSTAS')
            # "{:.2e}".format(12300000)
            ax.set_title("p-value = {:.2e}".format(pval[sel],3))
            ax.set_xlabel(features[sel])
            ax.set_ylabel('Probability density')
            ax.legend()
    f.tight_layout()       
    
  # __________----------------------____________________--------------------___________________------------------_____--      
def find_correlated(correlation_p,param):
    columns = np.full((correlation_p.shape[0],), True, dtype=bool)
    for i in range(correlation_p.shape[0]):
        for j in range(i+1, correlation_p.shape[0]):# dovrebbe contare a partire da destra
            if np.abs(correlation_p[i,j]) >= param['corr_cut']: #taglio anche le corr negative
                if columns[j]:
                   columns[j] = False
    return columns



def corr_matrix(X,param):
    
    if param['remove_out']:
        for ind in range (0,X.shape[1]):

            z_scores = stats.zscore(X[:,ind])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores > 2.7)

            X[filtered_entries,ind]=X[np.invert(filtered_entries),ind].mean()
        
    
    
    
    if param['corr_type']=='pearson':
        correlation_p=np.corrcoef(X,rowvar=False)
    elif param['corr_type']=='spearman':
        correlation_p=stats.spearmanr(X,axis=0)
        correlation_p=correlation_p.correlation
    elif param['corr_type']=='mixed':
        
        correlation_p = np.zeros([X.shape[1],X.shape[1]])
        for i in range(correlation_p.shape[0]):
            s,p1 = stats.shapiro(X[:,i])
            for j in range(i+1, correlation_p.shape[0]):# dovrebbe contare a partire da destra
                s,p2 = stats.shapiro(X[:,i])
                if ((p1 > 0.05) & (p2>0.05)): #taglio anche le corr negative
                    correlation_p[i,j]=np.corrcoef(X[:,i],X[:,j])[0,1]
                    
                else:
                    correlation_p[i,j]=stats.spearmanr(X[:,4],X[:,7],axis=0).correlation
                
                correlation_p[j,i]=correlation_p[i,j]
        
        
    else:
        print('No correlation type detected')
        return 0
            

    return correlation_p










