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
from sklearn import metrics
#from scipy import stats

def print_confusion_matrix(preds,gt,title_prefix=''):
    f,axex=plt.subplots(1,3,figsize=(15,5))
    for i,tit,cl,y in zip(range(3),['Banale','test','train'],preds,gt):
        confusion_m=confusion_matrix( y,cl,normalize='pred')
        df_cm = df_cm = pd.DataFrame(confusion_m, index=["NOSTAS","STAS"], columns=["NOSTAS","STAS"])
        
        with sns.plotting_context(font_scale=1.5):
            sns.heatmap(data=df_cm, ax=axex[i],annot=True, annot_kws={"size": 16},vmin=0,vmax=1).set_title(title_prefix+tit) # 
    #f.clf()
    
    
def print_corr_matrix(correlation_p,names):
        T=correlation_p.shape[0]
        f = plt.figure(figsize=(2*T, 2*T))
        ax = f.add_subplot(1, 1, 1)
        plt.matshow(correlation_p, fignum=f.number,vmin=-1, vmax=1)
        plt.xticks(range(len(names)),names, fontsize=4*T, rotation=90)
        plt.yticks(range(len(names)),names, fontsize=4*T)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6*T)
        ax.xaxis.set_ticks_position('top')
        #f.clf()
        
        
def print_hist_rack(X,y,features,sorted_features,pval):
    f,axex=plt.subplots(4,2,figsize=(15,10))
    for i, ax in enumerate(axex.flat):
            sel=sorted_features[i]
            (n, bins, patches) = ax.hist(X[:,sel],density=True,histtype='step' ,alpha=0.5,label='ALL')
            ax.hist(X[y==1,sel],bins=bins,density=True,histtype='bar',alpha=0.4,label='STAS')
            ax.hist(X[y==0,sel],bins=bins,density=True,histtype='bar',alpha=0.4,label='NOSTAS')
            # "{:.2e}".format(12300000)
            ax.set_title("p-value = {:.2e}".format(pval[sel],3))
            ax.set_xlabel(features[sel])
            ax.set_ylabel('Probability density')
            ax.set_ylim(bottom=0,top=2*max(n))
            #ax.legend()
    f.tight_layout()       
    
  # __________----------------------____________________--------------------___________________------------------_____--      


def print_model_report(model,X_train,X_test,y_train,y_test,param,title=''):
    K_pred_train = model.predict(np.array(X_train))
    K_pred_test = model.predict(X_test)
    K_pred_base = np.ones(y_test.shape)
    
    print('\nTRAIN')
    print('Accuracy model on train: %.2f ' % (metrics.accuracy_score(K_pred_train, y_train)))
    print('Precision and Sensibility model on train: %.2f' % (metrics.f1_score(K_pred_train, y_train)))
    
    
    print('\nTEST')
    print('Accuracy model on test: %.2f ' % (metrics.accuracy_score(K_pred_test, y_test)))
    print('Precision and Sensibility model on test: %.2f' % (metrics.f1_score(K_pred_test, y_test)))
    
    
    print('\nSTUPIDO classificatore con tutti 1 on TEST')
    print('Accuracy base on test: %.2f ' % (metrics.accuracy_score(K_pred_base, y_test)))
    print('Precision and Sensibility base on test: %.2f' % (metrics.f1_score(K_pred_base, y_test)))
    
    if param['print_fig']:
        print_confusion_matrix([K_pred_base,K_pred_test,K_pred_train],[y_test,y_test,y_train],title_prefix=title)
    
    return metrics.accuracy_score(K_pred_test, y_test)






