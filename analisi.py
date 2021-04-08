#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:52 2021

@author: andrea
"""
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest,f_classif

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
            for j in range(i+1, correlation_p.shape[0]):# 
                s,p2 = stats.shapiro(X[:,i])
                if ((p1 > 0.05) & (p2>0.05)): #normality test
                    correlation_p[i,j]=np.corrcoef(X[:,i],X[:,j])[0,1]
                    
                else:
                    correlation_p[i,j]=stats.spearmanr(X[:,4],X[:,7],axis=0).correlation
                
                correlation_p[j,i]=correlation_p[i,j]
        
        
    else:
        print('No correlation type detected')
        return 0
            

    return correlation_p

def univariate_selection(X,y,param):
    
    if param['remove_out']:
        for ind in range (0,X.shape[1]):

            z_scores = stats.zscore(X[:,ind])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores > 2.7)

            X[filtered_entries,ind]=X[np.invert(filtered_entries),ind].mean()
    
    if param['univar']=='custom':
    
        p_values=np.ones(X.shape[1])
        for i in range(X.shape[1]):
        
        #p_values[i]=stats.mannwhitneyu(X[y==1,i],X[y==0,i]).pvalue
            p_values[i]=stats.kruskal(X[y==1,i],X[y==0,i]).pvalue
        return p_values    
    
    else:
        
        selector = SelectKBest(f_classif)

        selector.fit(X, y)
        
        return selector.pvalues_
    