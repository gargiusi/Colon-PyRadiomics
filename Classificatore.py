from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFromModel, SelectPercentile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics, linear_model,svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report
from sklearn.linear_model import LassoCV, LinearRegression, Lasso
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import _random_over_sampler, BorderlineSMOTE, SMOTENC, SMOTE, RandomOverSampler
from sklearn.linear_model import LogisticRegression, LassoCV
from collections import Counter
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
import xlwt
import os, sys, re
import tkinter
from tkinter import filedialog, messagebox
from ast import literal_eval
from matplotlib import cm as cm
import warnings
from scipy import stats

warnings.simplefilter('ignore')


try:
    root = tkinter.Tk()
    root.withdraw()
    currdir = os.getcwd()
    pathname = filedialog.askdirectory(parent=root, initialdir=currdir, title='Select the Directory of all the Set Data')
    if os.path.isdir(pathname) != True:
        exit()

    # parametri
    PercorsoParam = pathname + '/Parametri.txt'
    TypeParam = np.loadtxt(PercorsoParam, delimiter='|', skiprows=0, dtype=str)[0, :]
    ItemParam = np.loadtxt(PercorsoParam, delimiter='|', skiprows=0, dtype=str)[1, :]
    ValueParam = np.loadtxt(PercorsoParam, delimiter='|', skiprows=0, dtype=str)[2, :]

    param = {}
    for n in range(0, ItemParam.shape[0]):
        if TypeParam[n] == "string":
            param[ItemParam[n]] = ValueParam[n]
        if TypeParam[n] == "float":
            param[ItemParam[n]] = float(ValueParam[n])
        if TypeParam[n] == "boolean":
            param[ItemParam[n]] = bool(ValueParam[n])
        if TypeParam[n] == "integer":
            param[ItemParam[n]] = int(ValueParam[n])


    List = os.listdir(pathname)

    #ExtractImmagini = messagebox.askquestion("Extract Image", "You want to save all Images?")
    #if ExtractImmagini == 'yes':
    if param['print_fig']:
        Richiesta_Immagini = True
    else:
        Richiesta_Immagini = False


    for C in List:

        if str(C) != "Parametri.txt" and str(C) != "Image":
            NomeFileDataSet=str(C)
            workbook = xlwt.Workbook()
            if Richiesta_Immagini==True:
                Esistente = False
                Esistente3 = False
                for tt in List:
                    if tt == 'Image':
                        Esistente = True
                        for ttt in os.listdir(pathname+'/Image'):
                            if ttt == str(C):
                                Esistente3 = True
                if Esistente == False:
                    os.mkdir(pathname + '/Image')
                if Esistente3 == False:
                    os.mkdir(pathname + '/Image/'+str(C))
                PercorsoImmagini_1 = pathname + '/Image/' + str(C)
            PercorsoCSV = pathname + '/' + str(C)
        if str(C) != "Parametri.txt" and str(C) != "Image":
            # prepara training e test set del dataset
            Filename = PercorsoCSV
            data = np.loadtxt(Filename, delimiter=';', skiprows=1)
            data2 = np.loadtxt(Filename, delimiter=';', skiprows=0, dtype=str)
            features = data2[0, 1:-16]
            X = data[:, 1:-16]  # Data=Features
            y = data[:, -1]  # Target=STAS

            if param['shuffle_labels']:
                y = np.random.permutation(y)

            out_k = param["outer_split"]
            out_nr = param["outer_split_repeat"]
            kfold = RepeatedStratifiedKFold ( n_splits = out_k , n_repeats = out_nr )


            Modelli = np.zeros([4])
            Modelli=['KNN','SVC','NB','RF']
            acc_after_lasso_classificatori= np.zeros([len(Modelli)])
            acc_after_lasso_classificatori_errore= np.zeros([len(Modelli)])
            f_used_after_classificatori= np.zeros([len(Modelli)])
            selected_feature_out_classificatori_value= np.empty((len(Modelli),8), dtype='object')
            selected_feature_out_classificatori_count = np.empty((len(Modelli), 8), dtype='object')


            parametri = np.array(['Accuracy Before Lasso', 'Accuracy After Lasso', 'Accuracy Before Lasso Test',
                         'Accuracy After Lasso Test', 'Feature used Before', 'Feature used After'],dtype='object')

            for tr in range(0,len(Modelli)):

                acc_before_lasso = np.zeros([out_k * out_nr])
                acc_after_lasso = np.zeros([out_k * out_nr])
                acc_lasso_test_before = np.zeros([out_k * out_nr])
                acc_lasso_test_after = np.zeros([out_k * out_nr])
                f_used_before = np.zeros([out_k * out_nr])
                f_used_after = np.zeros([out_k * out_nr])
                selected_feature_out = []
                JL=1


                for k, [train_out, test_out] in enumerate(kfold.split(X, y)):
                    print('\n\t                  FOLD = {}'.format(k))
                    print('                  ------------------\n')
                    X_train = X[train_out, :]
                    y_train = y[train_out]
                    X_test = X[test_out, :]
                    y_test = y[test_out]
                    # Scale the X data (per garantire che nessuna informazione al di fuori dei dati di addestramento venga utilizzata per creare il modello)
                    if param['do_scaler']:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    if Richiesta_Immagini == True:
                        Esistente22 = False
                        for tt in os.listdir(PercorsoImmagini_1):
                            if tt == Modelli[tr]:
                                Esistente22 = True
                        if Esistente22 == False:
                            os.mkdir('{}/{}'.format(PercorsoImmagini_1,Modelli[tr]))


                        Esistente2 = False
                        for tt in os.listdir(PercorsoImmagini_1+'/'+Modelli[tr]):
                            if tt == 'FOLD ' + str(k+1):
                                Esistente2 = True
                        if Esistente2 == False:
                            os.mkdir('{}/FOLD {}'.format(PercorsoImmagini_1+'/'+Modelli[tr],k+1))
                        PercorsoImmagini_2 = PercorsoImmagini_1 + '/' + Modelli[tr] + '/FOLD ' + str(k+1)+'/'


                    # %% #..................................................................................................................

                    #             Univariate Features Selection
                    # ................................................#
                    # standardizzo
                    X_uni = X_train
                    y_uni = y_train
                    # y_uni=np.random.permutation(y_uni)
                    T = param['N_features']  # iNSERISCI IL NUMERO DI FEATURES MESSE IN GIOCO

                    if param['remove_out']:
                        for ind in range(0, X_uni.figura[1]):
                            z_scores = stats.zscore(X_uni[:, ind])
                            abs_z_scores = np.abs(z_scores)
                            filtered_entries = (abs_z_scores > 3)
                            X[filtered_entries, ind] = X_uni[np.invert(filtered_entries), ind].mean()

                    if param['univar'] == 'custom':
                            p_values = np.array(X_uni.figura[1])
                            for i in range(X_uni.figura[1]):
                                p_values[i] = stats.kruskal(X_uni[y_uni == 1, i], X_uni[y_uni == 0, i])

                    else:
                        selector = SelectKBest(f_classif)
                        selector.fit(X_uni, y_uni)
                        p_values=selector.pvalues_

                    sorted_features = np.argsort(p_values)
                    print(features[sorted_features[:T]])

                    # %%trasformo e ordino - lo faccio a mano perchè così sono ordinati
                    print(''.join(45 * ['-']))
                    print('SELEZIONO {} Features'.format(T))
                    X_uni = X_uni[:, sorted_features[:T]]

                    X_test = X_test[:, sorted_features[:T]]

                    names = features[sorted_features[:T]]

                    if param['PCA_denoise']:  # denoiser

                        pca = PCA(n_components=round(0.70 * T))
                        pca.fit(X_uni)
                        X_uni = pca.transform(X_uni)
                        X_test = pca.transform(X_test)
                        X_uni = pca.inverse_transform(X_uni)
                        X_test = pca.inverse_transform(X_test)


                    # %%istogramma tutte
                    if Richiesta_Immagini == True:
                        plt.figure()
                        plt.hist(p_values, 25, density=True)
                        plt.suptitle('p-value, full dataset')
                        plt.xlabel('p-value')
                        plt.ylabel('Probability density')
                        plt.tight_layout()
                        plt.savefig('{}p value of full dataset.png'.format(PercorsoImmagini_2),bbox_inches ="tight")
                        plt.close()

                    # istogramma best
                    if Richiesta_Immagini == True:
                        plt.figure()
                        plt.hist(p_values[sorted_features[:T]], 15, density=True)
                        plt.suptitle('p-value, best {}'.format(T))
                        plt.xlabel('p-value')
                        plt.ylabel('Probability density')
                        plt.tight_layout()
                        plt.savefig('{}p value of best {}.png'.format(PercorsoImmagini_2,T),bbox_inches ="tight")
                        plt.close()

                    # %%histogram best 8 features
                    if Richiesta_Immagini == True:
                        f, axex = plt.subplots(4, 2, figsize=(15, 10))
                        for i, ax in enumerate(axex.flat):
                            sel = sorted_features[i]
                            (n, bins, patches) = ax.hist(X[:, sel], density=True, histtype='step', alpha=0.5, label='ALL')
                            ax.hist(X[y == 1, sel], density=True, histtype='bar', alpha=0.4, label='STAS')
                            ax.hist(X[y == 0, sel], density=True, histtype='bar', alpha=0.4, label='NOSTAS')
                            ax.set_title("p-value = {:.2e}".format(p_values[sel], 3))
                            ax.set_xlabel(features[sel])
                            ax.set_ylabel('Probability density')
                            ax.legend()
                        f.tight_layout()
                        plt.tight_layout()
                        plt.savefig('{}histogram best 8 features.png'.format(PercorsoImmagini_2),bbox_inches ="tight")
                        plt.close()



                    ### Matrice di correlazione
                    #Rimuovere outlier
                    if param['remove_out']:
                        for ind in range(0, X_uni.figura[1]):
                            z_scores = stats.zscore(X_uni[:, ind])
                            abs_z_scores = np.abs(z_scores)
                            filtered_entries = (abs_z_scores > 3)
                            X_uni[filtered_entries, ind] = X_uni[np.invert(filtered_entries), ind].mean()
                    #Corr_Type
                    if param['corr_type'] == 'pearson':
                        correlation_p = np.corrcoef(X_uni, rowvar=False)
                    elif param['corr_type'] == 'spearman':
                        correlation_p = stats.spearmanr(X_uni, axis=0)
                        correlation_p = correlation_p.correlazione
                    elif param['corr_type'] == 'mixed':
                        correlation_p = np.zeros([X_uni.figura[1], X_uni.figura[1]])
                        for i in range(correlation_p.shape[0]):
                            s, p1 = stats.shapiro(X_uni[:, i])
                            for j in range(i + 1, correlation_p.shape[0]):
                                s, p2 = stats.shapiro(X_uni[:, i])
                                if (p1 > 0.05) and (p2 > 0.05):  # normality test
                                    correlation_p[i, j] = np.corrcoef(X_uni[:, i], X_uni[:, j])[0, 1]

                                else:
                                    correlation_p[i, j] = stats.spearmanr(X_uni[:, 4], X_uni[:, 7], axis=0).correlazione

                                correlation_p[j, i] = correlation_p[i, j]

                    else:
                        print("Nessun tipo di correlazione rilevato")

                    if Richiesta_Immagini == True:
                        plt.figure(figsize=(6, 6))
                        cmap = cm.get_cmap('jet', 30)
                        im=plt.imshow(correlation_p, interpolation="nearest",cmap=cmap)
                        plt.xticks(range(len(names)), names, rotation=90,fontsize=8)
                        plt.yticks(range(len(names)), names,fontsize=8)
                        plt.grid(False)
                        plt.clim(-1, 1)
                        cb=plt.colorbar(im)
                        cb.ax.tick_params(labelsize=8)
                        plt.title('Correlation Matrix')
                        plt.tight_layout()
                        plt.savefig('{}Correlation Matrix.png'.format(PercorsoImmagini_2),bbox_inches ="tight")
                        plt.close()


                    # find correlato e genera una nuova matrice di correlazione
                    X_tmp = X_uni
                    columns = np.full((correlation_p.shape[0],), True, dtype=bool)
                    if param['cut_mean_corr_first']:
                        b = np.argsort(np.mean(np.abs(correlation_p - np.eye(correlation_p.shape[0])), axis=1))[- round(0.15 * correlation_p.shape[0]): - 1]
                        correlation_p[b, :] = 0
                        correlation_p[:, b] = 0
                        columns[b] = False
                    for i in range(correlation_p.shape[0]):
                        for j in range(i, correlation_p.shape[0] - 2):  # il loop parte da destra
                            if np.abs(correlation_p[i, correlation_p.shape[1] - j - 1]) >= param['corr_cut']:  # taglio anche le corr negative
                                if columns[correlation_p.shape[1] - j - 1]:
                                    columns[correlation_p.shape[1] - j - 1] = False

                    X_tmp = X_tmp[:, columns]
                    names = names[columns]  # possibile bug taglia  sempre


                    # Rimuovere outlier
                    if param['remove_out']:
                        for ind in range(0, X_tmp.figura[1]):
                            z_scores = stats.zscore(X_tmp[:, ind])
                            abs_z_scores = np.abs(z_scores)
                            filtered_entries = (abs_z_scores > 3)
                            X_tmp[filtered_entries, ind] = X_tmp[np.invert(filtered_entries), ind].mean()
                    # Corr_Type
                    if param['corr_type'] == 'pearson':
                        correlation_p = np.corrcoef(X_tmp, rowvar=False)
                    elif param['corr_type'] == 'spearman':
                        correlation_p = stats.spearmanr(X_tmp, axis=0)
                        correlation_p = correlation_p.correlazione
                    elif param['corr_type'] == 'mixed':
                        correlation_p = np.zeros([X_tmp.figura[1], X_tmp.figura[1]])
                        for i in range(correlation_p.shape[0]):
                            s, p1 = stats.shapiro(X_tmp[:, i])
                            for j in range(i + 1, correlation_p.shape[0]):
                                s, p2 = stats.shapiro(X_tmp[:, i])
                                if (p1 > 0.05) and (p2 > 0.05):  # normality test
                                    correlation_p[i, j] = np.corrcoef(X_tmp[:, i], X_tmp[:, j])[0, 1]

                                else:
                                    correlation_p[i, j] = stats.spearmanr(X_tmp[:, 4], X_tmp[:, 7], axis=0).correlazione

                                correlation_p[j, i] = correlation_p[i, j]

                    else:
                        print("Nessun tipo di correlazione rilevato")



                    if Richiesta_Immagini == True:
                        plt.figure(figsize=(6, 6))
                        im = plt.imshow(correlation_p, interpolation="nearest", cmap=cmap)
                        plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
                        plt.yticks(range(len(names)), names, fontsize=8)
                        plt.grid(False)
                        plt.clim(-1, 1)
                        cb = plt.colorbar(im)
                        cb.ax.tick_params(labelsize=8)
                        plt.title('Correlation Matrix Post Cut')
                        plt.savefig('{}Correlation Matrix Post Cut.png'.format(PercorsoImmagini_2),bbox_inches ="tight")
                        plt.close()


                    if param['do_corr_cut']:
                        print(''.join(45 * ['-']))
                        print('TAGLIO SULLA CORRELAZIONE\n |corr| > {:.2f} --> Tengo {} Features'.format(param['corr_cut'],
                                                                                                         np.sum(columns)))
                        X_uni = X_tmp
                        X_test = X_test[:, columns]
                        print(names)
                    # %%
                    if param['do_SMOTE']:
                        print(''.join(45 * ['-']))
                        print('OVERSAMPLING SMOTE\n')
                        oversample = SMOTE(k_neighbors=4)
                        X_uni, y_uni = oversample.fit_resample(X_uni, y_uni)

                    if Modelli[tr]=="KNN":
                        # %%Studio per scegliere in modo ottimo il numero di vicini k
                        print('\n\nStudio per scegliere in modo ottimo il numero di vicini k\n\n')
                        y_uni = y_uni.astype('int16')

                        ## studio per scegliere in modo ottimo il numero di vicini k

                        skf = StratifiedKFold(n_splits=5, shuffle=True)
                        auc = np.zeros([5])
                        acc = np.zeros([5])
                        accMedio = np.zeros([5])
                        aucMedio = np.zeros([5])
                        pesi = np.zeros([5, 5])
                        prev = np.zeros([5])  # prevalenza di stas
                        Accuracy = np.zeros([5, 5])
                        Aucuracy = np.zeros([5, 5])
                        for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
                            accuracy = []
                            aucuracy = []
                            for j in range(1, 6):
                                model = KNeighborsClassifier(n_neighbors=j, weights='distance', algorithm='brute',
                                                             metric='mahalanobis',
                                                             metric_params={'VI': np.cov(X_uni)})
                                model.fit(X_uni[train, :], y_uni[train])
                                fpr, tpr, thresholds = metrics.roc_curve(y_uni[test], model.predict(X_uni[test, :]))
                                acc[i] = metrics.accuracy_score(y_uni[test], model.predict(X_uni[test, :]))
                                auc[i] = metrics.auc(fpr, tpr)
                                accuracy.append(acc[i])
                                aucuracy.append(acc[i])
                            Accuracy[:, i] = np.asarray(accuracy).transpose()
                            Aucuracy[:, i] = np.asarray(aucuracy).transpose()
                            accMedio[i] = np.mean(Accuracy[:, i])
                            aucMedio[i] = np.mean(Aucuracy[:, i])
                            print('K= {} |AUC -  {:.2f}   |   ACC -  {:.2f}'.format(i, aucMedio[i], accMedio[i]))
                            del (accuracy, aucuracy)
                            pesi[:, i] = (np.round(Accuracy[:, i] / np.max(np.abs(Accuracy[:, i])), 2))  # ?
                            prev[i] = np.sum(y_uni[test] == 1) / len(y_uni[test])

                        print('\nMean AUC: {:.2f} +/- {:.2f}   |  Mean ACC: {:.2f} +/- {:.2f}\n'.format(
                            aucMedio.mean(), aucMedio.std(), accMedio.mean(), accMedio.std()))
                        # %%stampo tabella dei pesi
                        if Richiesta_Immagini == True:
                            plt.figure(5, figsize=(10, 10))
                            po = ['% Per split n.' + str(eel) for eel in range(1, 6)]
                            list2 = ['ACC: ' + el for el in list(map(str, np.round(accMedio * 100).astype('int16')))]
                            list4 = []
                            for u in range(0, 5):
                                list3 = list2[u] + po[u]
                                list4.append(list3)
                            list2 = np.asarray(list4).transpose()

                            n_neighbors = ['K=' + le for le in list(map(str, range(1, 6)))]
                            values = np.arange(0, 1, 0.1)
                            value_increment = 1
                            # Get some pastel shades for the colors
                            colors = plt.cm.BuPu(np.linspace(0.5, 1.2, len(names)))
                            # colors = ['r', 'g', 'b','y','v']
                            n_rows = len(pesi)
                            index = np.arange(len(list2)) + 0.1
                            bar_width = 0.1
                            # Initialize the vertical-offset for the stacked bar chart.
                            y_offset = np.zeros(len(list2))
                            # Plot bars and create text labels for the table
                            cell_text = []
                            for row in range(n_rows):
                                plt.bar(index, pesi[row], bar_width, bottom=y_offset, color=colors[row])
                                index = index + 0.1

                            the_table = plt.table(cellText=pesi, cellLoc='center', rowLabels=n_neighbors, rowColours=colors,
                                                  rowLoc='center', colLabels=list2, colLoc='center', loc='bottom')
                            # Adjust layout to make room for the table:
                            the_table.scale(1, 4)
                            plt.subplots_adjust(left=0.2, bottom=0.2)
                            plt.ylabel("Peso dell'accuratezza")
                            plt.yticks(values)
                            plt.xticks([])
                            plt.title("Andamento del peso dell'accuratezza per kfold al variare del numero di vicini del KNN",
                                      loc='left')
                            plt.tight_layout()
                            plt.savefig(
                                '{}Andamento del peso dell accuratezza per kfold al variare del numero di vicini del KNN.png'.format(
                                    PercorsoImmagini_2), bbox_inches="tight")
                            plt.close()

                        mediaAritmetica = np.zeros([5])
                        for p in range(0, 5):
                            x = pesi[p, :]
                            mediaAritmetica[p] = x.mean()
                            print('Media Aritmetica per K={} è: {:.2f} +/- {:.2f} '.format(p + 1, mediaAritmetica[p],
                                                                                           x.std()))

                        # Risultato ottimo
                        mediaAritmetica_massima = mediaAritmetica[0]
                        n_neighbors_Ottimo = 5
                        for p in range(0, 5):
                            if mediaAritmetica[p] >= mediaAritmetica_massima:
                                mediaAritmetica_massima = mediaAritmetica[p]
                                n_neighbors_Ottimo = p + 1

                        print('\nIl K neighbors migliore è ={}\n\n'.format(n_neighbors_Ottimo))





                    # %%%%%%%%%%%%%%%        Modello SCELTO before Lasso    %%%%%%%%%%%%%%%%%% #
                    print(''.join(90* ['-']))
                    print('MODELLO {} PRIMA DEL LASSO'.format(Modelli[tr]))
                    print(''.join(35* ['-']))

                    skf = StratifiedKFold(n_splits=5, shuffle=True)
                    auc = np.zeros([5])
                    acc = np.zeros([5])
                    for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
                        print('\n\nKFOLD=%.2f' % (int(i + 1)))
                        if Modelli[tr]=="KNN":
                            if param['mahalanobis_Distance']:
                                model = KNeighborsClassifier(n_neighbors=n_neighbors_Ottimo, weights='uniform',algorithm='brute',metric='mahalanobis', metric_params={'V': np.cov(X_uni)})
                            else:
                                model = KNeighborsClassifier(n_neighbors=n_neighbors_Ottimo, weights='uniform',algorithm='brute')
                        if Modelli[tr] == "SVC":
                            kernel = 1, 5 * RBF(0, 5)
                            model = svm.SVC(class_weight='balanced', kernel='rbf', degree=3)
                        if Modelli[tr] == "RF":
                            model =RandomForestClassifier(n_jobs=2,random_state=0)
                        if Modelli[tr] == "NB":
                            model=GaussianNB()

                        model.fit(X_uni[train, :], y_uni[train])
                        fpr, tpr, thresholds = metrics.roc_curve(y_uni[test], model.predict(X_uni[test, :]))
                        acc[i] = metrics.accuracy_score(y_uni[test], model.predict(X_uni[test, :]))
                        auc[i] = metrics.auc(fpr, tpr)

                        K_pred_base = np.ones(y_uni[test].shape)
                        print('\nSTUPIDO TEST (tutti 1 on TEST):')
                        print('Accuracy: %.2f ' % (metrics.accuracy_score(K_pred_base, y_uni[test])))
                        print('Precision and Sensibility: %.2f' % (metrics.f1_score(K_pred_base, y_uni[test])))

                        print('TEST:')
                        print('AUC -  {:.2f}   |   ACC -  {:.2f}'.format(auc[i], acc[i]))

                        # Disegno curva Roc
                        if Richiesta_Immagini == True:
                            plt.figure()
                            plt.plot(fpr, tpr, color='orange', linestyle='dashed', label='Curva Roc KNN area %0.2f' % auc[i])
                            # Disegno le curve di riferimento
                            plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Modello errato')  # tutto errato
                            plt.plot([0.01, 0.01], [0.01, 1.01], color='green', linestyle=':',
                                     label='Modello ideale')  # tutto corretto
                            plt.plot([0.01, 1.01], [1.01, 1.01], color='green', linestyle=':')  # tutto corretto
                            plt.xlabel('FPR False Positive Rate')
                            plt.ylabel('TPR True Positive Rate')
                            plt.title('Receiver Operating Characteristic (ROC) for kfold split %0.2f' % (int(i + 1)))
                            plt.legend(loc="lower right")
                            plt.savefig('{}Receiver Operating Characteristic (ROC Before Lasso) for kfold split {} ({}).png'.format(PercorsoImmagini_2, int(i + 1),Modelli[tr]), bbox_inches="tight")
                            plt.close()

                        # matrici di confusione
                        confusion_matrix(y_uni[test], K_pred_base) / len(y_uni[test])
                        K_pred_train = model.predict(np.array(X_uni[train, :]))
                        K_pred_test = model.predict(X_uni[test, :])
                        for tit, cl, y_cl in zip(['Banale', 'test', 'train'], [K_pred_base, K_pred_test, K_pred_train],
                                                 [y_uni[test], y_uni[test], y_uni[train]]):
                            confusion_m = confusion_matrix(y_cl, cl, normalize='pred')
                            df_cm = pd.DataFrame(confusion_m, index=["NOSTAS", "STAS"], columns=["NOSTAS", "STAS"])
                            if Richiesta_Immagini == True:
                                plt.figure(figsize=(5, 5))
                                sns.set(font_scale=1.4)  # for label size
                                sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, vmin=0, vmax=1)
                                plt.title(tit + ':Confusion matrix for kfold {:.2f}'.format(i + 1))
                                f.tight_layout()
                                plt.tight_layout()
                                plt.savefig('{}{}_Confusion matrix (Before Lasso) for kfold {} ({}).png'.format(PercorsoImmagini_2, tit,int(i + 1),Modelli[tr]),bbox_inches="tight")
                                plt.close()

                    print('\nMean Test AUC: {:.2f} +/- {:.2f}   |  Mean Test ACC: {:.2f} +/- {:.2f}'.format(
                        auc.mean(), auc.std(), acc.mean(), acc.std()))

                    acc_before_lasso[k] = acc.mean()






                    # %%%%%%%%%%%%%%%%%%       Cross validation with lasso     %%%%%%%%%%%%%%%%%%%%% #
                    print('\nLASSO')
                    print(''.join(10 * ['-']))
                    print('\n\nCROSS VALIDATION with lasso for feature selection\n\n')
                    y_uni = y_uni.astype('int16')

                    del(model,acc,auc,skf,train,test,correlation_p,confusion_m,tit,cl,y_cl,K_pred_train,K_pred_test,fpr, tpr, thresholds)
                    skf = StratifiedKFold(n_splits=5, shuffle=True)
                    auc = np.zeros([5])
                    acc = np.zeros([5])
                    pesi = np.zeros([len(names), 5])
                    prev = np.zeros([5])  # prevalenza di stas
                    for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
                        model = LogisticRegression(C=0.2, penalty='l1', class_weight='balanced', solver='liblinear')
                        model.fit(X_uni[train, :], y_uni[train])
                        fpr, tpr, thresholds = metrics.roc_curve(y_uni[test], model.predict(X_uni[test, :]))
                        acc[i] = metrics.accuracy_score(y_uni[test], model.predict(X_uni[test, :]))
                        auc[i] = metrics.auc(fpr, tpr)
                        print('K= {} |AUC -  {:.2f}   |   ACC -  {:.2f}'.format(
                            i, auc[i], acc[i]))

                        pesi[:, i] = (np.round(model.coef_ / np.max(np.abs(model.coef_)), 2))
                        prev[i] = np.sum(y_uni[test] == 1) / len(y_uni[test])

                    print('\nMean AUC: {:.2f} +/- {:.2f}   |  Mean ACC: {:.2f} +/- {:.2f}'.format(
                        auc.mean(), auc.std(), acc.mean(), acc.std()))
                    # %%stampo tabella dei pesi
                    list2 = ['ACC: ' + el + '%' for el in list(map(str, np.round(acc * 100).astype('int16')))]
                    rcolors = plt.cm.BuPu(np.full(len(names), 0.1))
                    ccolors = plt.cm.BuPu(np.full(len(list2), 0.1))
                    if Richiesta_Immagini == True:
                        plt.figure(figsize=(8, 8))
                        plt.axis('tight')
                        plt.axis('off')
                        ytable = plt.table(cellText=pesi, rowLabels=names, cellLoc='center',rowLoc='center', colLabels=list2, colLoc='center',loc='upper left',rowColours =rcolors,colColours =ccolors, edges='closed')
                        #plt.title("Cross Validation with lasso for feature selection",loc='left')
                        plt.tight_layout()
                        plt.savefig('{}Cross Validation with lasso for feature selection.png'.format(PercorsoImmagini_2))
                        plt.close()
                    del (auc, skf, train, test,fpr, tpr, thresholds,list2)



                    # %%Fit con il modello Logistic Regression su X_uni e y_uni
                    f_used_before[k] = len(names)
                    model.fit(X_uni, y_uni)
                    acc_lasso_test_before [k] = metrics.accuracy_score(y_test, model.predict(X_test))

                    print('\n\nlasso TEST')
                    print('Accuracy lasso on test: %.2f ' % (acc_lasso_test_before [k]))

                    #Funzioni che hanno un'accuratezza minore della prevalenza in tutti gli splits
                    print('\nlasso dropped {} feature/s '.format(np.sum(acc < prev)))

                    # %% Selection based on weights of accuracy in all splits of KFOLD
                    if (np.sum(acc < prev) > 0 & param['lasso_cut']):
                        selected_over_lasso = abs(np.average(pesi, axis=1, weights=((acc > prev) * acc) + 1e-3)) > 0.15
                        X_uni = X_uni[:, selected_over_lasso]
                        X_test = X_test[:, selected_over_lasso]
                        names = names[selected_over_lasso]

                        print('Selected Features for TEST :\n {}'.format(names))
                        f_used_after[k] = len(names)
                        model.fit(X_uni, y_uni)
                        acc_lasso_test_after [k]= metrics.accuracy_score(y_test, model.predict(X_test))

                        print('\n\nLASSO TEST su spazio ristretto')
                        print('Accuracy LASSO on test: %.2f ' % (acc_lasso_test_after [k]))



                        # %%%%%%%%%%%%%%%%        Ripeto modello SCELTO su spazio ristretto      %%%%%%%%%%%%%%%% #
                        print('\n\nRipeto {} su spazio ristretto\n\n'.format(Modelli[tr]))

                        skf = StratifiedKFold(n_splits=5, shuffle=True)
                        auc = np.zeros([5])
                        acc = np.zeros([5])
                        for i, [train, test] in enumerate(skf.split(X_uni, y_uni)):
                            print('\n\nKFOLD=%.2f' % (int(i + 1)))
                            if Modelli[tr] == "KNN":
                                if param['mahalanobis_Distance']:
                                    model = KNeighborsClassifier(n_neighbors=n_neighbors_Ottimo, weights='uniform',algorithm='brute', metric='mahalanobis',metric_params={'V': np.cov(X_uni)})
                                else:
                                    model = KNeighborsClassifier(n_neighbors=n_neighbors_Ottimo, weights='uniform',algorithm='brute')
                            if Modelli[tr] == "SVC":
                                kernel = 1, 5 * RBF(0, 5)
                                model = svm.SVC(class_weight='balanced', kernel='rbf', degree=3)
                            if Modelli[tr] == "RF":
                                model = RandomForestClassifier(n_jobs=2, random_state=0)
                            if Modelli[tr] == "NB":
                                model = GaussianNB()
                            model.fit(X_uni[train, :], y_uni[train])
                            fpr, tpr, thresholds = metrics.roc_curve(y_uni[test], model.predict(X_uni[test, :]))
                            acc[i] = metrics.accuracy_score(y_uni[test], model.predict(X_uni[test, :]))
                            auc[i] = metrics.auc(fpr, tpr)

                            K_pred_base = np.ones(y_uni[test].shape)
                            print('\nSTUPIDO TEST (tutti 1 on TEST):')
                            print('Accuracy: %.2f ' % (metrics.accuracy_score(K_pred_base, y_uni[test])))
                            print('Precision and Sensibility: %.2f' % (metrics.f1_score(K_pred_base, y_uni[test])))

                            print('TEST:')
                            print('AUC -  {:.2f}   |   ACC -  {:.2f}'.format(auc[i], acc[i]))

                            # Disegno curva Roc
                            if Richiesta_Immagini == True:
                                plt.figure()
                                plt.plot(fpr, tpr, color='orange', linestyle='dashed', label='Curva Roc KNN area %0.2f' % auc[i])
                                # Disegno le curve di riferimento
                                plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Modello errato')  # tutto errato
                                plt.plot([0.01, 0.01], [0.01, 1.01], color='green', linestyle=':',
                                         label='Modello ideale')  # tutto corretto
                                plt.plot([0.01, 1.01], [1.01, 1.01], color='green', linestyle=':')  # tutto corretto
                                plt.xlabel('FPR False Positive Rate')
                                plt.ylabel('TPR True Positive Rate')
                                plt.title('Receiver Operating Characteristic (ROC) for kfold split %0.2f' % (int(i + 1)))
                                plt.legend(loc="lower right")
                                plt.savefig('{}Receiver Operating Characteristic (ROC After Lasso) for kfold split {} ({}).png'.format(PercorsoImmagini_2,int(i + 1),Modelli[tr]))
                                plt.close()

                            # matrici di confusione
                            confusion_matrix(y_uni[test], K_pred_base) / len(y_uni[test])
                            K_pred_train = model.predict(np.array(X_uni[train, :]))
                            K_pred_test = model.predict(X_uni[test, :])
                            for tit, cl, y_cl in zip(['Banale', 'test', 'train'], [K_pred_base, K_pred_test, K_pred_train],
                                                  [y_uni[test], y_uni[test], y_uni[train]]):
                                confusion_m = confusion_matrix(y_cl, cl, normalize='pred')
                                df_cm = pd.DataFrame(confusion_m, index=["NOSTAS", "STAS"], columns=["NOSTAS", "STAS"])
                                if Richiesta_Immagini == True:
                                    plt.figure(figsize=(5, 5))
                                    sns.set(font_scale=1.4)  # for label size
                                    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, vmin=0, vmax=1)
                                    plt.title(tit + ':Confusion matrix for kfold {:.2f}'.format(i + 1))
                                    f.tight_layout()
                                    plt.tight_layout()
                                    plt.savefig('{}{}_Confusion matrix (After Lasso) for kfold {} ({}).png'.format(PercorsoImmagini_2, tit,int(i + 1),Modelli[tr]))
                                    plt.close()

                        print('\nMean Test AUC: {:.2f} +/- {:.2f}   |  Mean Test ACC: {:.2f} +/- {:.2f}'.format(
                            auc.mean(), auc.std(), acc.mean(), acc.std()))

                        acc_after_lasso[k] = acc.mean()

                        del(y_cl,cl,tit,confusion_m,K_pred_train,K_pred_test,model,K_pred_base)


                    elif np.sum(acc < prev) == 0:
                        print('LASSO DID NOT FIND FEATURES TO DROP')
                        acc_after_lasso[k] = acc_before_lasso[k]
                        acc_lasso_test_after[k] = acc_lasso_test_before[k]
                        f_used_after[k] = f_used_before[k]

                    #plt.show()
                    selected_feature_out.append(names)
                    del (X_train, y_train, X_test, y_test,i,j,acc)
                    print(''.join(150 * ['-']))

                    risultati=np.array([acc_before_lasso[k],acc_after_lasso[k],acc_lasso_test_before[k],acc_lasso_test_after[k],f_used_before[k],f_used_after[k]],dtype='object')
                    if k==0:
                        sheet1 = workbook.add_sheet(Modelli[tr])
                        for iu in range(0,len(parametri)):
                            sheet1.write(0, iu+1, parametri[iu])
                        sheet1.write(0, iu + 2, 'Features After Lasso')

                    sheet1.write(JL, iu + 2," , ".join(names))
                    sheet1.write(JL, 0, 'FOLD ' + str(k + 1))
                    for iu in range(0, len(parametri)):
                        sheet1.write(JL, iu + 1, risultati[iu])

                    JL = JL + 1


                print('\n{}:'.format(NomeFileDataSet))
                print('\n MODELLO LASSO:')
                print('\nMean ACC (Before Lasso): {: .2f} +/- {: .2f}'.format(acc_lasso_test_before.mean(),acc_lasso_test_before.std()))
                print('\nMean ACC (After Lasso): {: .2f} +/- {: .2f}'.format(acc_lasso_test_after.mean(),acc_lasso_test_after.std()))
                t, p = stats.ttest_rel(acc_lasso_test_before, acc_lasso_test_after)
                if p> 0.05:
                    print('\nPaired-t-test (Ipotesis of equal average): the average is the same\n')
                else:
                    print('\nPaired-t-test (Ipotesis of equal average): the average is different\n')


                print('\n MODELLO SCELTO:')
                print('\nMean ACC {} (Before Lasso): {:.2f} +/- {:.2f}'.format(Modelli[tr],acc_before_lasso.mean(), acc_before_lasso.std()))
                print('\nMean ACC {} (After Lasso): {:.2f} +/- {:.2f}'.format(Modelli[tr],acc_after_lasso.mean(),acc_after_lasso.std()))
                t, p = stats.ttest_rel(acc_before_lasso, acc_after_lasso)
                if p > 0.05:
                    print('\nPaired-t-test (Ipotesis of equal average): the average is the same\n')
                else:
                    print('\nPaired-t-test (Ipotesis of equal average): the average is different\n')

                acc_after_lasso_classificatori[tr]=acc_after_lasso.mean()
                acc_after_lasso_classificatori_errore[tr]=acc_after_lasso.std()

                flat_list=[item for sublist in selected_feature_out for item in sublist]
                sel = Counter ( flat_list ). most_common ( 8 )
                sel=np.asarray(sel,dtype=str)
                for op in range(0,8):
                    selected_feature_out_classificatori_value[tr,op]=str(sel[op,0])
                    selected_feature_out_classificatori_count[tr, op] = str(sel[op, 1])

                f_used_after_classificatori[tr]=round(f_used_after.mean(),3)

                sheet1.write(JL, iu + 2, " , ".join(str(sel[:,0])))
                sheet1.write(JL+1, iu + 2, " , ".join(str(sel[:, 1])))


            print('\nMiglior classificatore:')
            indice_massimo=np.argmax(acc_after_lasso_classificatori)
            print('\nIl modello migliore in termini di accuratezza è:\n{} | ACC={} +- {}'.format(Modelli[indice_massimo],acc_after_lasso_classificatori[indice_massimo],acc_after_lasso_classificatori_errore[indice_massimo]))
            print('\nLe feature selezionate dal miglior classificatore sono:\n')
            for op in range(0, 8):
                print('Feature: {}   |  Count: {}'.format(selected_feature_out_classificatori_value[indice_massimo,op],selected_feature_out_classificatori_count[indice_massimo,op]))

            print('\nLe feature medie selezionate dal miglior classificatore sono:\n{}'.format(f_used_after_classificatori[indice_massimo]))



            workbook.save(pathname + '/Report-'+NomeFileDataSet)
            del(workbook)


except:
    print('Il file: ' + str(C) + ' ha dato un errore')
    workbook.save(pathname + '/Results.csv')
