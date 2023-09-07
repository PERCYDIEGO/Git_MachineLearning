
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import scikitplot as skplt
from scikitplot.helpers import binary_ks_curve

from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score,recall_score

def Graficos (y_true,y_probas):

    """
    y_true: (Series) 'valores reales' 
    
    y_prob: (Series) {valores predicho}
    
    """
    y_prob = y_probas[:,1]
    
    df = pd.DataFrame({'real':y_true, 'pred':y_prob})

    performances =[]
    #######ks######
    res = binary_ks_curve(y_true, y_probas[:,1])
    ks_stat = res[3]
    

    #######gini######
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    # Calcular el coeficiente GINI
    gini_coefficient = (2 * roc_auc) - 1
    
  


    # Obtener los valores de KS y Gini
    ks = ks_stat
    gini = gini_coefficient


    # Calcular la curva ROC y el AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)

    # Obtener el índice del punto máximo en la curva ROC (máximo AUC)
    idx_max_auc = np.argmax(roc_auc)

    # Obtener el umbral (threshold) asociado al punto máximo
    threshold_max_auc = thresholds[idx_max_auc]

    # Calcular la matriz de confusión
    confusion = confusion_matrix(y_true, np.round(y_prob))

    # Etiquetas de las clases
    clases = ['Malo', 'Bueno']


    # Ejemplo de etiquetas de clase (0: Malo, 1: Bueno)
    etiquetas = y_true

    
    # Calcular la frecuencia relativa acumulada de la clase buena y mala
    buenos = etiquetas == 1
    malos = etiquetas == 0
    freq_bueno = np.cumsum(buenos) / np.sum(buenos)
    freq_malo = np.cumsum(malos) / np.sum(malos)
    # Calcular la diferencia absoluta entre las frecuencias relativas acumuladas
    ks = np.max(np.abs(freq_bueno - freq_malo))
    
    
    #########################Gráficos#########################
    ##########################################################

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
  
    # Crear el gráfico de matrix de confisión
    im = axs[0,0].imshow(confusion, cmap='Blues')
    # Mostrar los valores de la matriz en cada cuadrante
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = "{:,}".format(confusion[i, j])  # Aplicar formato con separador de miles
            color = 'white' if confusion[i, j] > confusion.max() / 2 else 'black'  # Cambiar color de texto según el valor
        
            axs[0,0].text(j, i, value, ha='center', va='center', color=color, fontsize=20)

    # Configurar el gráfico de m
    axs[0,0].set_xticks(np.arange(confusion.shape[1]))
    axs[0,0].set_yticks(np.arange(confusion.shape[0]))
    axs[0,0].set_xticklabels(clases)
    axs[0,0].set_yticklabels(clases)
    axs[0,0].set_xlabel('Predicción')
    axs[0,0].set_ylabel('Verdadero')

    # Mostrar la barra de color
    cbar = axs[0,1].figure.colorbar(im, ax=axs[1,0])

    # Mostrar el gráfico
    axs[0,0].set_title('Matriz de Confusión')


    
    sns.kdeplot(y_prob[y_true==1], ax=axs[0,1], label='bueno' ,cut=0, bw_adjust=2)
    sns.kdeplot(y_prob[y_true==0], ax=axs[0,1], label='malo',  cut=0, bw_adjust=2)
    axs[0,1].set_title('Distribución de las clases')
    #axs[0,0].set_xlim(0,1)
    #axs[0,0].set_ylim(0,1)
    axs[0,1].legend()
    
    
   # Crear el gráfico de líneas para la curva ROC
    axs[1,0].plot(fpr, tpr, color='blue', label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
    axs[1,0].plot([0, 1], [0, 1], color='red', linestyle='--', label='Clasificador Aleatorio')
    axs[1,0].set_xlabel('Tasa de Falsos Positivos')
    axs[1,0].set_ylabel('Tasa de Verdaderos Positivos')
    axs[1,0].set_title('Curva ROC')
    axs[1,0].legend()
    axs[1,0].grid(True)



    # Gráfico del KS
    muestra_size = np.arange(1, len(etiquetas) + 1)
    skplt.metrics.plot_ks_statistic(y_true, y_probas, ax=axs[1,1])
    axs[1,1].set_title('Gráfico del KS')
    axs[1,1].legend()
    axs[1,1].grid(True)

 
    performances.append(['ks',ks])
    performances.append(['gini',gini])
    performances.append(['roc',roc_auc])

    Performace = pd.DataFrame(performances)



def Performance (y_true,y_probas,threshold = 0.5, change = False):

    """
    y_true: (Series) 'valores reales' 
    
    y_prob: (Series) {valores predicho}
    
    """
    y_prob = y_probas[:,1]
    if change == True:

        y_prob= (y_prob>=threshold).astype(int)
 

    
    performances =[]

    res = binary_ks_curve(y_true, y_probas[:,1])
    ks_stat = res[3]
    

    #######gini######
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    # Calcular el coeficiente GINI
    gini_coefficient = (2 * roc_auc) - 1
    

    # Obtener los valores de KS y Gini
    ks = ks_stat
    gini = gini_coefficient


    # Calcular la curva ROC y el AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)

 
    # Calcular la matriz de confusión
    confusion = confusion_matrix(y_true, np.round(y_prob))

    # Etiquetas de las clases
    clases = ['Malo', 'Bueno']


    # Ejemplo de etiquetas de clase (0: Malo, 1: Bueno)
    etiquetas = y_true

    


    #Seleccionando el Threshold
    y_pred = np.where(y_prob>=threshold,1,0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
  
    performances.append(['ks',ks])
    performances.append(['gini',gini])
    performances.append(['roc',roc_auc])
    performances.append(['accuracy',accuracy])
    performances.append(['precision',precision])
    performances.append(['recall',recall])


    Performace = pd.DataFrame(performances, columns=["Metrica","Valor"])
 
    return  Performace  