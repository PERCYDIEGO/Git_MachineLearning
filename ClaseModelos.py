################################FUNCIÓN PARA ANALIZAR LA DATA################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb  
from Modulos.ValidadorModelos import Graficos, Performance
import scipy.stats as stats

class CreateModelos:
    
    """
    Autor: Creado por PDLA.

    parametros
    ------


    X: |DataFrame| 

    y: |Series|

    test_size (int, float): Tamaño del test

    """
    def __init__(self, X_train, X_test, y_train, y_test):
    
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
   
        print('X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape))
        
        
    def ModelSelection(self,num_model, score, threshold=0.5, change=False):
        """
        num_model: (int) {0: 'Regresión lógistica', 1: 'Random Forest', 2: 'Xgboost', 3: 'Lgbm'}

        """


        if num_model ==0:
            self.model ="LogisticRegression"

            #random Forest
            pipeline = LogisticRegression()
            
            parameters = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1.0, 10.0],
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500],
                'intercept_scaling': [1, 2, 5],
                'multi_class': ['auto', 'ovr', 'multinomial'],
                'warm_start': [True, False]
                }
            
        elif num_model ==1:
            self.model ="RandomForestClassifier"
            #random Forest
            pipeline = RandomForestClassifier()

            parameters = {
                'n_estimators': [10, 20, 50],
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [10, 15],
                'min_samples_leaf': [10, 15],
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced']
                }            
       
       
        elif num_model ==2:
            self.model ="LGBMClassifier"
            #random Forest
            pipeline = lgb.LGBMClassifier()

            parameters = {
                'boosting_type': ['gbdt', 'goss'],
                'num_leaves': [31, 50],
                'learning_rate': [0.1, 0.01],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1],
                'min_child_samples': [50, 100, 200]
                }



        elif num_model ==3:
            self.model ="XGBClassifier"
            #random Forest
            pipeline =  XGBClassifier(objective='binary:logistic',
                    nthread=4,
                    seed=42)

            parameters = {
            'max_depth': [3, 5],
            'learning_rate': stats.uniform(0.01, 0.1),
            'n_estimators': stats.randint(50, 150),
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3]
            }




        else:
            print('Eliga un valor del 1-3')
            breakpoint
            
        grid_pipeline = GridSearchCV(pipeline, parameters,cv=5, n_jobs=-1,scoring=score,  verbose=True)
        grid_pipeline.fit(self.X_train, self.y_train)   

        self.y_true_test = self.y_test
        self.y_probas_test = grid_pipeline.predict_proba(self.X_test)

        self.y_true_train = self.y_train
        self.y_probas_train  = grid_pipeline.predict_proba(self.X_train)

        
        #Graficos(y_true,y_prob)
        t1= Performance(self.y_true_test,self.y_probas_test,threshold,change)
        t1['Tipo'] = 'Testing'
        t1['modelo'] = self.model

        t2= Performance(self.y_true_train,self.y_probas_train,threshold, change)
        t2['Tipo'] = 'Train'
        t2['modelo'] = self.model
      
        df = pd.concat([t1,t2])

        return df.reset_index(drop=True)[['modelo', 'Tipo','Metrica','Valor']]
        #return df.reset_index(drop=True)
    
    def Showplot(self,tipo=0):
        
        if tipo==0:
            print("Información del Test usando el modelo:{}".format(self.model))
            Graficos(self.y_true_test,self.y_probas_test)

        elif tipo==1:
            print("Información del Train usando el modelo:{}".format(self.model))
            Graficos(self.y_true_train,self.y_probas_train)
        else:
            print('Elige la opción:{0:Test, 2:Train')

      

