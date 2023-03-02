# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:14:33 2023

@author: pc
"""

import time

from numba import jit, cuda
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV

from scipy.stats import normaltest, kruskal

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from statsmodels.formula.api import logit
from joblib import dump, load
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import re 



def create_path():
    path="C:/Users/pc/Nextcloud/Python/Flask"
    path_fig=path+'/figures'
    path_models=path+'/models'


    isExist = os.path.exists(path_fig)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_fig)
    
    isExit=os.path.exists(path_models)
    if not isExist:
        os.makedirs(path_models)
    
    return(path,path_fig,path_models)
    
    
    

def data_load():
    X, y=load_breast_cancer(return_X_y=True, as_frame=True)
    
    return(X,pd.DataFrame(y))


def split (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    x_mean=X_train.mean(axis=0)
    x_std =X_train.std(axis=0)
    
    X_train_scaled= (X_train - x_mean)/x_std
       
    
    dict_stat={"x_mean": x_mean,
               'x_std':x_std}
    
    return(X_train_scaled, X_test, y_train, y_test, dict_stat)


@jit
def train( X_train_scaled, X_test, y_train, y_test, dict_stat, models,params,path_models,names):
    
    x_mean=dict_stat["x_mean"]
    x_std=dict_stat["x_std"]
   
    i=0
    for model in models:
        sco=pd.DataFrame(data=None,columns=["Score", "best_model"])  
        for score in ['roc_auc','precision','recall','accuracy']:
            clf=GridSearchCV(model, params[i],  return_train_score=True, scoring=score )
            clf.fit(X_train_scaled,y_train)
            #sco_temp=pd.DataFrame(data=[[score, clf.best_estimator_]],columns=["Score", "best_model"])
            dump(clf.best_estimator_, path_models+"/"+names[i ]+"_"+score+".joblib") 
            #sco=sco.append(sco_temp,ignore_index=True)
        i=i+1




def main():
    
    path,path_fig,path_models=create_path()
    
    
    X,y=data_load()
    
    X.describe()
    
    #See if thee are NA in data 
    X.isna().any()

    #Number of NA in each feature
    X.isna().sum()

    # features with continues nature
    X.dtypes
    #target with discret nature (0 : not sick , 1: sick)
    y.dtypes
    pd.unique(y['target'])

    len(X.columns)
    data=pd.concat([X,y],axis=1)
    #i=1

    #historgram for features considering target variable
    plt.Figure(figsize=(18,18))
    for col in X.columns:
        ax=sns.histplot(x=col,hue='target',data=data,stat="probability",kde=True )
        ax.set(xlabel=col)
        #i=i+1
    
        plt.savefig(path_fig+"/"+"histoplot_"+col.replace(" ","_")+".png")
        plt.show()
     
        # distribution of target variable 
    y.value_counts().plot.pie()
    plt.savefig(path_fig+"/target.png")


    ### dependency between features and target variable : anova analaysis ####

    """ first approach : test point biserial test but first we need to check normality"""
    # normality test : normality is ok for all features
    for col in X.columns: 
        k2,p =normaltest(X[col])
        print(p)

    rslt=pd.DataFrame(data=None, columns= ["Variable", "Correlation", "p_value"])
    for col in X.columns:
        x=stats.pointbiserialr(X[col],y)
        rslt_temp= pd.DataFrame(data=[[col,x[0],x[1]]], columns=["Variable", "Correlation", "p_value"])
        rslt=rslt.append(rslt_temp, ignore_index=True)
    
    
    variable_corr_non= rslt[rslt["p_value"]<=0.05]["Variable"]


    """ second approach : test significativity for logistic regression coefficient """
    rslt=pd.DataFrame(data=None, columns= ["Variable", "p_value"])
    for col in X.columns:
        data_temp=pd.concat([y,X[col]], axis=1)
        data_temp.rename(columns = {'target':'y', col:"x"}, inplace = True)
        model=logit('y ~ 0+x', data=data_temp)
        rs=model.fit()
        rslt_temp= pd.DataFrame(data=[[col,float(rs.pvalues)]], columns=["Variable", "p_value"])
        rslt=rslt.append(rslt_temp, ignore_index=True)
    
    variable_non_sign = rslt[rslt["p_value"]<=0.05]["Variable"]


    """ third approach :  Kruskal-Wallis test """
    rslt=pd.DataFrame(data=None, columns= ["Variable", "p_value"])
    for col in X.columns:
        k2,p=kruskal(pd.DataFrame(X[col]),y)
    
        rslt_temp= pd.DataFrame(data=[[col,p]], columns=["Variable", "p_value"])
        rslt=rslt.append(rslt_temp, ignore_index=True)
    
    variable_non_sign_kru = rslt[rslt["p_value"]<=0.05]["Variable"]

    temp=pd.merge(variable_corr_non, variable_non_sign ,how="outer",on='Variable')
    variable_final=pd.merge(temp, variable_non_sign_kru)
    variable_final=list(variable_final['Variable'])


    X=X[variable_final]
    
    X_train_scaled, X_test, y_train, y_test, dict_stat=split(X,y)
    
    model1=LogisticRegression()
    param1={'penalty' :['l1', 'l2', 'elasticnet'],
            'solver' :['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'fit_intercept' :[True, False]        
           }
    
    model2=LinearSVC()
    param2={'penalty' : ['l1', 'l2'],
            'fit_intercept' :[True, False]
           }
    
    model3=SGDClassifier()
    param3={ 'loss': ['hinge', 'log_loss', 'log', 'modified_huber',
                      'squared_hinge', 'perceptron', 'squared_error', 'huber',
                      'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty' :['l1', 'l2', 'elasticnet'],
            'fit_intercept' :[True, False] 
           }
    
 
    model4=KNeighborsClassifier()
    param4={'n_neighbors' : [1,2,3,4,5,6,7,8,9],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
           }
    
    
    model5=DecisionTreeClassifier()
    param5={'criterion' : ['gini', 'entropy', 'log_loss'] 
           }
    
    models=[model1,model2,model3,model4,model5]
    params=[param1,param2,param3,param4,param5]
    
    names=["Logistic_regression", "Linear_SVC","SGD_classifier","KNN","Decision_tree"]
    
    train(X_train_scaled, X_test, y_train, y_test, dict_stat, models,params,path_models,names)
    

    


tps1 = time.time()
main()
tps2 = time.time()

print(tps2-tps1)
    