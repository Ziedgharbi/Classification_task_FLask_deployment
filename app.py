import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import r2_score 
from joblib import dump, load

from numba import jit, cuda

import os 
import re 

import sys

project_root="C:/Users/pc/Nextcloud/Python/Flask"

model_path=project_root+"/models/"
sys.path.append(str(project_root))


from annexe_app import data_load, split

from flask import Flask , request, render_template


app=Flask(__name__,template_folder=project_root,static_folder=project_root+"/test_files")



@app.route('/')
def home():
    return render_template('index.html')

@app.route("/" , methods=["POST"])
def predict():
    
    X,y=data_load()
    X_train_scaled, X_test, y_train, y_test, dict_stat=split(X,y)
    x_mean=dict_stat["x_mean"]
    x_std=dict_stat["x_std"]
    
    
    models= request.form.getlist ("models")
    criteria= request.form.getlist ("metric")
    dataaa =[x for x in request.form.values() ]
    data=dataaa[0:30]
    
    data=  np.array(data, dtype=float)

    data_scaled=(data-x_mean)/x_std

   
 
    
    l=[]
    for model in models :
        for crit in criteria :
             l.append(model+"_"+crit)
  
    rs={ }
    
    for mod in l :
        model=load(model_path+mod+'.joblib')
        data=np.array(data)
        rslt= model.predict([data_scaled])
        if rslt.item() ==0:
            rsl= "Malignant"
        if rslt.item() ==1:
            rsl= "benign "
        rs[mod]= rsl
        
    
    txt=[]
    for key, value in rs.items():
        
        txt_temp= str(key)+" ----> "+ str(value)
        txt.append( txt_temp)
        
  
    return render_template('index.html', pred=txt)
   
   
if __name__ == '__main__':
    app.run()

   
