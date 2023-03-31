import pandas as pd
import numpy as np
import sidetable
import src as sp
from scipy import stats
import math

#librerias limpieza de nulos
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

#Normalizacion(Variable Respuesta)
from sklearn.preprocessing import MinMaxScaler

#Estandarización
## esto es un metodo que estandariza automaticamente todas las columnas del dataframe que le pasemos
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler #menos sensible a los outliers
#libreria para el balanceo
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# librerías de visualización
import seaborn as sns
import matplotlib.pyplot as plt

# librerías para crear el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder  

from sklearn import tree

# para calcular las métricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import cohen_kappa_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")
import pickle


def norma_minmaxscaller(df,columna): #la columna entre comillas
    # construir el modelo de escalador
    minmax = MinMaxScaler()
    # ajustamos el modelo utilizando nuestro set de datos , ***DOBLE CORCHETE
    minmax.fit(df[[columna]])
    # transformamos los datos
    X_normalizadas = minmax.transform(df[[columna]])
    # lo unimos a nuestro dataframe original
    df[columna + '_NORM'] = X_normalizadas
    return df
    

#%%
 # construir el modelo de escalador
#robust = RobustScaler() hay que ponerlo fuera del 
def estandar_robustscaller(df,lista_columnas,robust):
    # ajustamos el modelo utilizando nuestro set de datos
    robust.fit(df[lista_columnas])
    X_robust = robust.transform(df[lista_columnas])
    df[lista_columnas] = X_robust

    with open (f"data/robust_{lista_columnas[0]}.pkl", "wb") as estandarizacion:
        pickle.dump(robust, estandarizacion)
    return df

"""#%%
def descision_tree()
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 
    # fit the regressor with X and Y data
    regressor.fit(X_train, y_train)"""