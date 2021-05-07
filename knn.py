#programa que utiliza el clasificador  k-nn
#Ortega Zitle Ariel 201719454 IA
#--------librerias-------------
import csv
import pandas as pd #para manejar los datos
import numpy as np #para operaciónes en matrices

#import matplotlib.pyplot as plt  #graficas
#from matplotlib.colors import ListedColormap
#import matplotlib.patches as mpatches
#import seaborn as sb
#import random#para inicializar k-means
#from IPython.display import display

#% matplotlib inline
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

#-----------------librerias para utilizar knn----------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix #matriz de confusion 
from sklearn.model_selection import KFold  #validacion cruzada
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
#-----------------------funciones------------------------
#función que lee el archivo donde se encuentra nuestro conjunto de entrenamiento
def leer_datos():
    print("dame el nombre del archivo (formato csv):")
    nombre_archivo = input()
    nombre_archivo = nombre_archivo + '.csv'
    print(nombre_archivo)
    df = pd.read_csv(nombre_archivo, sep=',', header=None) 
    df
    print(df)
    return df
def knn_implementado(df,k):
    print("Dame el numero de la columna que sera el conjunto de entrenamiento:")
    entrenamiento_colum = int(input())
    entrenamiento_colum = entrenamiento_colum-1
    #print(entrenamiento_colum)
    X = df.copy() #creamos copia sin conjunto de entrenamiento 
    del(X[entrenamiento_colum]) #eliminamos la columna a predecir
    Y = df[entrenamiento_colum].values #dejamos solo el conjunto de entrenamiento 

    #preparamos las entradas
    x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=0) #entrenamiento 75% y test 25%
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #clasificacion
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)#hace el entrenamiento
    print('Precisión del clasificador K-NN en el conjunto de entrenamiento: {:.2f}'
        .format(knn.score(x_train, y_train)))
    print('Precisión del clasificador K-NN en el conjunto de prueba:: {:.2f}'
        .format(knn.score(x_test, y_test)))
    #validación cruzada (solo utilizaremos el de set de entrenamiento por el momento)
    print("Dame el numero de k para la validacion cruzada")
    n_splits = int(input())
    kf = KFold(n_splits)
    scores = cross_val_score(knn, x_train, y_train, cv=kf, scoring="accuracy") 
    print("Metricas cross_validation", scores)
    print("Media de cross_validation", scores.mean())
    #matriz de confucion sobre set de prueba 
    pred = knn.predict(x_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
#-------------------------variables-----------------------------    
f=0
op = 1
#-------------------------Clase principal--------------------------

while f!=1:
    print("MENU")
    print("1.clasificación de datos con K-NN")
    print("2.Salir")
    op = int(input())
    if op == 1:
        print("leyendo datos")
        df=leer_datos()
        print(df.describe())
        k = int(input("dame el valor para K: "))
        knn_implementado(df,k)
    elif op == 2:
        print("Saliendo")
        f=1
    else:
        print("Opnción incorrecta ingrese nuevamente")
