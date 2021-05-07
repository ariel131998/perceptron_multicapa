#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import csv
import pandas as pd #para manejar los datos


# In[2]:


#lectura de dataset
df = pd.read_csv("pima.csv", sep=',', header=None) 
df


# In[3]:


#data = np.array(df)
#print(data)
#print(type(data))


# In[4]:


#obtenemos indice de la ultima fila (que sera la de objetivo)
m,k = df.shape # Número de ejemplos de entrenamiento(filas), número de dimensiones en los datos(columnas) 
entrenamiento_colum = k-1
print(entrenamiento_colum)


# In[5]:


#separar conjunto de entrenamiento en prueba y entrenamiento
X = df.copy() #creamos copia sin conjunto de entrenamiento 
del(X[entrenamiento_colum]) #eliminamos la columna a predecir
Y = df[entrenamiento_colum].values #dejamos solo el conjunto de entrenamiento
training_data = np.array(X,"int") #los convertimos en matriz
target_data = np.array(Y,"int")


# #### estructura de perceptron multicapa

# In[6]:


model = Sequential()
model.add(Dense(30, input_dim=8, activation='relu')) #2 capas ocultas y la entrada con 16 perceptrones en la capa oculta
model.add(Dense(1, activation='sigmoid')) #capa de salida con funcion de activación sigmoid


# ### Función de error

# In[7]:


model.compile(loss='mean_squared_error', #aqui podemos cambiar la función de error
              optimizer='adam',
              metrics=['binary_accuracy'])


# #### Realizamos entrenamiento

# In[8]:


model.fit(training_data, target_data, epochs=1000)


# #### Evaluamos el modelo

# In[9]:


scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:





# In[ ]:


#pruebas en los datos 
#input = df[:,0:8]
#output = df[:,8]

