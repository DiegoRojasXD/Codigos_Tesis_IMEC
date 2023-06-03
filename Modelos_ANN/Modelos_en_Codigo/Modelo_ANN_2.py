# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:37:05 2023

@author: HP 690 -000B
"""
import os
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

wb=openpyxl.load_workbook('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IMEC\Datos_2.xlsx')
print(wb.sheetnames)

Data=pd.read_excel(io='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IMEC\Datos_2.xlsx', sheet_name='Datos_2',header=0,names=None, 
                   index_col=None,usecols='A:B',engine= 'openpyxl' )
print(Data.head(10))

Data_numeros=Data["Meteocontrol - Radiación [W/m2]"]
for i in range(len(Data_numeros)):
  if Data_numeros[i] <1:
    Data_numeros[i]=0

Data_train=Data_numeros[:70]
Data_test=Data_numeros[70:]


"""# Machine Learning"""

Data['fecha']=pd.to_datetime(Data.Timestamp, format='%Y-%m-%d %H:%M:%S', errors='coerce')

Data.head()

Data[Data.fecha.isna()==True]

Data=Data[['fecha', 'Meteocontrol - Radiación [W/m2]']]

data_time_s=Data.pop('fecha')

timemap_s=data_time_s.map(datetime.datetime.timestamp)

day=24*60*60
year=(365.2425)*day

Data['day_sin']=np.sin(timemap_s*(2*np.pi/day))
Data['day_cos']=np.cos(timemap_s*(2*np.pi/day))
Data['year_sin']=np.sin(timemap_s*(2*np.pi/year))
Data['year_cos']=np.cos(timemap_s*(2*np.pi/year))

Data['fecha']=data_time_s

Data.head()

print("Fecha de inicio es: ", Data.fecha.min())
print("Fecha de fin es: ", Data.fecha.max())

Data=Data.loc[Data.fecha >'2023-02-02']
train,test=Data.loc[Data.fecha <='2023-02-10'],Data.loc[Data.fecha >'2023-02-15']

data_time_train=train.pop('fecha')
data_time_test=test.pop('fecha')

test.head()

scaler=MinMaxScaler()
scaler=scaler.fit(train)

train[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']]=scaler.transform(train[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']])
test[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']]=scaler.transform(test[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']])

train.head()

def create_dataset(X,y,times_steps=1):
  Xs,ys=[],[]
  for i in range(len(X)-time_steps):
    v=X.iloc[i:(i+time_steps)].values
    Xs.append(v)
    ys.append(y.iloc[i+time_steps])
  return np.array(Xs), np.array(ys)

train["Meteocontrol"]=train["Meteocontrol - Radiación [W/m2]"]
train=train[['Meteocontrol','day_sin','day_cos','year_sin','year_cos']]

test["Meteocontrol"]=test["Meteocontrol - Radiación [W/m2]"]
test=test[['Meteocontrol','day_sin','day_cos','year_sin','year_cos']]

numero_dias_anteriores_para_predecir=6
time_steps=24*12*numero_dias_anteriores_para_predecir

X_train,y_train=create_dataset(train, train.Meteocontrol, time_steps)


X_test,y_test=create_dataset(test,test.Meteocontrol,time_steps)

print(X_train.shape,y_train.shape)

model=tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(
    units=512,
    input_shape=(X_train.shape[1],X_train.shape[2])
))

model.add(tf.keras.layers.Dense(units=1))
model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics='accuracy'
)

callback=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history=model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    shuffle=False,
    callbacks=[callback]
)

# hacemos las grafica 2 perdidas
plt.plot(history.history['loss'],label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.title('Perdidas del modelo')
plt.ylabel('perdidas')
plt.xlabel('epocas')
plt.legend(['Entrenamiento', 'test'], loc='upper left')
plt.show()

# hacemos las grafica 1 precision
plt.plot(history.history['accuracy'],label='Training loss')
plt.plot(history.history['val_accuracy'],label='Validation loss')
plt.title('Precision del modelo')
plt.ylabel('Precision')
plt.xlabel('epocas')
plt.legend(['Entrenamiento', 'test'], loc='upper left')
plt.show()


target_dir = 'D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IMEC\Modelos\Modelos_ANN\Resultado_de_modelos\modelo2'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IMEC\Modelos\Modelos_ANN\Resultado_de_modelos\modelo2\modelo.h5')
model.save_weights('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IMEC\Modelos\Modelos_ANN\Resultado_de_modelos\modelo2\pesos.h5')



y_pred=model.predict(X_test)

plt.plot(y_test*1200, label='Real')
plt.plot(y_pred*1200, label='Predicted')
plt.title('Predicción del modelo')
plt.ylabel('Irradiancia [W/m2]')
plt.xlabel('Horas [h]')
plt.legend(['Real', 'Predicción'], loc='upper left')
plt.show()

sum=0
sum2=0
for i in range(len(y_pred)):
    e=y_test[i]-y_pred[i]
    sum=sum+abs(e)
    
for j in range(len(y_test)):
    sum2=sum2+abs(y_test[j])
WAPE=sum/sum2
print(WAPE)