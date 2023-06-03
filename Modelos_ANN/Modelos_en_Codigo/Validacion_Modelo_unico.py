# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:37:47 2023

@author: Usuario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


Data=pd.read_excel(io='D:\Documentos\Tesis_Diego_Rojas_2023\IMEC\Datos_2.xlsx', sheet_name='Datos_2',header=0,names=None, 
                   index_col=None,usecols='A:B',engine= 'openpyxl' )


"""# Machine Learning"""

Data['fecha']=pd.to_datetime(Data.Timestamp, format='%Y-%m-%d %H:%M:%S', errors='coerce')

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

print("Fecha de inicio es: ", Data.fecha.min())
print("Fecha de fin es: ", Data.fecha.max())

Data=Data.loc[Data.fecha >'2023-02-01']
test=Data.loc[Data.fecha <='2023-02-12']

data_time_test=test.pop('fecha')

scaler=MinMaxScaler()
scaler=scaler.fit(test)

test[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']]=scaler.transform(test[['Meteocontrol - Radiación [W/m2]','day_sin','day_cos','year_sin','year_cos']])

def create_dataset(X,y,times_steps=1):
  Xs,ys=[],[]
  for i in range(len(X)-time_steps):
    v=X.iloc[i:(i+time_steps)].values
    Xs.append(v)
    ys.append(y.iloc[i+time_steps])
  return np.array(Xs), np.array(ys)

test["Meteocontrol"]=test["Meteocontrol - Radiación [W/m2]"]
test=test[['Meteocontrol','day_sin','day_cos','year_sin','year_cos']]

numero_dias_anteriores_para_predecir=6
time_steps=24*12*numero_dias_anteriores_para_predecir


X_test,y_test=create_dataset(test,test.Meteocontrol,time_steps)


longitud, altura = 60, 60
modelo = 'D:\Documentos\Tesis_Diego_Rojas_2023\IMEC\Modelos_ANN\Resultado_de_modelos\modelo3\modelo.h5'
pesos_modelo = 'D:\Documentos\Tesis_Diego_Rojas_2023\IMEC\Modelos_ANN\Resultado_de_modelos\modelo3\pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)

  return answer

y_pred=model.predict(X_test)

y_pred=y_pred+0.2
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


for i in range(len(y_pred)):
    if y_pred[i]*1200<400:
        y_pred[i]=0
        
plt.plot(y_test*1200, label='Real')
plt.plot((y_pred)*1200, label='Predicted')
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
print('El WAPE es de: '+str(WAPE))
EMA=sum/len(y_pred)
print('El EMA es de: '+str(EMA))

sum3=0
for i in range(len(y_pred)):
    e=(y_test[i]-y_pred[i])**2
    sum3=sum3+e
ECM=sum3/len(y_pred)
print('El ECM es de: '+str(ECM))

RECM=np.sqrt(ECM)
print('El RECM es de: '+str(RECM))