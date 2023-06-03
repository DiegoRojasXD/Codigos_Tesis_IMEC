# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:25:32 2023

@author: HP 690 -000B
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

rango=['01','08','14','20','25']
potencia=[]
energia=[]
for f in range(1,len(rango)):

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
    
    Data=Data.loc[Data.fecha >'2022-03-'+rango[f-1]]
    test=Data.loc[Data.fecha <='2022-03-'+rango[f+1]]
    
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
        if y_pred[i]*1200<150:
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
    print(WAPE)

    Longitud_Celda=0.1617
    Area_Celda=Longitud_Celda*Longitud_Celda
    Area_Panel=Area_Celda*72
    Area_Total=Area_Panel*200
    Potencia=Area_Total*y_pred
    Energía=Potencia*(5/60)
    
    
    plt.plot(Potencia)
    plt.title('Predicción potencia')
    plt.ylabel('Potencia [W]')
    plt.xlabel('Horas [h]')
    plt.show()
    
    plt.plot(Energía)
    plt.title('Energía disponible')
    plt.ylabel('Energía [kW]')
    plt.xlabel('Horas [h]')
    plt.show()
    
    Potencia_1=Potencia[0:250]
    Potencia_2=Potencia[250:500]
    Potencia_3=Potencia[500:750]
    Potencia_4=Potencia[750:1000]
    Potencia_5=Potencia[1250:1500]

    Potencia_Maxima=[max(Potencia_1),max(Potencia_2),max(Potencia_3),max(Potencia_4),max(Potencia_5)]
    
    potencia.append(Potencia_Maxima)
dia=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.stem(dia,potencia)
plt.title('Predicciòn de las potencias máximas del mes de febrero del 22 al 26')
plt.ylabel('Potencia_Maxima [kW]')
plt.xlabel('Día')
plt.show()

errores=[1.46499/2,0.78551/2,0.83976/2,0.85298/2,0.859032/2,0.932513/2,0.9324524/2,0.942828/2,1.0173898/2,1.2236219/2]
neuronas_segunda_capa=[2,4,8,16,32,64,128,256,512,1024]
errores_2=[0.8712,0.9188,1.012,0.8584,0.8371,0.8952,0.911,0.8418,0.9399,0.84531]

errores_3=[19.84,]

plt.plot(neuronas_segunda_capa,errores)
plt.xscale("log")
plt.title('Errores a partir de las neuronas de segunda capa')
plt.ylabel('Errores')
plt.xlabel('Número de neuronas')
plt.show()