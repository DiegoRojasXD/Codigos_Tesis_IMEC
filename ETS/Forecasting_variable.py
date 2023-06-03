# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:19:43 2023

@author: HP 690 -000B
"""

from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt

for k in range(1,20):
    Data=pd.read_excel(io='/content/drive/MyDrive/Colab Notebooks/Datos_3.xlsx', sheet_name='Datos_3',header=0,names=None, 
                      index_col=None,usecols='A:B',engine= 'openpyxl' )
    Data_numeros=Data["Meteocontrol - Radiación [W/m2]"]



    for i in range(len(Data_numeros)):
      if Data_numeros[i] <1:
        Data_numeros[i]=0

    Data_train=Data_numeros[:70]
    Data_test=Data_numeros[70:]
    cont=len(Data_train)
    c = pd.DataFrame(0, index=range(len(Data_test)), columns=range(1))
    for i in range(int(len(Data_test)/k)):
        # Build Model
        alpha = 0.97
        ses = SimpleExpSmoothing(Data_train)
        model = ses.fit(smoothing_level = alpha, optimized = False)

        # Forecast
        forecast = model.forecast(k) 
        for j in range(k):
          if forecast.iloc[j]<1:
            forecast.iloc[j]=0

          c.iloc[i+j]=forecast.iloc[j]
          Data_train[cont+i+j]=Data_test.iloc[i+j]
    fc_series2 = c.set_index(Data_test.index)

    sum=0
    sum2=0
    for i in range(len(Data_test)):
      e=Data_test.iloc[i]-fc_series2.iloc[i]
      sum=sum+abs(e.iloc[0])
    for j in range(len(Data_test)):
      sum2=sum2+abs(Data_test.iloc[i])
    WAPE=sum/sum2

    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(Data_train, label='training')
    plt.plot(Data_test, label='actual')
    plt.plot(fc_series2, label='Forcast')
    #plt.plot(c, label='Forcast2')
    plt.title('Forecast vs Actuals con alpha de: '+str(alpha)+' con k de: '+str(k)+' y WAPE de: '+str(round(WAPE*100,2))+'%')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()