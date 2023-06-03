# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:19:22 2023

@author: HP 690 -000B
"""

from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt

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
for j in range(len(Data_test)):
    # Build Model
    alpha = 0.6
    ses = SimpleExpSmoothing(Data_train)
    model = ses.fit(smoothing_level = alpha, optimized = False)

    # Forecast
    forecast = model.forecast(2)  
    if forecast.iloc[1]<1:
      forecast.iloc[1]=0
    c.loc[j]=forecast.iloc[1]
    Data_train[cont+j]=Data_test.iloc[j]
fc_series2 = c.set_index(Data_test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(Data_train, label='training')
plt.plot(Data_test, label='actual')
plt.plot(fc_series2, label='Forcast')
#plt.plot(c, label='Forcast2')
plt.title('Forecast vs Actuals con alpha de: '+str(alpha))
plt.legend(loc='upper left', fontsize=8)
plt.show()