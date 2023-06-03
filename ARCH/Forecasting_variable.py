# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:22:06 2023

@author: HP 690 -000B
"""

from arch import arch_model
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt


wb=openpyxl.load_workbook('/content/drive/MyDrive/Colab Notebooks/Datos_2.xlsx')
print(wb.sheetnames)

Data=pd.read_excel(io='/content/drive/MyDrive/Colab Notebooks/Datos_2.xlsx', sheet_name='Datos_2',header=0,names=None, 
                   index_col=None,usecols='A:B',engine= 'openpyxl' )
Data.head(10)

Data_numeros=Data["Meteocontrol - Radiación [W/m2]"]
for i in range(len(Data_numeros)):
  if Data_numeros[i] <1:
    Data_numeros[i]=0

Data_train=Data_numeros[:70]
Data_test=Data_numeros[70:]

for k in range(1,20):
    Data_train=Data_numeros[:70]
    Data_test=Data_numeros[70:]

    cont=len(Data_train)
    c3= pd.DataFrame(0, index=range(len(Data_test)), columns=range(1))
    for i in range(int(len(Data_test)/k)):
        # Build Model
        returns = Data_train.dropna()
        am = arch_model(returns, vol="GARCH", power=2.0, p=k, o=2, q=k)
        res = am.fit(update_freq=5)

        # Forecast
        forecasts = res.forecast(horizon=k, reindex=False)
        for j in range(k):
          if forecasts.variance.iloc[-1].values[j]<1:
            forecasts.variance.iloc[-1].values[j]=0

          c3.loc[i+j]=forecasts.variance.iloc[-1].values[j]/1000
          Data_train[cont+i+j]=Data_test.iloc[i+j]
    fc_series3 = c3.set_index(Data_test.index)

    sum=0
    sum2=0
    for i in range(len(Data_test)):
      e=Data_test.iloc[i]-fc_series3.iloc[i]
      sum=sum+abs(e.iloc[0])
    for j in range(len(Data_test)):
      sum2=sum2+abs(Data_test.iloc[i])
    WAPE=sum/sum2


    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(Data_train, label='training')
    plt.plot(Data_test, label='actual')
    plt.plot(fc_series3, label='Forcast')
    plt.title('Forecast vs Actuals con k de: '+str(k)+' y WAPE de: '+str(round(WAPE*100,2))+'%')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()