# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:21:23 2023

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

cont=len(Data_train)
c3= pd.DataFrame(0, index=range(len(Data_test)), columns=range(1))
for i in range(len(Data_test)):
    # Build Model
    returns = Data_train.dropna()
    am = arch_model(returns, vol="GARCH", power=2.0, p=1, o=5, q=1)
    res = am.fit(update_freq=5)

    # Forecast
    forecasts = res.forecast(horizon=len(Data_test), reindex=False)
    if forecasts.variance.iloc[-1].values[0]<1:
      forecasts.variance.iloc[-1].values[0]=0
    c3.loc[i]=forecasts.variance.iloc[-1].values[0]/1000
    Data_train[cont+i]=Data_test.iloc[i]
fc_series3 = c3.set_index(Data_test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(Data_train, label='training')
plt.plot(Data_test, label='actual')
plt.plot(fc_series3, label='Forcast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

sum=0
sum2=0
for i in range(len(Data_test)):
  e=Data_test.iloc[i]-fc_series3.iloc[i]
  sum=sum+abs(e.iloc[0])
for j in range(len(Data_test)):
  sum2=sum2+abs(Data_test.iloc[i])
WAPE=sum/sum2
print(WAPE*100)