# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:14:04 2023

@author: HP 690 -000B
"""

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict
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

# Difference the orginal sales data
plt.figure(figsize=(15,8));
plt.plot(Data_numeros)

Data_train=Data_numeros[:70]
Data_test=Data_numeros[70:]

df1 = pd.DataFrame()
cont=len(Data_train)
c = pd.DataFrame(0, index=range(len(Data_test)), columns=range(1))
for i in range(len(Data_test)):
  # Build Model
  model = sm.tsa.arima.ARIMA(Data_train, order=(3,1,0))  
  fitted = model.fit()  
  print(fitted.summary())

  # Actual vs Fitted
  fig, ax = plt.subplots()
  ax = Data_train.plot(ax=ax)
  plot_predict(fitted, ax=ax)
  plt.show()

  # Forecast
  fc = fitted.forecast(alpha=0.01)  # 95% conf
  df1[cont+i]=fc.iloc[0]
  if fc.iloc[0]<1:
    fc.iloc[0]=0

  c.iloc[i]=fc.iloc[0]
  Data_train[cont+i]=Data_test.iloc[i]
  #for i in range(len(fc_series)):
  #  fc_series.iloc[i]=fc.iloc[i]
# Plot
# Make as pandas series
fc_series = c.set_index(Data_test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(Data_train, label='training')
plt.plot(Data_test, label='actual')
plt.plot(fc_series, label='Forcast')
#plt.plot(c, label='Forcast2')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
