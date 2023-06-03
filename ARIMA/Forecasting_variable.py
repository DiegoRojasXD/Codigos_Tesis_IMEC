# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:16:30 2023

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

for k in range(1,20):
  Data_train=Data_numeros[:70]
  Data_test=Data_numeros[70:]
  df1 = pd.DataFrame()
  cont=len(Data_train)
  c = pd.DataFrame(0, index=range(len(Data_test)), columns=range(1))
  for i in range(int(len(Data_test)/k)):
    # Build Model
    model = sm.tsa.arima.ARIMA(Data_train, order=(3,1,0))  
    fitted = model.fit()  
    #print(fitted.summary())

    # Actual vs Fitted
    #fig, ax = plt.subplots()
    #ax = Data_train.plot(ax=ax)
    #plot_predict(fitted, ax=ax)
    #plt.show()

    # Forecast
    fc = fitted.forecast(k,alpha=0.01)  # 95% conf
    for j in range(k):
        df1[cont+j+i]=fc.iloc[j]
        if fc.iloc[j]<1:
          fc.iloc[j]=0

        c.iloc[i+j]=fc.iloc[j]
        Data_train[cont+i+j]=Data_test.iloc[i+j]
    #for i in range(len(fc_series)):
    #  fc_series.iloc[i]=fc.iloc[i]
  # Plot
  # Make as pandas series
  fc_series = c.set_index(Data_test.index)


  sum=0
  sum2=0
  for i in range(len(Data_test)):
    e=Data_test.iloc[i]-fc_series.iloc[i]
    sum=sum+abs(e.iloc[0])
  for j in range(len(Data_test)):
    sum2=sum2+abs(Data_test.iloc[i])
  WAPE=sum/sum2


  plt.figure(figsize=(12,5), dpi=100)
  plt.plot(Data_train, label='training')
  plt.plot(Data_test, label='actual')
  plt.plot(fc_series, label='Forcast')
  #plt.plot(c, label='Forcast2')
  plt.title('Forecast vs Actuals con k de: '+str(k)+' y WAPE de: '+str(round(WAPE*100,2))+'%')
  plt.legend(loc='upper left', fontsize=8)
  plt.show()