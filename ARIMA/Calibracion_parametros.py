# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:12:54 2023

@author: HP 690 -000B
"""
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm


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


Test=int(len(Data_numeros)*0.7)
Data_train=Data_numeros[:Test]
Data_test=Data_numeros[Test:]


df = Data_train

model = pm.auto_arima(df, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())