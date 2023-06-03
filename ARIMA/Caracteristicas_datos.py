# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:10:38 2023

@author: HP 690 -000B
"""
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


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


# Difference the orginal sales data
plt.figure(figsize=(15,8));
train_diff_seasonal = Data_train - Data_train.shift(3)
plt.plot(train_diff_seasonal)

# Conduct the test
series = train_diff_seasonal.dropna().values
result = adfuller(series)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

Data_train = Data_train.dropna()
Data_train.isna().sum()

series = Data_train.values
result = adfuller(series)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Split train, validation and test sets
train = Data_train[:85]
validation = Data_train[85:100]
test = Data_train[100:]

# ACF and PACF for orginal data
series=train.dropna()
fig, ax = plt.subplots(2,1, figsize=(10,8))
fig = sm.graphics.tsa.plot_acf(series, lags=None, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(series, lags=None, ax=ax[1])
plt.show()