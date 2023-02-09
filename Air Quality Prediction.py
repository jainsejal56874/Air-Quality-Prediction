#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################load data
df=pd.read_excel("Downloads/AirQualityUCI.xlsx")
#print(data.shape)#(9357, 15)
import numpy as np
df.replace(-200,np.nan,inplace=True)
#  replace  the  missing values  by the maen of that  column
df.replace(np.nan,df.mean(),inplace=True)
#data.dropna(how='all',inplace=True)


air=df.drop(['Date','Time'],axis=1)
Y=air['RH'] #####output
X=air.drop(['RH'],axis=1)   #but drop is a function so use () and axis
Y=Y.astype('int')
#print(Y)
#print(X.shape)
#print(Y.shape)
#####################################################
####split data set for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
#print("Training data size",X_train.shape)
#print("Testing data size",X_test.shape)
#####################################################
#########    LINEAR   REGRESSION   #################
#####################################################
def l_reg():
    from sklearn.linear_model import LinearRegression
    L=LinearRegression()
    ########Train the model
    L.fit(X_train,Y_train)
    ######test the model by making prediction
    Y_pred=L.predict(X_test)
    from sklearn.metrics import mean_squared_error
    rmse_linearreg=np.sqrt(mean_squared_error(Y_test,Y_pred))
    #print("Root mean squared error in Linear regressionis",mse_linearreg)
    m.showinfo(title="Linear Regression",message="root mean squared error is"+str(round(rmse_linearreg,2)))
#####################################################
#########    SUPPORT  VECTOR  MACHINE   #################
#####################################################
def svm():
    from sklearn.svm import SVR #import support vector regressor
    svm=SVR(gamma='scale')
    svm.fit(X_train,Y_train)
    Y_pred=svm.predict(X_test)
    from sklearn.metrics import mean_squared_error
    rmse_svm=np.sqrt(mean_squared_error(Y_test,Y_pred))
    m.showinfo(title="SVM",message="root mean squared error in svm is"+str(round(rmse_svm,2)))
#####################################################
#########    DECISION   TREE   #################
#####################################################
def dt():
    from sklearn.tree import DecisionTreeRegressor
    D=DecisionTreeRegressor()
    ########Train the model
    D.fit(X_train,Y_train)
    ######test the model by making prediction
    Y_pred=D.predict(X_test)
    from sklearn.metrics import mean_squared_error
    rmse_dt=np.sqrt(mean_squared_error(Y_test,Y_pred))
    #print("root mean squared error in Decision Tree regressionis",rmse_dt)
    m.showinfo(title="DECISION TREE",message="root mean squared error in Decision Tree is"+str(round(rmse_dt,2)))
#####################################################
#########    RANDOM FOREST   #################
#####################################################
def rf():
    #import random forest regressor
    global r_f
    Classifier
    from sklearn.metrics import mean_squared_error
    rmse_rf=np.sqrt(mean_squared_error(Y_test,Y_pred))
    
    m.showinfo(title="SVM",message="root mean squared error in Random Forest is"+str(round(rmse_rf,2)))
def predict():
    a=float(v1.get())
    b=float(v2.get())
    c=float(v3.get())
    d=float(v4.get())
    e=float(v5.get())
    f=float(v6.get())
    g=float(v7.get())
    h=float(v8.get())
    i=float(v9.get())
    j=float(v10.get())
    k=float(v11.get())
    l=float(v12.get())
    ######predict by random forest as it is best
    rh=r_f.predict([[a,b,c,d,e,f,g,h,i,j,k,l]])
    m.showinfo(title="Relative Humidity",message="Relative Humidity is"+str(rh[0]))
    
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")
    v5.set("")
    v6.set("")
    v7.set("")
    v8.set("")
    v9.set("")
    v10.set("")
    v11.set("")
    v12.set("")
    
from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.title("Air Quality")
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
v5=StringVar()
v6=StringVar()
v7=StringVar()
v8=StringVar()
v9=StringVar()
v10=StringVar()
v11=StringVar()
v12=StringVar()
L=Label(w,relief="groove",font=('arial',30,'bold'),text="AIR QUALITY PREDICTION USING MACHINE LEARNING ",bg='cyan',fg='red',)
B1=Button(w,bg="white",command=l_reg,relief="solid",text='Linear Regression',font=('arial',20,'bold'))
B2=Button(w,bg="white",command=svm,relief="solid",text='Support Vector Machine',font=('arial',20,'bold'))
B3=Button(w,bg="white",command=rf,relief="solid",text='Random Forest',font=('arial',20,'bold'))
B4=Button(w,bg="white",command=dt,relief="solid",text='Decision Tree Regression',font=('arial',20,'bold'))
L.grid(row=1,column=1,columnspan=4)
B1.grid(row=2,column=1)
B2.grid(row=2,column=2)
B3.grid(row=2,column=3)
B4.grid(row=2,column=4)
####################################################
L1=Label(w,text='CO(GT)',font=('arial',20,'bold'))
L2=Label(w,text='PT08.S1',font=('arial',20,'bold'))
L3=Label(w,text='NMHC(GT)',font=('arial',20,'bold'))
L4=Label(w,text='C6H6(GT)',font=('arial',20,'bold'))
L5=Label(w,text='PT08.S2',font=('arial',20,'bold'))
L6=Label(w,text='NOx(GT)',font=('arial',20,'bold'))
L7=Label(w,text='PT08.S3',font=('arial',20,'bold'))
L8=Label(w,text='NO2(GT)',font=('arial',20,'bold'))
L9=Label(w,text='PT08.S4',font=('arial',20,'bold'))
L10=Label(w,text='PT08.S5(O3)',font=('arial',20,'bold'))
L11=Label(w,text='Temp',font=('arial',20,'bold'))
L12=Label(w,text='AH',font=('arial',20,'bold'))

E1=Entry(w,textvariable=v1,bg='yellow',font=('arial',20,'bold'))
E2=Entry(w,textvariable=v2,bg='yellow',font=('arial',20,'bold'))
E3=Entry(w,textvariable=v3,bg='yellow',font=('arial',20,'bold'))
E4=Entry(w,textvariable=v4,bg='yellow',font=('arial',20,'bold'))
E5=Entry(w,textvariable=v5,bg='yellow',font=('arial',20,'bold'))
E6=Entry(w,textvariable=v6,bg='yellow',font=('arial',20,'bold'))
E7=Entry(w,textvariable=v7,bg='yellow',font=('arial',20,'bold'))
E8=Entry(w,textvariable=v8,bg='yellow',font=('arial',20,'bold'))
E9=Entry(w,textvariable=v9,bg='yellow',font=('arial',20,'bold'))
E10=Entry(w,textvariable=v10,bg='yellow',font=('arial',20,'bold'))
E11=Entry(w,textvariable=v11,bg='yellow',font=('arial',20,'bold'))
E12=Entry(w,textvariable=v12,bg='yellow',font=('arial',20,'bold'))

L1.grid(row=3,column=1)
E1.grid(row=3,column=2)
L2.grid(row=3,column=3)
E2.grid(row=3,column=4)
L3.grid(row=4,column=1)
E3.grid(row=4,column=2)
L4.grid(row=4,column=3)
E4.grid(row=4,column=4)
#######################################
L5.grid(row=5,column=1)
E5.grid(row=5,column=2)
L6.grid(row=5,column=3)
E6.grid(row=5,column=4)
L7.grid(row=6,column=1)
E7.grid(row=6,column=2)
L8.grid(row=6,column=3)
E8.grid(row=6,column=4)
###################################################
L9.grid(row=7,column=1)
E9.grid(row=7,column=2)
L10.grid(row=7,column=3)
E10.grid(row=7,column=4)
L11.grid(row=8,column=1)
E11.grid(row=8,column=2)
L12.grid(row=8,column=3)
E12.grid(row=8,column=4)
########################################
B5=Button(w,relief="solid",bg='white',fg="blue",text="SUBMIT",font=('arial',20,'bold'),command=predict)
B6=Button(w,relief="solid",bg='white',fg='blue',text="RESET",font=('arial',20,'bold'),command=reset)
B5.grid(row=9,column=2)
B6.grid(row=9,column=4)
##############################################
Lend=Label(w,relief="solid",font=('arial',30,'bold'),text="           RELATIVE HUMIDITY PREDICTION          ",bg='cyan',fg='red',)
Lend.grid(row=10,column=1,columnspan=4)
w.mainloop()


