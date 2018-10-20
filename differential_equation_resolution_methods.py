#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:17:32 2018

@author: timothydelille
"""

#----- Numerical methods for the resolution of differential equations -----

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Euler's Method (first order with initial condition)

def F(t,y):
    return 1+np.sin(y)
    
def Euler(F,t0,tf,y0,n) :
    t=t0
    y=y0
    h=(tf-t0)/float(n)
    temps=[t0]
    fonction=[y0]
    for i in range(n) :
        y=y+h*F(t,y)
        t=t+h
        temps.append(t)
        fonction.append(y)
    plt.plot(temps,fonction,'b')
    return fonction

# Euler's Method - Vector Form (application to a metal rod subjected to a torque c(t))
A=0.1
g=9.81
l=0.2

def G(t,theta):
    return np.array([[0,g/l],[1,0]])+np.array([[0],[A*np.sin(t)]])

Sol=Euler(G,0,1,np.array([[1],[0]]),10)
plt.show()

# Runge-Kutta 2nd Order Method (Euler-Cauchy)

def rungekutta2(a,b,phi,y0,n):
    x=np.linspace(a,b,n+1)
    y=np.empty(n+1)
    y[0]=y0
    pas=(b-a)/float(n)
    for k in range(n):
        ym=y[k]+phi(x[k],y[k])*pas/2
        pm=phi(x[k]+pas/2,ym)
        y[k+1]=y[k]+pas*pm
    return(x,y)

# Runge-Kutta 4th Order Method
    
def runge_kutta4(a,b,phi,y0,n):
    x=np.linspace(a,b,n+1)
    y=np.empty(n+1)
    y[0]=y0
    pas=(b-a)/float(n)
    for k in range(n):
        k1=phi(x[k],y[k])
        k2=phi(x[k]+pas/2.,y[k]+k1*pas/2.)
        k3=phi(x[k]+pas/2.,y[k]+k2*pas/2.)
        k4=phi(x[k]+pas,y[k]+k3*pas)
        y[k+1]=y[k]+pas*(k1+2*k2+2*k3+k4)/6
    return(x,y)
    

# Application to Van der pol Oscillator

def f(X,t):
    x,dx=X
    return[dx,mu*(1-x*x)*dx-x]
    
t=np.linspace(0,30,500)
mu=1
for v in [.001,.01,.1,1]:
    X=spi.odeint(f,[0,v],t)
    plt.figure(1)
    plt.plot(t,X[:,0])
    plt.figure(2)
    plt.plot(X[:,0],X[:,1])
    
plt.figure(1)
plt.title('Van der Pol Equation')
plt.grid()

