#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:01:30 2018

@author: timothydelille
"""

import numpy as np
from numpy import linalg as LA

#-----Wilson matrix conditioning-----

print("Conditionnement de la matrice de Wilson")
A=np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]])
C=np.zeros([4,4])
print(C)

b=[32,23,33,31]
c=[32.1,22.9,33.1,30.9]

x=LA.solve(A,b)
y=LA.linalg.solve(A,c)

#Calcul de la matrice inverse
B=LA.inv(A)

print(x)
print(y)
print("ecart de la solution:",x-y)

print("Calcul du conditionnement 1")

def Norme1(A):
    new=0
    for i in range(len(A[1])):
        somme=0
        for j in range(len(A[0])):
            somme+=abs(A[j,i])
        if somme>new:
            new=somme
    return new
print("Norme de A pour p=1:",Norme1(A))

def cond1(A):
    inverseA=LA.inv(A)
    return Norme1(A)*Norme1(inverseA)

print(cond1(A))
print("Calcul du conditionnement 2")

def Norme2(A):
    tA=np.transpose(A)
    norme=max(np.abs(LA.eigvals(np.dot(tA,A))))
    return np.sqrt(norme)

print("Norme de A pour p=2:",Norme2(A))

def cond2(A):
    inverseA=LA.inv(A)
    return Norme2(A)*Norme2(inverseA)

print(cond2(A))

def Hilbert(n):
    return [[1/(i+j-1) for j in range(1,n+1)] for i in range(1,n+1)]

# Calcul du conditionnement de la matrice d'Hilbert 
print("Conditionnement de la matrice de Hilbert") 
n=10
H=Hilbert(n)
#Creation d'un vecteur de dimension n avec uniquement des 1 
x=np.ones(n)
#Creation du vecteur b=H.x
b=np.dot(H,x)
#Resolution du systeme lineaire H.X=b 
X=LA.solve(H,b)
# normalement x=X ???
print(x-X)

print(LA.eigvals(H)) #valeur propre très proche de zéro -> presque pas inversible

#-----Jacobi iterative resolution method-----

def Jacobi(A,b,niter):
    eps=0.000001
    x=np.zeros(n)
    k=1
    while k<=niter and LA.norm(b-np.dot(A,x))/LA.norm(b)>eps:
        for i in range(n):
            if np.abs(A[i,i])>0.0:
                s=0.0
                for j in range(n):
                    if j!=i:
                        s=s+A[i,j]*x[j]
                x[i]=(b[i]-s)/A[i,i]
        k=k+1
    print("nombre itérations",k-1)
    return x

A=np.array([[6,1,-1],[0,-4,2],[1,0,3]])
b=np.array([[1],[1],[1]])

print(Jacobi(A,b,5))

