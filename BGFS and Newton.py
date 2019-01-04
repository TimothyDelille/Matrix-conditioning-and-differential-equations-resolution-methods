"""
Script TP descente de Newton.
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import scipy.optimize as spo
import sys
import matplotlib.pyplot as plt

################################
##### FUNCTION DEFINITIONS #####
################################
#***** Fonctions test *****
#----- Fonction quadratique -----

def func_quad(xx):
    yy=xx.flatten()
    return np.asarray(2.0*yy[0]**2.0+yy[0]*yy[1]+yy[1]**2.0+yy[0]+yy[1])

def grad_quad(xx):
    yy=xx.flatten()
    return np.asarray([[4.0*yy[0]+yy[1]+1.0],[2.0*yy[1]+xx[0]+1.0]])

def hess_quad(xx):
    yy=xx.flatten()
    return np.asarray([[4.0,1.0],[1.0,2.0]])

#----- Fonction de Rosenbrock -----
def func_rosen(xx):
    yy=xx.flatten()
    return spo.rosen(yy)

def grad_rosen(xx):
    yy=xx.flatten()
    gra=spo.rosen_der(yy)
    return gra.reshape((gra.shape[0],1))

def hess_rosen(xx):
    yy=xx.flatten()
    return spo.rosen_hess(yy)

#***** Algorithmes *****
#----- Algorithme de calcul du pas -----
def compute_rho_Newton():
    
    return 1.0

def compute_rho_BFGS(xx,dire,func,tol,intervalle):
    """
    Algorithme de la section doree.

    xx : le point courant, numpy array
    dire : la direction courante, numpy array
    func : la fonctionnelle a minimiser, fonction python
    tol : une tolerance pour l'algorithme de la section doree, float
    intervalle : borne sup pour [0.0;borne_sup], float

    Retourne la valeur du pas, float.
    """
    k = 0
    tau = (1.+np.sqrt(5))/2
    a = 0.
    b = intervalle
    while  np.abs(b-a)>= tol and k<=10000 :
        ap = a + (b-a)/(tau**2)
        bp = a + (b-a)/tau
        if func(xx+dire*ap)<func(xx+dire*bp):
            b = bp
        elif func(xx+dire*ap)>func(xx+dire*bp):
            a = ap
        else:
            a = ap
            b = bp
        k = k+1
    rho = (a+b)/2

    return rho

#----- Algorithme de calcul de la direction de descente -----
def compute_dire_Newton(xx,grad,hess):
    
    dire = np.dot(-np.linalg.inv(hess(xx)),grad(xx))
    
    return dire

def compute_dire_BFGS(xx,rho,grad,dire,Hk):
    """
    Calcul de la direction de descente par BFGS.

    xx : le point courant, numpy array
    rho : le pas de descente, float
    grad : la fonction gradient, fonction python
    dire : la direction courante, numpy array
    Hk : la matrice BFGS, numpy array

    Retourne la direction de descente, numpy array.
    Retourn la nouvelle matrice Hk, numpy array.
    """
    deltax = rho*dire
    deltag = grad(xx)-grad(xx-rho*dire)
    
    a = 1.0 + (deltag.T.dot(Hk.dot(deltag)))/(deltax.T.dot(deltag))
    b = deltax.dot(deltax.T)/(deltax.T.dot(deltag))
    c = Hk.dot(deltag.dot(deltax.T))+ (Hk.dot(deltag.dot(deltax.T))).T
    d = deltax.T.dot(deltag)
    
    Uk = a*b - c/d
    Hk = Hk + Uk
    
    dire = - Hk.dot(grad(xx))
    
    return dire,Hk
    
#***** Fonction pour sortie graphique *****
def plot(xx_list,residu_list,function):
    """
    Sortie des resultats.
    """
    #***** Plot results *****
    plt.figure()

    #***** Calcul des coordonnees *****
    tmp_array=np.array(xx_list,ndmin=2)[:,:,0].T
    X1_min, X1_max=tmp_array[0].min(), tmp_array[0].max()
    X2_min, X2_max=tmp_array[1].min(), tmp_array[1].max()

    X1, X2 = np.meshgrid(np.linspace(X1_min-0.1*abs(X1_max-X1_min), \
                                     X1_max+0.1*abs(X1_max-X1_min), \
                                     101), \
                         np.linspace(X2_min-0.1*abs(X2_max-X2_min), \
                                     X2_max+0.1*abs(X2_max-X2_min), \
                                     101))

    #***** Calcul de la fonction aux points de coordonnees *****
    Z=np.zeros_like(X1)
    for i in xrange(Z.shape[0]):
        for j in xrange(Z.shape[1]):
            Z[i,j]=function(np.array([[X1[i,j]],[X2[i,j]]]))

    #***** Trace de la courbe *****
    plt.contour(X1,X2,Z)

    xx_list=np.array(xx_list,ndmin=2)[:,:,0].T
    plt.plot(xx_list[0],xx_list[1],'k-x')

    #***** Plot history of convergence *****
    plt.figure()
    plt.plot(residu_list)
    plt.yscale('log')
    plt.grid()

    plt.show()

###############################
##### METHODE DE GRADIENT #####
###############################
"""
Ne rien modifier dans cette partie.
"""
from gradients import *
##### TEST 01: generic gradient #####
def func_01(xx):
    return np.asarray(2.0*xx[0]**2.0+xx[0]*xx[1]+xx[1]**2.0+xx[0]+xx[1])

def grad_01(xx):
    return np.asarray([4.0*xx[0]+xx[1]+1.0,2.0*xx[1]+xx[0]+1.0])

test_01=gene_grad()

test_01.param['function']=func_01
test_01.param['gradient']=grad_01

test_01.param['descent']['method']='conjugate'
test_01.param['step']['method']='optimal'
test_01.param['step']['optimal']['golden section']['tolerance']=1.0e-8
test_01.param['step']['optimal']['golden section']['interval']=2.0

test_01.param['guess']=np.asarray([[4.0],[-3.0]])
test_01.param['tolerance']=1.0e-8
test_01.param['itermax']=10000

print "###################"
print "##### TEST 01 #####"
print "###################"
test_01.run()
test_01.plot()

##### TEST 02: generic gradient : Rosenborck function #####
test_02=gene_grad()

test_02.param['function']=spo.rosen
test_02.param['gradient']=spo.rosen_der

test_02.param['descent']['method']='conjugate'

test_02.param['step']['method']='optimal'
test_02.param['step']['optimal']['method']='golden section'
test_02.param['step']['optimal']['golden section']['tolerance']=1.0e-8
test_02.param['step']['optimal']['golden section']['interval']=10.0

test_02.param['guess']=np.asarray([[-1.0],[-1.0]])
test_02.param['tolerance']=1.0e-10
test_02.param['itermax']=100000

print "###################"
print "##### TEST 02 #####"
print "###################"
test_02.run()
test_02.plot()

##### TEST 03: generic gradient : Rosenborck function #####
test_03=gene_grad()

test_03.param['function']=spo.rosen
test_03.param['gradient']=spo.rosen_der

test_03.param['descent']['method']='conjugate'

test_03.param['step']['method']='optimal'
test_03.param['step']['optimal']['method']='golden section'
test_03.param['step']['optimal']['golden section']['tolerance']=1.0e-8
test_03.param['step']['optimal']['golden section']['interval']=10.0

test_03.param['guess']=np.asarray([[-10.0],[-10.0]])
test_03.param['tolerance']=1.0e-10
test_03.param['itermax']=100000

print "###################"
print "##### TEST 03 #####"
print "###################"
test_03.run()
test_03.plot()

#############################
##### METHODE DE NEWTON #####
#############################
"""
ATTENTION : on prendra soin de coder les algorithmes en utilsant des tableaux en colonne
(de dimension (n,1)) pour les vecteurs.
"""
##### Cas 1 : Newton simple, fonctionnelle quadratique #####
#***** Initialisation *****
xx=np.array([[4.0],[-3.0]])
list_xx=[xx]

dire=compute_dire_Newton(xx,grad_quad,hess_quad)

list_residu=[np.linalg.norm(grad_quad(xx),2)]

kmax=10000
tol=1.0e-10
k=0
#***** Boucle *****
while np.linalg.norm(grad_quad(xx),2)>=tol and k<=kmax:
    #----- Calcul de rho(k) -----
    rho=compute_rho_Newton()

    #----- Calcul de x(k+1) -----
    xx=xx+rho*dire
    list_xx.append(xx)
    
    #----- Calcul de d(k+1) -----
    dire=compute_dire_Newton(xx,grad_quad,hess_quad)

    #----- Enregistrement du residu -----
    list_residu.append(np.linalg.norm(grad_quad(xx),2))

    #----- Avancement de k -----
    k+=1

#***** Resultats *****
print xx,k,np.linalg.norm(grad_quad(xx),2)
plot (list_xx,list_residu,func_quad)

##### Cas 2 : BFGS, fonctionnelle quadratique #####
#***** Initialisation *****
xx=np.array([[4.0],[-3.0]])
list_xx=[xx]

Hk=np.identity(2)
dire=-Hk.dot(grad_quad(xx))

list_residu=[np.linalg.norm(grad_quad(xx),2)]

kmax=10000
tol=1.0e-09
k=0
#***** Boucle *****
while np.linalg.norm(grad_quad(xx),2)>=tol and k<=kmax:
    #----- Calcul de rho(k) -----
    rho=compute_rho_BFGS(xx,dire,func_quad,1.0e-7,10.0)

    #----- Calcul de x(k+1) -----
    xx=xx+rho*dire
    list_xx.append(xx)

    #----- Calcul de d(k+1) -----
    dire,Hk=compute_dire_BFGS(xx,rho,grad_quad,dire,Hk)

    #----- Enregistrement du residu -----
    list_residu.append(np.linalg.norm(grad_quad(xx),2))

    #----- Avancement de k -----
    k+=1

#***** Resultats *****
print xx,k,np.linalg.norm(grad_quad(xx),2)
plot (list_xx,list_residu,func_quad)

