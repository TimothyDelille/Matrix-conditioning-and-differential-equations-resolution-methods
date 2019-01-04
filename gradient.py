"""
Methodes de gradients.
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import _pickle as cPickle
import scipy.linalg as spl

def function(A,b,c,xx):
    shape=xx[0].shape
    ZZ=np.zeros_like(xx[0])
    for item in itertools.product(*map(range,shape)):
        vect=list()
        for elem in xx:
            vect.append(elem[item])
        vec=np.array(vect,ndmin=1)
        ZZ[item]=0.5*np.dot(vec.T,A.dot(vec))-np.dot(vec.T,b) + c
    return ZZ

def gradient(A,b,x):
    return A.dot(x)-b

def pas_fixe(x0,fonction,pas=1.0e-2,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']

    xx=[x0]
    direction=-gradient(A,b,xx[-1])
    residu=[np.linalg.norm(gradient(A,b,xx[-1]))]
    
    k=0
    while residu[-1] >= tol and k<=itermax:
        k+=1
        #----- x(k+1) -----
        xx.append(xx[-1]+pas*direction)

        #----- nouvelle direction de descente d(x+1) -----
        direction = -gradient(A,b,xx[-1])
        
        #----- residu r(k+1) -----
        residu.append(np.linalg.norm(gradient(A,b,xx[-1])))
    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def conjugate(x0,fonction,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']

    #***** Initialisation *****
    xx=[x0]
    direction=-gradient(A,b,xx[-1])
    residu=[np.linalg.norm(gradient(A,b,xx[-1]))]

    k=0
    while residu[-1] >= tol and k<=itermax:
        k+=1
        #----- ho(k) -----
        rho = -np.dot(gradient(A,b,xx[-1]).T,direction)/np.vdot(np.dot(A,direction),direction)
        beta=np.vdot(np.dot(A,gradient(A,b,xx[-1])),direction)/np.vdot(np.dot(A,direction),direction)

        #----- x(k+1) -----
        xx.append(xx[-1]+rho*direction)
        
        #----- nouvelle direction de descente d(x+1) -----
        direction = -gradient(A,b,xx[-1])+beta*direction

        #----- residu r(k+1) -----
        residu.append(np.linalg.norm(gradient(A,b,xx[-1])))

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def plot(xx,res):
    plt.figure()
    
    X1, X2 = np.meshgrid(np.linspace(-5.0,5.0,101),np.linspace(-5.0,5.0,101))
    Z=function(A,b,c,[X1,X2])
    
    plt.contour(X1,X2,Z)

    plt.plot(xx.T[0],xx.T[1],'k-x')

    plt.axes().set_aspect('equal')
    
    plt.figure()
    plt.plot(res)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel(r'$||\nabla f(x) ||_2$')
    plt.title('Convergence')

    plt.show()

if __name__=="__main__":

    A=np.array([[16,0],
                [0,16]])
    b=np.array([1634,
                 -1914])
    x0=np.array([1,1])
    c=np.array([980054])

    fonction={'A':A, 'b':b, 'c':c}

    #----- Pas fixe -----
    cas_01=pas_fixe(x0,fonction,1.0e-1,1.0e-6,10000)
    print(cas_01['xx'][-1], len(cas_01['xx'])-1)

    plot(cas_01['xx'],cas_01['residu'])

    #----- Gradient conjugue -----
    cas_02=conjugate(x0,fonction,1.0e-6,10000)
    print(cas_02['xx'][-1], len(cas_02['xx'])-1)

    plot(cas_02['xx'],cas_02['residu'])

    
