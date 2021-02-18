import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import cdist

def variogram_model(m, d): 
     #Spherical model, m contains the values of parameters and type of variogram: ['spherical',psill, range, nugget]
     typevariogram=m[3]
     if typevariogram=='spherical':
        psill = float(m[0]) 
        range_ = float(m[1]) 
        nugget = float(m[2]) 
        return np.piecewise(d, [d <= range_, d > range_],[lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget]) 

     #linear_variogram_model(m,d): 
     if typevariogram=='linear':
        range_ = float(m[1])
        nugget = float(m[2]) 
        s=10#pendiente
        return (s*d+nugget) 
        
     if typevariogram=='gaussian':
        psill = float(m[0]) 
        range_ = float(m[1]) 
        nugget = float(m[2]) 
        return (np.piecewise(d, [d <= range_],[lambda x:nugget+psill*(1-np.exp((-3*(x)**2)/(range_**2)))]))

def invers_matrix_OK(semi_variogram):
    #forzamos que la diagonal tenga valores 0 (para la interpolacion). Significa que en los propios puntos el valor es el propio al del punto hace que pase por el punto la interpolacion
    np.fill_diagonal(semi_variogram,0) 
    # introduciomos los vecotres de 1 en ultima fila i ultima columna i valor de 0 en la ultima casilla de la matriz
    n=len(semi_variogram[0]+1)
    a=np.zeros((n+1,n+1))
    a[:n,:n]=semi_variogram
    a[n,:]=1.0
    a[:,n]=1.0
    a[n,n]=0.0
    #print('A=Kriging Matrix=',a)
    a_inv=inv(a)
    return(a_inv)


def f_OrdinaryKriging(X,pointsxy,z,parameters,a_inv):#kriging model
    #calculomos el variograma para el nuevo punto
    #print(i)
    n=len(X)
    X_ini=X[0]
    for i in range(1,n):#creamos un vector con los inputs nuevos que queremos evaluar
        new_points=np.c_[X_ini,X[i]]
        X_ini=new_points
    
    b_distance=cdist(pointsxy,new_points)#calculamos la ditancia de los puntos de muestra al nuevo punto que queremos aproximar
    b_variogram=variogram_model(parameters,b_distance)
    
    n=len(pointsxy[0]+1)
    b=np.zeros((n+1,1))
    b[:n]=b_variogram
    b[n]=1.0
    #print('b',b)
    w=np.dot(a_inv,b) #pesos
    #print('Weights',w)
    #calculamos en valor del nuevo punto mediante algoritmo de kirigng
    z2=np.sum(w[:n,0]*z)
    return(z2)
    
    
    
