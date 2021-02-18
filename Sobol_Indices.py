import numpy as np
from itertools import repeat
from itertools import combinations
import math as m
import pandas as pd

def saltelli_matrix(A_saltelli,B_saltelli):
    
    #saltelli matrices A and B
    n=len(A_saltelli[0])
    A_B_saltelli=np.c_[[A_saltelli,B_saltelli]]
           
    #Saltelli matrices AB
    AB_saltelli=[[] for i in repeat (None,n)] 
    
    for i in range (0,n):
        ini=A_B_saltelli[0][:,0]
        for j in range(1,n):
            AB_saltelli[i]=np.c_[ini,A_B_saltelli[0][:,j]]
            ini=AB_saltelli[i]
    AB_saltelli=np.asarray(AB_saltelli)
    
    #hacemos que B_saltelli este en las posiciones de la diagonal principal
    for i in range(0,n):
        AB_saltelli[i][:,i]=A_B_saltelli[1][:,i]

    return(AB_saltelli)  

    
def FirstSobolIndices(A,B,AB,n,z):
    N=len(A)     
    d=[[] for i in repeat (None,n)]
     
    for i in range(0,n):
        for j in range (0,N):
            d[i].append(B[j]*(AB[i][j]-A[j]))
    
    var=[[] for i in repeat (None,n)]
    for i in range(0,n):
        var[i].append(sum(d[i])/N)
        
    var_tot=np.var(z)
         
    S=[[] for i in repeat (None,n)]
    for i in range(0,n):
        S[i].append(abs(var[i]/var_tot))
    S=np.asarray(S)
    S=np.reshape(S,(n))
    S= pd.DataFrame(S)
    
    return (S)


def TotalSobolIndices(A,B,AB,n,z):
    N=len(A)     
    d=[[] for i in repeat (None,n)]
    
    for i in range(0,n):
        for j in range (0,N):
            d[i].append((A[j]-AB[i][j])**2)
            
    var=[[] for i in repeat (None,n)]
    for i in range(0,n):
        var[i].append(sum(d[i])/(2*N))  
    
    var_tot=np.var(z)
         
    ST=[[] for i in repeat (None,n)]
    for i in range(0,n):
        ST[i].append(abs(var[i]/var_tot))
    ST=np.asarray(ST)
    ST=np.reshape(ST,(n))
    ST= pd.DataFrame(ST) 
    
    return (ST)     


def SecondSobolIndices(A,B,AB,n,z,ST):
    N=len(A)
    n_d=m.factorial(n)/(m.factorial(2)*(m.factorial(n-2)))#numero de variables d para almacenar los indices
    n_d=int(n_d)
    index = list(combinations(range(0,n),2))#lista con los indices a evaluar
    indx=np.asarray(index)
        
    d=[[] for i in repeat (None,n_d)]
    for i in range(0,n_d):
        for j in range(0,N):
            d[i].append((AB[indx[i][0]][j]-AB[indx[i][1]][j])**2)
    
    var=[[] for i in repeat (None,n_d)]
    for i in range(0,n_d):
        var[i].append(sum(d[i])/(2*N)) 
        
    var_tot=np.var(z)
    S2=[[] for i in repeat (None,n_d)]
    for i in range(0,n_d):
        S2[i].append(var[i]/var_tot) 
        
    S_2=[[] for i in repeat (None,n_d)]
    for i in range(0,n_d):
        S_2[i].append(ST[0][indx[i][0]]+ST[0][indx[i][1]]-S2[i]) 
    S_2=np.asarray(S_2)
    S_2=np.reshape(S_2,n_d) 
    S_2= pd.DataFrame(S_2,index=index)  
       
    return(S_2)

