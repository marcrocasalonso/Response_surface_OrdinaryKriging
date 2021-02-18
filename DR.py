import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel
from scipy.linalg import eigh
from scipy import exp
from sklearn import manifold

def dimreduction(X,method='PCA',parameters=[1,'gaussian',3,0.9]):
    
    
    '''
    INPUTS:
    
    X=input matrix to reduce dimension
    method= techniques for DR 'PCA or 'kPCA'
    parameters= [gamma of KPCA,type of kernel,degree_kernel,percentage of information]
    
    OUTPUTS:
    z=reduced space 
    Eigenvalues
    Eigenvectors
    
    '''
    
    n_components=1 #invariant for the method
    gamma=parameters[0]
    percentage=parameters[3]
    
    if method=='PCA':
    
        '''
        Dimensionality reduction Principal Component analysis (PCA)
        ''' 
        print('Dimensionality reduction PCA launched:')
        
        ##centramos las muestras
        #print(X.mean(axis=1))
        X=X.sub(X.mean(axis=1), axis=0)
        #print('X',X)
        
        U,S,VT=np.linalg.svd(X,full_matrices=True)
        '''
        print('U',U.shape,U)
        print('S',S.shape,S)
        print('VT',VT.shape,VT)
        '''

        U_transpose=np.transpose(U)
        z=U_transpose.dot(X)
                      
        #### Percentage of the eigenvalues
        Z1_perc=0
        eigval=S**2#autovalores
        suma=sum(eigval)#
        suma_eig=sum(eigval[0:n_components])*100
        for i in z[0]:
            if percentage>Z1_perc:
                suma_eig=sum(eigval[0:n_components])*100
                Z1_perc=(suma_eig)/suma #Porcentaje del primer autovalor=[0:1];;;;;;Porcentaje hasta el 4rt autovalor=[0:4]
                print('Percentage of the first {} Eigenvalues:'.format(n_components),Z1_perc)
                Z1_perc=Z1_perc/100
                n_components=n_components+1
        
        eigvecs=U[:,:n_components-1] 
        z=z[:n_components-1,:]
        
        
      
    elif method=='KPCA':
    
        '''
        Dimensionality reduction Kernel Principal Component analysis (KPCA)
        ''' 
        
        print('Dimensionality reduction KPCA launched:')
                
        ##centramos las muestras
        X=X.sub(X.mean(axis=1), axis=0)
        X=np.transpose(X)
        #print(X)
           
        #Computing the dxd kernel matrix----> C2=K=exp(-gamma ||x1-xj||**2)
        
        if parameters[1]=='gaussian':
            # squared euclidean distance for every pair of samples x
            sq_dists=pdist(X,'sqeuclidean') 
            #converting the pairwise distances into a symmetric dxd matrix
            mat_sq_dists=squareform(sq_dists)
            K=exp(-gamma*(mat_sq_dists))#gaussian kernel
        
        if parameters[1]=='polynomial':
            degree=parameters[2]
            K=polynomial_kernel(X, Y=None, degree=degree, gamma=gamma)
            
        if parameters[1]=='triangular':
            dists=pdist(X,'euclidean') 
            mat_dists=squareform(dists)
            K=1-(gamma*mat_dists)
        
        if parameters[1]=='sigmoid':
            K=np.tanh(polynomial_kernel(X, Y=None, degree=1, gamma=gamma))
            
        #print('Matrix K',K.shape,K)
        #np.savetxt('Kernel_matrx.csv', K) 
    
        #centering the symmetric NsxNs kernel matrix
        ns=K.shape[0]
        one_ns=np.ones((ns,ns))/ns
        K=K-one_ns.dot(K) -K.dot(one_ns) +one_ns.dot(K).dot(one_ns)
        #print('lenght',len(K))
        #print('Normalized Matrix K=C2:',K.shape,K)
    
        #Obtaining eigenvalues in descending order with 
        #corresponding eigenvectors from the symmetric matrix K
        eigvals,eigvecs=eigh(K)
        
        #print('Singular Values:',eigvals)
        #print('')
        #print('Eigenvectors:',eigvecs)
        
        #obtaining the i eigenvectors that corresponds to the i highest eigenvalues
        X_pc=np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
        lambdas=[eigvals[-i] for i in range(1,n_components+1)]
        #return X_pc, lambdas, K
                
        #SVD implmentation for K matrix
        U,S,VT=np.linalg.svd(K,full_matrices=True)
        z=VT.dot(K)#reduced space or latent space
        #print('Singular values',S)
        
        
        Z1_perc=0
        eigval=S#autovalores
        suma=sum(eigval)#
        suma_eig=sum(eigval[0:n_components])*100
        perc_eigenval=[]
        for i in z[0]:
            if percentage>Z1_perc:
                suma_eig=sum(eigval[0:n_components])*100
                Z1_perc=(suma_eig)/suma #Porcentaje del primer autovalor=[0:1];;;;;;Porcentaje hasta el 4rt autovalor=[0:4]
                print('Percentage of the first {} Eigenvalues:'.format(n_components),Z1_perc)
                perc_eigenval.append(Z1_perc)
                Z1_perc=Z1_perc/100
                n_components=n_components+1
        #reduced space
        latent_space=VT.dot(K)               
        z=latent_space[:n_components-1,:]
    
    
    else:
        print('Warning: Invalid method')

    
    return(z,eigval,eigvecs,perc_eigenval)

def backward_PCA(X,z,eigenvecs):

    '''
    X=input matrix (dxns)
    z=z samples obtained with a metamodel
    eigenvecs=eigenvector matrix obtained with PCA
    
    '''
    
    
    X_backward=eigenvecs.dot(z)
    #we add the mean because the samples are centered previously
    x_b=pd.DataFrame(X_backward)#we convert numpy matrix to pandas matrix
    X_backward=x_b.add(X.mean(axis=1), axis=0)
    #print('X backward:',X_backward)
    
    return (X_backward)
     
def backward_KPCA(X,z,z_metamodel,eigenvecs,n_dim):
            
    X_back=X
    #points=([0,0.1,0.3]) #punto de z1 que queremos aproximar su QoI correspondiente
    #vector de valores de Z1 que se supone que viene de evaluar una superficie de respuesta como PGD
    #print('New values of Z1 created with a metamodel',points)
    z=z[:n_dim,:]
    ns=len(z[0])
    input_matrix_preimage=[]
    for k in range(0,len(z_metamodel)):
        dist=[]
        for i in range(0,ns):
            dist.append(np.linalg.norm(z_metamodel[k,:]-z[:,i]))#valor absoluto entre puntos. si z1 fuera un vector de dimension superior a 1 hay que hacer la norma. Hacer lo que hace edu en su memoria y berto en su paper sobre KPCA
        #print(dist)
        suma=np.zeros((ns,1))
        for i in range(0,ns):
            suma[i]=1/(dist[i])**2
        suma_tot=sum(suma)
        w=np.zeros((ns,1))
        for i in range(0,ns):
                w[i]=(1/(dist[i])**2)/suma_tot
        
        #print(w)
        pp=[]
        for i in range(0,ns):
            pp.append(w[i]*X_back[:,i])#la matriz de 142 filas por 2366 columnas volviendo atras
        pp=sum(pp)
        tra=np.transpose(pp)
        input_matrix_preimage.append(tra)#matrix of 3000 x 329 
        #print('weights Z1->X->QoI',w)
        print(k)
    
    
    input_matrix_preimage=np.asarray(input_matrix_preimage)
    X_backward=np.transpose(input_matrix_preimage)
    
    #print('Input Matrix X for KPCA:',X_back.shape,X_back)
    #print('Input matrix X by going back with weighting distance technique',X_backward.shape,X_backward)

        

    return (X_backward)

'''
--------------------------------------------
Example for a dimensionality reduction case:
--------------------------------------------
'''
'''
#Example 1 PCA

#Dimensionality reduction
X_input = pd.read_csv('X_test_10.csv', header=None,delimiter=' ')
z,eigenval,eigenvec=dimreduction(X_input,'PCA',[0.1,'polynomial',3,0.80])
print('Reduced space',z.shape,z)

#backward
z_metamodel=([[0.12,0.10],[0.10,0.10]])#backward of one sample generated with a metamodel of dimension 2 because we need z1 and z2 to obtain 80% of information
X_backward=backward_PCA(X_input,z_metamodel,eigenvec)
print('X backward',X_backward.shape,X_backward)


#-----------------

#Example 2 KPCA

#Dimensionality reduction
X_input = pd.read_csv('X_test_10.csv', header=None,delimiter=' ')
print('Input matrix X:',X_input.shape,'\n',X_input)
z,eigenval,eigenvec=dimreduction(X_input,'KPCA',[0.1,'polynomial',3,0.92])
print('Reduced space:',z.shape,'\n',z)

#backward
z_metamodel=([-0.0073,0.008],[-0.121,-0.01],[-0.078,-0.007])#backward of three samples generated with a metamodel of dimension 2 because we need z1 and z2 to obtain 99% of information
print('New values of Z1 created with a metamodel:\n',z_metamodel)
#z_metamodel=([-0.0073,0.0083],[-0.121,-0.012],[-0.078,-0.0077])#backward of three samples generated with a metamodel of dimension 2 because we need z1 and z2 to obtain 99% of information
X_backward=backward_KPCA(X_input,z,z_metamodel,eigenvec)
print('X backward:',X_backward.shape,'\n',X_backward)

'''



