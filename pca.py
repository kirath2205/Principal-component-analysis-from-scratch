import numpy as np
import scipy.linalg
from scipy.io import loadmat
from numpy import linalg as lg
from scipy.linalg import eigh
import matplotlib.pyplot
def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = dataset['fea']
    n = len(x)
    d = len(x[0])
    k=np.array(x,dtype='float64')
    k=k- np.mean(k, axis=0)
    return k

def get_covariance(dataset):
    x=np.array(dataset)
    k=np.dot(np.transpose(x), x)
    k=k/(len(dataset)-1)
    return k

def get_eig(S,m):
    k0,k1=scipy.linalg.eigh(S, eigvals=(len(S)-m, len(S)-1))
    list1=list()
    Lambda=list()
    for i in range(m):
        for j in range(m):
            list1.append(0)
        Lambda.append(list1.copy())
        list1.clear()
    k0.sort()
    for i in range(m):
        for j in range(m):
            if(i==j):
                Lambda[i][j]=k0[len(k0)-i-1]
    
    k1=np.fliplr(k1)
    list1=list()
    vectors=list()
    for i in range(len(k1)):
        for j in range(m):
            list1.append(k1[i][j])
        vectors.append(list1.copy())
        list1.clear()

    
    return Lambda,vectors
def project_image(image,U):
    U = np.transpose(np.asarray(U))
    aij=np.dot(U[0],image)
    proj = np.dot(aij, np.transpose(U[0]))
    return proj
     
def display_image(orig,proj):
 
    orig = np.reshape(orig,[32,32])
    proj = np.reshape(proj,[32,32])
    orig=np.transpose(orig)
    proj=np.transpose(proj)
    fig, axs = matplotlib.pyplot.subplots(1, 2, constrained_layout=True)
    k1=axs[0].imshow(orig,aspect='equal')
    axs[0].set_title('Original')
    fig.colorbar(k1, ax=axs[0],shrink=0.46)
    k2=axs[1].imshow(proj,aspect='equal')
    axs[1].set_title('Projection')
    fig.colorbar(k2, ax=axs[1],shrink=0.46)
    matplotlib.pyplot.savefig('image.png')
    matplotlib.pyplot.show(axs[0])
    
S=get_covariance(load_and_center_dataset('YaleB_32x32.mat'))
Lambda,U=get_eig(S,2)

display_image(load_and_center_dataset('YaleB_32x32.mat')[769],project_image(load_and_center_dataset('YaleB_32x32.mat')[769],U))