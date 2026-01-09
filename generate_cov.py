import numpy as np

def generate_cov(min_l :int,max_l :int):

    '''Generation of the covariance matrix from given eigenvalues'''
    
    eigval=np.diag(np.arange(min_l,max_l))

    U=np.random.normal(0,1,eigval.shape)

    U,_=np.linalg.qr(U)

    return U @ eigval @ U.T # подходит для вещественных значений


if '__main__' == __name__:

   cov=generate_cov(0,100)
   print(np.linalg.eig(cov)[0])