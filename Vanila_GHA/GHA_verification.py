import numpy as np
from numpy import float64
import matplotlib.pyplot as plt
from generate_cov import generate_cov

############## generate covariance matrix ###################
k=m=10
matrix_o=generate_cov(1,m+1)



############## init matrix ###################
weigth=np.random.randn(m,m)*0.01
weigth=weigth-np.mean(weigth)
mask=np.tril(np.ones((m,m)))
x=np.zeros((m,20_000),dtype=float64)
y=np.zeros((m),dtype=float64)



############## init parameters ###################
etha=1e-4 # learning rate
epoch=50
n=20_000
NMSE=np.zeros((k,epoch))


############## generate n dimensional gaussian distribution ###################
matrix=np.random.multivariate_normal(np.zeros(m),matrix_o,n)
matrix=matrix-np.mean(matrix,axis=0)



############## solve eigvals ###################
eigval=np.linalg.eigvals(matrix_o)
eigval_sorted=np.sort(eigval)[::-1]
print(f"array of eigvals {eigval_sorted}")



############## main GHA ###################
for e in range(epoch):
    for i in range(n):
      x=matrix[i]
      y=np.einsum('ij,j->i',weigth , x)
      weigth+=etha*(np.einsum('i,j->ij', y, x)-np.einsum('j,k,ki,jk->ji', y, y, weigth, mask))

    projections=np.dot(weigth,matrix.T)
    lambdas=np.var(np.dot(weigth,matrix.T),axis=1,ddof=1)

    for index in range(k):
      NMSE[index,e]=((eigval_sorted[index]-lambdas[index])**2)/(eigval_sorted[index]**2)
    



#print all NMSE
for index in range(k):
  print(f'NMSE {10*np.log10(NMSE[index,:])} dB of {index+1} eigenvalue per {epoch} epoch')


############## plot NMSE/epoch & eigvalues/epoch ###################
for i in range(k):
    plt.plot(range(epoch),10*np.log10(NMSE[i,:]),label=f'NMSE for {i+1} eigvalue')
    plt.xlabel(f'epoch {epoch}')
    plt.ylabel('NMSE')
    plt.title(f'GHA NMSE vs epoch epoch={epoch}')
    plt.legend()

plt.show()


for i in range(k):
    plt.plot(range(epoch),np.ones(epoch)*lambdas[i],label=f'estimate {i+1} eigvalue={lambdas[i]}')
    plt.plot(range(epoch),np.ones(epoch)*eigval_sorted[i],label=f'true {i+1} eigvalue={eigval_sorted[i]}')
    plt.xlabel(f'epoch {epoch}')
    plt.ylabel('eigvalue')
    plt.title(f'GHA estimate eigvalues vs epoch epoch={epoch}')
    plt.legend()

plt.show()


