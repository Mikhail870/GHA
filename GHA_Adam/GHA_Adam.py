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
epoch=2
n=20_000
NMSE=np.zeros((k,epoch))
nmse=np.zeros(n*epoch)
count=0


############## generate n dimensional gaussian distribution ###################
matrix=np.random.multivariate_normal(np.zeros(m),matrix_o,n)
matrix=matrix-np.mean(matrix,axis=0)



############## solve eigvals ###################
eigval=np.linalg.eigvals(matrix_o)
eigval_sorted=np.sort(eigval)[::-1]
print(f"array of eigvals {eigval_sorted}")


############## Adam parameters ###################
alpha=1e-3
betha_1=9e-1
betha_2=999e-3
eps=1e-8
m_t=np.zeros_like(weigth)
u_t=np.zeros_like(weigth)
t=0 # counter for Adam


############## main GHA + Adam ###################
for e in range(epoch):
    for i in range(n):
      x=matrix[i]
      y=np.einsum('ij,j->i',weigth , x)
      # Adam
      t+=1
      g_t=(np.einsum('i,j->ij', y, x)-np.einsum('j,k,ki,jk->ji', y, y, weigth, mask))
      m_t=betha_1*m_t+(1-betha_1)*g_t
      u_t=betha_2*u_t+(1-betha_2)*g_t**2
      m_est=m_t/(1-betha_1**t)
      u_est=u_t/(1-betha_2**t)
      update_param=alpha*m_est/(np.sqrt(u_est)+eps)
      # Update weights
      weigth=weigth+update_param
   
      
    projections=np.dot(weigth,matrix.T)
    lambdas=np.var(projections,axis=1,ddof=1)
    for index in range(k):
      NMSE[index,e]=((eigval_sorted[index]-lambdas[index])**2)/(eigval_sorted[index]**2)






############## plot NMSE/epoch & eigvalues/epoch ###################
for i in range(k):
    plt.plot(range(epoch),10*np.log10(NMSE[i,:]),label=f'NMSE for {i+1} eigvalue')
    plt.xlabel(f'epoch {epoch}')
    plt.ylabel('NMSE')
    plt.title(f'GHA + Adam NMSE vs epoch epoch={epoch}, alpha={alpha}, betha_1={betha_1}, betha_2={betha_2}')
    plt.legend()

plt.show()


for i in range(k):
    plt.plot(range(epoch),np.ones(epoch)*lambdas[i],label=f'estimate {i+1} eigvalue={lambdas[i]}')
    plt.plot(range(epoch),np.ones(epoch)*eigval_sorted[i],label=f'true {i+1} eigvalue={eigval_sorted[i]}')
    plt.xlabel(f'epoch {epoch}')
    plt.ylabel('eigvalue')
    plt.title(f'GHA + Adam estimate eigvalues vs epoch epoch={epoch}, alpha={alpha}, betha_1={betha_1}, betha_2={betha_2}')
    plt.legend()

plt.show()
      

