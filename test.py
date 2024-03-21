import numpy as np
from corey import coreyWater, coreyOil
import matplotlib.pyplot as plt

relpermWater = coreyWater(2.0,0.4,0.2,0.2)
print(relpermWater)
Ncells = 8
matrixJ = np.zeros((2*Ncells,2*Ncells))


matrixJ[0,0] = 1
matrixJ[0,1] = 1
matrixJ[1,0] = 1
matrixJ[1,1] = 1
matrixJ[0,2] = 2
matrixJ[0,3] = 2
matrixJ[1,2] = 2
matrixJ[1,3] = 2

for ii in np.arange(2,2*Ncells-2,2):
    print(ii)
    matrixJ[ii,ii-2] = ii
    matrixJ[ii,ii-1] = ii
    matrixJ[ii+1,ii-2] = ii
    matrixJ[ii+1,ii-1] = ii
    
    matrixJ[ii,ii] = ii+1
    matrixJ[ii,ii+1] = ii+1
    matrixJ[ii+1,ii] =ii+1
    matrixJ[ii+1,ii+1] =ii+1

    matrixJ[ii,ii+2] = ii+2
    matrixJ[ii,ii+3] = ii+2
    matrixJ[ii+1,ii+2] = ii+2
    matrixJ[ii+1,ii+3] = ii+2
    
matrixJ[-2,-4] = Ncells -1
matrixJ[-2,-3] = Ncells -1
matrixJ[-1,-4] = Ncells -1
matrixJ[-1,-3] = Ncells -1

matrixJ[-2,-2] = Ncells
matrixJ[-2,-1] = Ncells
matrixJ[-1,-2] = Ncells
matrixJ[-1,-1] = Ncells
print(matrixJ)
print(np.linalg.det(matrixJ))

pressure = 1.0E7*np.ones(Ncells)
saturation = 0.2*np.ones(Ncells)
rightpressure = 2000
vectorX =  np.zeros(2*Ncells)
vectorX[::2] =pressure
vectorX[1::2] = saturation
vectorX[-2] = rightpressure
print('vectorX -', vectorX)

Residual = np.dot(matrixJ,vectorX)
print('Residual - ', Residual)