import math
import numpy as np

def calculateGauss(dataSetArray,averageValue,sigmod):
    return 1/(math.sqrt(2*math.pi)*sigmod)*np.exp(-(dataSetArray-averageValue)**2/(2*sigmod**2))

def E_step(dataSetArray,alpha0,averageValue0,sigmod0,alpha1,averageValue1,sigmod1):
    gauss0=calculateGauss(dataSetArray,averageValue0,sigmod0)
    gauss1=calculateGauss(dataSetArray,averageValue1,sigmod1)


def M_step():

def EM_train():
