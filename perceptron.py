import numpy as np

def perceptron(dataArray,labelArray,time):
    dataMatrix=np.matrix(dataArray)
    labelMatrix=np.matrix(labelArray).T
    m,n=np.shape(dataArray)
    w=np.zeros((1,np.shape(dataArray)[1]))
    b=0
    h=0.1
    for k in range(time):
        for i in range(m):
            xi=dataMatrix[i]
            yi=labelMatrix[i]
            if(-1*yi*(w*xi.T+b)>=0):
                w=w+h*xi*yi
                b=b+h*yi
    return w,b

def model_test(dataArray,labelArray,w,b):
    dataMatrix=np.matrix(dataArray)
    labelMatrix=np.matrix(labelArray).T
    m,n=np.shape(dataMatrix,labelMatrix)
    errorcount=0
    for i in range(m):
        xi=dataMatrix[i]
        yi=labelMatrix[i]
        judge=-1*yi*(w*xi+b)
        if(judge>=0):
            errorcount+=1
    return 1-errorcount/m

