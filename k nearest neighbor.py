import numpy as np

def calculateDistance(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

def getClosest(trainDataMatrix,trainlabelMatrix,x,topK):
    distanceList=[0]*len(trainlabelMatrix)
    for index in len(trainDataMatrix):
        distanceList[index]=calculateDistance(trainDataMatrix[index],x)
    topKList=np.argsort(np.array(distanceList))[:topK]
    labelList=[0]*topK
    for index in topKList:
        labelList[int(trainlabelMatrix[index])]+=1
    return labelList.index(max(labelList))