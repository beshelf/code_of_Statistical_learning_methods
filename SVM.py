import numpy as np
class SVM:
    def __init__(self,trainDataList,trainLabelList,sigma=10,C=200,toler=0.001):
        self.trainDataMatrix=np.mat(trainDataList)
        self.trainLabelMatrix=np.mat(trainLabelList).T
        self.sigma=sigma
        self.C=C
        self.toler=toler

        self.k=self.calculateKernel()
        self.b=0
        self.alpha=[0].self.trainDataMatrix[0]
        self.E=
        self.supportVectorIndex=[]
    def calculateKernel(self):

    def isSatisfyKKT(self):

    def calculate_gxi(self):

    def caculate_Ei(self):

    def getAlphaJ(self):

    def calculatesingleKernel(self):
