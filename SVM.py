import numpy as np
import math
import random
class SVM:
    def __init__(self,trainDataList,trainLabelList,sigma=10,C=200,toler=0.001):
        self.trainDataMatrix=np.mat(trainDataList)
        self.trainLabelMatrix=np.mat(trainLabelList).T
        self.m,self.n=np.shape(self.trainDataMatrix)
        self.sigma=sigma
        self.C=C
        self.toler=toler

        self.k=self.calculateKernel()
        self.b=0
        self.alpha=[0].self.trainDataMatrix[0]
        self.E=[0 * self.trainLabelMatrix[i,0] for i in range(self.trainDataMatrix.shape[0])]
        self.supportVectorIndex=[]

    def calculateKernel(self):
        k=[[0 for i in range(self.m)] for j in range(self.m) ]
        for i in range(self.m):
            X=self.trainDataMatrix[i:]
            for j in range(i,self.m):
                Z=self.trainDataMatrix[j:]
                result=(X-Z)*(X-Z).T
                result=np.exp((-1)*result/(2*self.sigma**2))
                k[i][j]=result
                k[j][i]=result
        return k

    def isSatisfyKKT(self):

    def calculate_gxi(self,i):
        gxi=0
        index=[i for i,alpha in enumerate(self.alpha) if alpha!=0]
        for j in index:
            gxi+=self.alpha[j]*self.trainLabelMatrix[j]*self.k[j][i]
        gxi+=self.b
        return gxi

    def caculate_Ei(self,i):
        gxi=self.calculate_gxi(self,i)
        return gxi-self.trainLabelMatrix[i]

    def getAlphaJ(self,E1,i):
        E2=0
        max_E1_E2=0
        maxIndex=-1
        noZeroE=[i for i,Ei in enumerate(self.E) if E1!=0]
        for j in noZeroE:
            E2_tmp=self.calculate_gxi(self,j)
            if(math.fabs(E1-E2_tmp)>max_E1_E2):
                max_E1_E2=math.fabs(E1-E2_tmp)
                E2=E2_tmp
                maxIndex=j
        if maxIndex==-1:
            maxIndex=i
            while maxIndex==i:
                maxIndex=int(random.uniform(0,self.m))
            E2=self.caculate_Ei(maxIndex)
        return E2,maxIndex

    def calculatesingleKernel(self):
