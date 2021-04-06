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
            X=self.trainDataMatrix[i,:]
            for j in range(i,self.m):
                Z=self.trainDataMatrix[j,:]
                result=(X-Z)*(X-Z).T
                result=np.exp((-1)*result/(2*self.sigma**2))
                k[i][j]=result
                k[j][i]=result
        return k

    def isSatisfyKKT(self,i):
        gxi=self.calculate_gxi(self,i)
        yi=self.trainLabelMatrix[i]
        if(math.fabs(self.alpha[i])<self.toler and yi*gxi>=1):
            return True
        elif(math.fabs(self.alpha[i]-self.C)<self.toler and yi*gxi<=1):
            return True
        elif(self.alpha[i]>-self.toler and self.alpha[i]<self.C+self.toler and math.fabs(gxi*yi-1)<=self.toler):
            return True
        return False

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

    def calculatesingleKernel(self,x1,x2):
        result=(x1-x2)*(x1-x2).T
        result=(-1)*result/(2*self.sigma**2)
        return np.exp(result)
    def train(self,iter=100):
        iterstep=0
        parameterchanged=1
        while(iterstep<iter and parameterchanged>0):
            iter+=1
            parameterchanged=0
            for i in range(self.m):
                if self.isSatisfyKKT(i)==False:
                    E1=self.caculate_Ei(i)
                    E2,j=self.getAlphaJ(E1,i)
                    y1=self.trainLabelMatrix[i]
                    y2=self.trainLabelMatrix[j]
                    alpha_1=self.alpha[i]
                    alpha_2=self.alpha[j]
                    if(y1!=y2):
                        L=max(0,alpha_2-alpha_1)
                        H=min(self.C,self.C+alpha_2-alpha_1)
                    else:
                        L=max(0,alpha_2+alpha_1-self.C)
                        H=min(self.C,alpha_2+alpha_1)
                    if(L==H):
                        continue
                    k11=self.k[i][i]
                    k12=self.k[i][j]
                    k21=self.k[j][i]
                    k22=self.k[j][j]
                    alpha_new_2=alpha_2+y2*(E1-E2)/(k11+k22-2*k12)
                    if(alpha_new_2<L):
                        alpha_new_2=L
                    elif(alpha_new_2>H):
                        alpha_new_2=H
                    alpha_new_1=alpha_1+y1*y2*(alpha_2-alpha_new_2)
                    b1_new=-1*E1-y1*k11*(alpha_new_1-alpha_1)-y2*k21*(alpha_new_2-alpha_2)+self.b
                    b2_new=-1*E2-y1*k12*(alpha_new_2-alpha_2)-y2*k22*(alpha_new_2-alpha_2)+self.b
                    if(alpha_new_1>0 and alpha_new_1<self.C):
                        b_new=b1_new
                    elif(alpha_new_2>0 and alpha_new_2<self.C):
                        b_new=b2_new
                    else:
                        b_new=(b1_new+b2_new)/2
                    self.alpha[i]=alpha_new_1
                    self.alpha[j]=alpha_new_2
                    self.b=b_new
                    self.E[i]=self.caculate_Ei(i)
                    self.E[j]=self.caculate_Ei(j)
                    if(math.fabs(alpha_new_2-alpha_2)>=0.00001):
                        parameterchanged+=1
        for i in range(self.m):
            if (self.alpha[i]>0):
                self.supportVectorIndex.append(i)
    def predict(self,x):
        result=0
        for i in self.supportVectorIndex:
            tmp=self.calculatesingleKernel(self.trainDataMatrix[i,:],np.matrix(x))
            result+=self.alpha[i]*self.trainLabelMatrix[i]*tmp
        result+=self.b
        return np.sign(result)
