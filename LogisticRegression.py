import numpy as np
import logging
import json
from utility import*
FILE_NAME_TRAIN='train.csv'
FILE_NAME_TEST='test.csv'
ALPHA=1e-0
EPOCHS=15000
MODEL_FILE='model/model1'
train_flag=True
logging.basicConfig(filename='output.log',level=logging.DEBUG)
np.set_printoptions(suppress=True)
def appendIntercept(X):
    m,n=X.shape
    col=np.ones((m,1))
    arr=np.hstack((col,X))
    return arr
def initialGuess(n_thetas):
    arr2=np.zeros(n_thetas)
    return arr2
def predict(X,theta):#extra m parameter
    #print X
    pre=np.dot(X,theta)
    sp=np.exp(-pre)
    sp=1+sp
    add=1./sp
    #print add
    return add
def makeGradientUpdate(theta,grads):
    return theta-ALPHA*grads
def calcGradients(X,y,y_predicted,m):
    i=y_predicted-y
    i=i.T
    cal=np.dot(i,X)
    cal=cal/(m)
    return cal
def train(theta,X,y,model):
    J=[]
    m=len(y)
    i=0
    for i in range(0,EPOCHS):
        arr=predict(X,theta)
        #arr2=costFunc(m,y,arr)
        #J.append(arr2)
        arr3=calcGradients(X,y,arr,m)
        theta=makeGradientUpdate(theta,arr3)
    #model['J']=J
    model['theta']=list(theta)
    return model
######
def main():
    if(train_flag):
        model={}
        X_df,y_df=loadData(FILE_NAME_TRAIN)
        X,y,model=normalizeData(X_df,y_df,model)
        X=appendIntercept(X)
        theta=initialGuess(X.shape[1])
        model=train(theta,X,y,model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))
        with open(MODEL_FILE,'r') as f:
            model=json.loads(f.read())
            X_df,y_df=loadData(FILE_NAME_TEST)
            X,y=normalizeTestData(X_df,y_df,model)
            X=appendIntercept(X)
            accuracy(X,y,model)
if __name__ == '__main__':
    main()
        






















