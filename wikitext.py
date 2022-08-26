#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import re
import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
import math
import sklearn
from sklearn.model_selection import train_test_split
import time
tf.random.set_seed(0)
np.random.seed(0)

def main():
    Method=sys.argv[1] # "normal"(=normal attention) or "linear"(linear attention) or "relation" or "mlp"(point wise mlp)
    UnitSize=int(sys.argv[2]) # 64 128 256
    
    learnx,learnt=[],[]
    fin=open("wikitext.sample.txt","r")
    for line in fin:
        line=line.rstrip()
        litmp=re.split("\t",line)
        learnt.append(int(litmp[0]))
        learnx.append(litmp[1])
    fin.close()
    
    tokenizer=tf.keras.preprocessing.text.Tokenizer(filters="",oov_token="<oov>")
    tokenizer.fit_on_texts(learnx)
    learnx=tokenizer.texts_to_sequences(learnx)
    vocabNumber=len(tokenizer.word_index)+1
    learnx=tf.keras.preprocessing.sequence.pad_sequences(learnx,padding="post",dtype=np.int32,value=0)
    learnt=np.asarray(learnt,dtype=np.int32)
    
    MaxEpochSize=1000000
    EmbedSize=100
    attentionUnitSize=UnitSize
    middleUnitSize=attentionUnitSize
    dropoutRate=0.5
    MiniBatchSize=50
    outputSize=len(np.unique(learnt))
    
    trainx,validx,traint,validt=train_test_split(learnx,learnt,test_size=0.2,random_state=0)
    
    trainSize=trainx.shape[0]
    validSize=validx.shape[0]
    trainMiniBatchNumber=trainSize//MiniBatchSize
    validMiniBatchNumber=validSize//MiniBatchSize
    
    model=OneSeqClass(attentionUnitSize,middleUnitSize,vocabNumber,EmbedSize,outputSize,dropoutRate,Method)
    cce=tf.keras.losses.SparseCategoricalCrossentropy()
    acc=tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer=tf.keras.optimizers.Adam()
    
    model(tf.zeros((MiniBatchSize,trainx.shape[1])),False)
    model.summary()

    @tf.function
    def run(tx,tt,flag):
        with tf.GradientTape() as tape:
            model.trainable=flag
            ty=model.call(tx,flag)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    startTime=time.time()
    accCounter,bestAcc=0,0
    for epoch in range(1,MaxEpochSize+1):
        trainIndex=np.random.permutation(trainSize)
        trainFlag=True
        trainCost,trainAcc=0,0
        for subepoch in range(trainMiniBatchNumber):
            miniTrainx=trainx[trainIndex[subepoch*MiniBatchSize:(subepoch+1)*MiniBatchSize]]
            miniTraint=traint[trainIndex[subepoch*MiniBatchSize:(subepoch+1)*MiniBatchSize]]
            miniTrainCost,miniTrainAcc=run(miniTrainx,miniTraint,trainFlag)
            trainCost+=miniTrainCost/trainMiniBatchNumber
            trainAcc+=miniTrainAcc/trainMiniBatchNumber
        
        trainFlag=False
        validIndex=np.random.permutation(validSize)
        validCost,validAcc=0,0
        for subepoch in range(validMiniBatchNumber):
            miniValidx=validx[validIndex[subepoch*MiniBatchSize:(subepoch+1)*MiniBatchSize]]
            miniValidt=validt[validIndex[subepoch*MiniBatchSize:(subepoch+1)*MiniBatchSize]]
            miniValidCost,miniValidAcc=run(miniValidx,miniValidt,trainFlag)
            validCost+=miniValidCost/validMiniBatchNumber
            validAcc+=miniValidAcc/validMiniBatchNumber
        
        print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}, Validation cost= {:.4f}, Validation ACC= {:.4f} (......)".format(epoch,trainCost,trainAcc,validCost,validAcc))
        if float(validAcc)>0.8:
            accCounter+=1
        else:
            accCounter=0
        if accCounter==10:
            endTime=time.time()
            break
    # Test
    testAcc=0
    if dataset=="imdb":
        testx=tf.keras.preprocessing.sequence.pad_sequences(testx,padding="post",dtype=np.int32,value=0)
        for i in range(100):
            miniTestx=testx[250*i:250*(i+1)]
            miniTestt=testt[250*i:250*(i+1)]
            prediction=model(miniTestx,False)
            testAcc+=float(acc(miniTestt,prediction))/100
    elif dataset=="reuters":
        testx=tf.keras.preprocessing.sequence.pad_sequences(testx,padding="post",dtype=np.int32,value=0)
        for i in range(5):
            miniTestx=testx[463*i:463*(i+1)]
            miniTestt=testt[463*i:463*(i+1)]
            prediction=model(miniTestx,False)
            testAcc+=float(acc(miniTestt,prediction))/5
    print("{} {} {} {} {:.4f} {:.4f} {:.4f}".format(dataset,Method,attentionUnitSize,model.count_params(),endTime-startTime,(endTime-startTime)/epoch,testAcc))

class LstmWiEmbed(tf.keras.Model):
    def __init__(self,middleUnitSize,vocabNumber,embedSize,outputSize,dropoutRate):
        super(LstmWiEmbed,self).__init__()
        self.lstm=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(middleUnitSize)) #,return_sequences=True)
        self.embed=tf.keras.layers.Embedding(input_dim=vocabNumber,output_dim=embedSize,mask_zero=True)
        self.w1=tf.keras.layers.Dense(middleUnitSize)
        self.w2=tf.keras.layers.Dense(outputSize)
    def call(self,x1,learningFlag):
        x1=self.embed(x1)
        y=self.lstm(x1)
        y=self.w1(y)
        y=tf.keras.activations.relu(y)
        y=self.w2(y)
        y=tf.nn.softmax(y)
        return y

class OneSeqClass(tf.keras.Model):
    def __init__(self,attentionUnitSize,middleUnitSize,vocabNumber,embedSize,outputSize,dropoutRate,Method):
        super(OneSeqClass,self).__init__()
        if Method=="normal":
            self.a=Attention(attentionUnitSize,dropoutRate)
        elif Method=="relation":
            self.a=Relation(attentionUnitSize,dropoutRate)
        elif Method=="linear":
            self.a=LinearAttention(attentionUnitSize,dropoutRate)
        elif Method=="mlp":
            self.a=MLP(attentionUnitSize,dropoutRate)
        self.embed=tf.keras.layers.Embedding(input_dim=vocabNumber,output_dim=embedSize,mask_zero=True)
        self.pe=PositionalEncoder()
        self.masker=AttentionMaskLabelSeq()
        self.w1=tf.keras.layers.Dense(middleUnitSize)
        self.w2=tf.keras.layers.Dense(outputSize)
        self.lr=keras.layers.LeakyReLU()
        self.dropout=tf.keras.layers.Dropout(dropoutRate)
    def call(self,x1,learningFlag):
        maskSeq,minNumber=self.masker(x1)
        x1=self.embed(x1)
        x1=self.pe(x1)
        y=self.a(x1,x1,maskSeq,minNumber,learningFlag)
        y=self.lr(y)
        y=self.dropout(y,training=learningFlag)
        y=self.w1(y)
        y=y[:,0,:]
        y=self.w2(y)
        y=tf.nn.softmax(y)
        return y

class Attention(tf.keras.Model):
    def __init__(self,Depth,DropoutRate):
        super(Attention,self).__init__()
        self.Depth=Depth
        self.wq=tf.keras.layers.Dense(Depth,use_bias=False)
        self.wk=tf.keras.layers.Dense(Depth,use_bias=False)
        self.wv=tf.keras.layers.Dense(Depth,use_bias=False)
        self.outputLayer=tf.keras.layers.Dense(Depth,use_bias=False)
        self.dropout=tf.keras.layers.Dropout(DropoutRate)
    def call(self,x1,x2,maskSeq,minNumber,learningFlag):
        q=self.wq(x1)
        k=self.wk(x2)
        v=self.wv(x2)
        q=q*self.Depth**(-0.5)
        a=tf.matmul(q,k,transpose_b=True)
        a=a+tf.cast(maskSeq,dtype=tf.float32)*minNumber
        a=tf.nn.softmax(a)
        a=self.dropout(a,training=learningFlag)
        y=tf.matmul(a,v)
        y=tf.keras.activations.relu(y)
        y=self.dropout(y,training=learningFlag)
        y=self.outputLayer(y)
        return y

class Relation(Attention):
    def __init__(self,Depth,DropoutRate):
        super().__init__(Depth,DropoutRate)
        self.wg=tf.keras.layers.Dense(Depth,use_bias=True)
        self.wh=tf.keras.layers.Dense(Depth,use_bias=True)
        self.w1=tf.keras.layers.Dense(Depth,use_bias=False)
    def call(self,x1,x2,maskSeq,minNumber,learningFlag):
        h=self.wh(x1)
        g=self.wg(x2)
        b=tf.transpose(maskSeq,perm=[0,2,1])
        h=h*tf.cast(~b,dtype=tf.float32)
        hsum=tf.reduce_sum(h,axis=1)
        c=tf.cast(~maskSeq,dtype=tf.float32)
        c=tf.reduce_sum(c,axis=2)
        hsum=hsum*c**(-1)
        hsum=tf.expand_dims(hsum,axis=1)
        y=hsum*g
        y=self.dropout(y,training=learningFlag)
        y=self.w1(y)
        y=tf.keras.activations.relu(y)
        y=self.dropout(y,training=learningFlag)
        y=self.outputLayer(y)
        return y

class MLP(Attention):
    def __init__(self,Depth,DropoutRate):
        super().__init__(Depth,DropoutRate)
        self.w1=tf.keras.layers.Dense(Depth,use_bias=True)
        self.w2=tf.keras.layers.Dense(Depth,use_bias=True)
        self.w3=tf.keras.layers.Dense(Depth,use_bias=False)
    def call(self,x1,x2,maskSeq,minNumber,learningFlag):
        q=self.w1(x1)
        k=self.w2(x2)
        b=tf.transpose(maskSeq,perm=[0,2,1])
        q=q*tf.cast(~b,dtype=tf.float32)
        k=k*tf.cast(~b,dtype=tf.float32)
        y=q*k
        y=self.dropout(y,training=learningFlag)
        y=self.w3(y)
        y=tf.keras.activations.relu(y)
        y=self.dropout(y,training=learningFlag)
        y=self.outputLayer(y)
        return y

class LinearAttention(Attention):
    def __init__(self,Depth,DropoutRate):
        super().__init__(Depth,DropoutRate)
        self.elu=tf.keras.layers.ELU()
    def call(self,x1,x2,maskSeq,minNumber,learningFlag):
        q=self.wq(x1)
        q=self.elu(q)+1
        k=self.wk(x2)
        v=self.wv(x2)
        q=q*x2.shape[2]**(-0.5)
        maskSeqT=tf.transpose(maskSeq,perm=[0,2,1])
        k=self.elu(k)+1
        m=tf.reduce_mean(k,axis=2)
        m=tf.expand_dims(m,axis=2)
        k=tf.nn.softmax(k)
        k=k*tf.cast(~maskSeqT,dtype=tf.float32)
        v=v*tf.cast(~maskSeqT,dtype=tf.float32)
        a=tf.matmul(k,v,transpose_a=True)
        a=self.dropout(a,training=learningFlag)
        y=tf.matmul(q,a)
        y=y/m
        y=tf.keras.activations.relu(y)
        y=self.dropout(y,training=learningFlag)
        y=self.outputLayer(y)
        return y

class AttentionMaskLabelSeq(tf.keras.Model):
    def call(self,x):
        x=tf.cast(x,dtype=tf.int32)
        inputBatchSize,inputLength=tf.unstack(tf.shape(x))
        maskSeq=tf.equal(x,0)
        maskSeq=tf.reshape(maskSeq,[inputBatchSize,1,inputLength])
        return maskSeq,x.dtype.min

class PositionalEncoder(tf.keras.layers.Layer):
    def call(self,x):
        dataType=x.dtype
        inputBatchSize,inputLength,inputEmbedSize=tf.unstack(tf.shape(x))
        Depth_counter=tf.range(inputEmbedSize)//2*2
        Depth_matrix=tf.tile(tf.expand_dims(Depth_counter,0),[inputLength,1])
        Depth_matrix=tf.pow(10000.0,tf.cast(Depth_matrix/inputEmbedSize,dataType))
        phase=tf.cast(tf.range(inputEmbedSize)%2,dataType)*math.pi/2
        phase_matrix=tf.tile(tf.expand_dims(phase,0),[inputLength,1])
        pos_counter=tf.range(inputLength)
        pos_matrix=tf.cast(tf.tile(tf.expand_dims(pos_counter,1),[1,inputEmbedSize]),dataType)
        positional_encoding=tf.sin(pos_matrix/Depth_matrix+phase_matrix)
        positional_encoding=tf.tile(tf.expand_dims(positional_encoding,0),[inputBatchSize,1,1])
        return x+positional_encoding

if __name__ == "__main__":
    main()
