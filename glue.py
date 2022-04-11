#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow import keras
import math
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # dataset and network
    Dataset="CoLA" # "CoLA" or "SST-2" or "MRPC" or "QQP" or "QNLI" or "RTE" or "WNLI" or "STS-B" or "MNLI-m" or "MNLI-mm"
    Method="relation" # "normal"(=normal attention) or "linear"(linear attention) or "relation" or "mlp"(point wise mlp)
    
    # hyperparameter
    EmbedSize=100
    Depth=128
    MiddleUnitSize=128
    DropoutRate=0.5
    MiniBatchSize=64
    Patience=5
    
    # parameter
    vocabNumber=10000 # the number of vocab (depends on the datasets)
    outputSize=0
    if Dataset=="CoLA" or Dataset=="SST-2" or Dataset=="MRPC" or Dataset=="QQP" or Dataset=="QNLI" or Dataset=="RTE" or Dataset=="WNLI":
        outputSize=2
    elif Dataset=="STS-B":
        outputSize=1
    elif Dataset=="MNLI-mm" or Dataset=="MNLI-m":
        outputSize=3
    
    model=""
    if Dataset=="STS-B":
        model=TwoSeqReg(Depth,MiddleUnitSize,vocabNumber,EmbedSize,outputSize,DropoutRate,Method)
    elif Dataset=="QNLI" or Dataset=="RTE" or Dataset=="MRPC" or Dataset=="QQP" or Dataset=="MNLI-m" or Dataset=="MNLI-mm":
        model=TwoSeqClass(Depth,MiddleUnitSize,vocabNumber,EmbedSize,outputSize,DropoutRate,Method)
    else:
        model=OneSeqClass(Depth,MiddleUnitSize,vocabNumber,EmbedSize,outputSize,DropoutRate,Method)

class OneSeqClass(tf.keras.Model):
    def __init__(self,Depth,MiddleUnitSize,vocabNumber,embedSize,outputSize,DropoutRate,Method):
        super(OneSeqClass,self).__init__()
        if Method=="normal":
            self.a=Attention(Depth,DropoutRate)
        elif Method=="relation":
            self.a=Relation(Depth,DropoutRate)
        elif Method=="linear":
            self.a=LinearAttention(Depth,DropoutRate)
        elif Method=="mlp":
            self.a=MLP(Depth,DropoutRate)
        self.embed=tf.keras.layers.Embedding(input_dim=vocabNumber,output_dim=embedSize,mask_zero=True)
        self.pe=PositionalEncoder()
        self.masker=AttentionMaskLabelSeq()
        self.w1=tf.keras.layers.Dense(MiddleUnitSize)
        self.w2=tf.keras.layers.Dense(outputSize)
        self.lr=keras.layers.LeakyReLU()
        self.dropout=tf.keras.layers.Dropout(DropoutRate)
    def call(self,x1,learningFlag):
        maskSeq,minNumber=self.masker(x1)
        x1=self.embed(x1)
        x1=self.pe(x1)
        y=self.a(x1,x1,maskSeq,minNumber,learningFlag)
        y=self.lr(y)
        y=self.dropout(y,training=learningFlag)
        y=self.w1(y)
        y=self.lr(y)
        y=y[:,0,:]
        y=self.w2(y)
        y=tf.nn.softmax(y)
        return y

class TwoSeqClass(tf.keras.Model):
    def __init__(self,Depth,MiddleUnitSize,vocabNumber,embedSize,outputSize,DropoutRate,Method):
        super(TwoSeqClass,self).__init__()
        if Method=="normal":
            self.a=Attention(Depth,DropoutRate)
        elif Method=="relation":
            self.a=Relation(Depth,DropoutRate)
        elif Method=="linear":
            self.a=LinearAttention(Depth,DropoutRate)
        elif Method=="mlp":
            self.a=MLP(Depth,DropoutRate)
        self.embed=tf.keras.layers.Embedding(input_dim=vocabNumber,output_dim=embedSize,mask_zero=True)
        self.pe=PositionalEncoder()
        self.masker=AttentionMaskLabelSeq()
        self.w1=tf.keras.layers.Dense(MiddleUnitSize)
        self.w2=tf.keras.layers.Dense(MiddleUnitSize)
        self.w3=tf.keras.layers.Dense(outputSize)
        self.lr=keras.layers.LeakyReLU()
        self.dropout=tf.keras.layers.Dropout(DropoutRate)
    def call(self,x1,x2,learningFlag):
        maskSeq,minNumber=self.masker(x1)
        x1=self.embed(x1)
        x1=self.pe(x1)
        y1=self.a(x1,x1,maskSeq,minNumber,learningFlag)
        y1=self.lr(y1)
        y1=self.dropout(y1,training=learningFlag)
        y1=self.w1(y1)
        y1=self.lr(y1)
        y1=y1[:,0,:]
        maskSeq2,minNumber2=self.masker(x2)
        x2=self.embed(x2)
        x2=self.pe(x2)
        y2=self.a(x2,x2,maskSeq2,minNumber2,learningFlag)
        y2=self.lr(y2)
        y2=self.dropout(y2,training=learningFlag)
        y2=self.w2(y2)
        y2=self.lr(y2)
        y2=y2[:,0,:]
        y=y1*y2
        y=self.w3(y)
        y=tf.nn.softmax(y)
        return y

class TwoSeqReg(tf.keras.Model):
    def __init__(self,Depth,MiddleUnitSize,vocabNumber,embedSize,outputSize,DropoutRate,Method):
        super(TwoSeqReg,self).__init__()
        if Method=="normal":
            self.a=Attention(Depth,DropoutRate)
        elif Method=="relation":
            self.a=Relation(Depth,DropoutRate)
        elif Method=="linear":
            self.a=LinearAttention(Depth,DropoutRate)
        elif Method=="mlp":
            self.a=MLP(Depth,DropoutRate)
        self.embed=tf.keras.layers.Embedding(input_dim=vocabNumber,output_dim=embedSize,mask_zero=True)
        self.pe=PositionalEncoder()
        self.masker=AttentionMaskLabelSeq()
        self.w1=tf.keras.layers.Dense(MiddleUnitSize)
        self.w2=tf.keras.layers.Dense(MiddleUnitSize)
        self.w3=tf.keras.layers.Dense(outputSize)
        self.lr=keras.layers.LeakyReLU()
        self.dropout=tf.keras.layers.Dropout(DropoutRate)
    def call(self,x1,x2,learningFlag):
        maskSeq,minNumber=self.masker(x1)
        x1=self.embed(x1)
        x1=self.pe(x1)
        y1=self.a(x1,x1,maskSeq,minNumber,learningFlag)
        y1=self.lr(y1)
        y1=self.dropout(y1,training=learningFlag)
        y1=self.w1(y1)
        y1=self.lr(y1)
        y1=y1[:,0,:]
        maskSeq2,minNumber2=self.masker(x2)
        x2=self.embed(x2)
        x2=self.pe(x2)
        y2=self.a(x2,x2,maskSeq2,minNumber2,learningFlag)
        y2=self.lr(y2)
        y2=self.dropout(y2,training=learningFlag)
        y2=self.w2(y2)
        y2=self.lr(y2)
        y2=y2[:,0,:]
        y=y1*y2
        y=self.w3(y)
        y=tf.keras.activations.sigmoid(y)
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
