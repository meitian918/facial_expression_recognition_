import numpy as np
import scipy.linalg
import os, time
import tensorflow as tf
import pickle    
import os    
import json
import os
from PIL import Image   
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import warp
import time

# load MNIST data

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()  
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = (np.array(data,dtype='float') - (255 / 2.0)) / 255
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(1,height*width))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def loadMNIST(fname):
    train, valid = pickle.load(open("D:/论文实现/dataset/ROI/data.pkl","rb"),encoding='iso-8859-1')
    trainData, validData, testData = {}, {}, {}
    data = []
#    test0 = np.fromstring(open("D:/论文实现/dataset/NORMAL_TEST/0.bin","rb").read(), np.uint8)
#    test1 = np.fromstring(open("D:/论文实现/dataset/NORMAL_TEST/1.bin","rb").read(), np.uint8)
#    test2 = np.fromstring(open("D:/论文实现/dataset/NORMAL_TEST/2.bin","rb").read(), np.uint8)
#    test3 = np.fromstring(open("D:/论文实现/dataset/NORMAL_TEST/3.bin","rb").read(), np.uint8)
 #   test4 = np.fromstring(open("D:/论文实现/dataset/NORMAL_TEST/4.bin","rb").read(), np.uint8)
#    test = np.hstack((test0, test1, test2,test3,test4))
    test = np.zeros([13824000,],dtype=np.float32)
    for i in range(1,13501):
        filename = "C:/Users/Administrator/Desktop/data/suoyou/"+str(i)+".jpg"    
    #    test[j] = ImageToMatrix(filename);
        t = ImageToMatrix(filename);
        data.extend(t)    
    for j in range(13500):
        for k in range(1024):
            test[1024*j + k] = data[j][k]

    trainData["image"] = (train[0] - (255 / 2.0)) / 255
    trainData["image"] = trainData["image"].reshape([-1,32, 32,1]).astype(np.float32)
    validData["image"] = (valid[0] - (255 / 2.0)) / 255
    validData["image"] = validData["image"].reshape([-1,32, 32,1]).astype(np.float32)
    testData["image"] = test.reshape(13500, 32 * 32)
#    testData["image"] = (testData["image"] - (255 / 2.0)) / 255
    testData["image"] = testData["image"].reshape([-1,32, 32,1]).astype(np.float32)
    trainData["label"] = train[1].astype(np.float32)	
    trainData["label"] = (np.arange(5) == trainData["label"][:, None]).astype(np.float32)
    validData["label"] = valid[1].astype(np.float32)
    validData["label"] = (np.arange(5) == validData["label"][:, None]).astype(np.float32)
#    testData["label"] = np.zeros([13500,5],dtype=np.float32)
    testData["label"] = np.zeros([13500,5],dtype=np.float32)
    for k in range(13500):
        if k < 2700:
            testData["label"][k] = [1,0,0,0,0]
        if k >= 2700 and k < 5400:
            testData["label"][k] = [0,1,0,0,0]
        if k >= 5400 and k < 8100:
            testData["label"][k] = [0,0,1,0,0]
        if k >= 8100 and k < 10800:
            testData["label"][k] = [0,0,0,1,0]
        if k >= 10800 and k < 13500:
            testData["label"][k] = [0,0,0,0,1]
#    testData1["label"] = testData["label"][10800:13500]
#    testData1["image"] = testData["image"][10800:13500]
    return trainData, validData, testData


# generate training batch  chan sheng rao dong ju zhen
def genPerturbations(params):
    with tf.name_scope("genPerturbations"):
        X = np.tile(params.canon4pts[:, 0], [params.batchSize, 1])
        Y = np.tile(params.canon4pts[:, 1], [params.batchSize, 1])
        dX = tf.random_normal([params.batchSize, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batchSize, 1]) * params.warpScale["trans"]
        dY = tf.random_normal([params.batchSize, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batchSize, 1]) * params.warpScale["trans"]
        O = np.zeros([params.batchSize, 4], dtype=np.float32)
        I = np.ones([params.batchSize, 4], dtype=np.float32)
        # fit warp parameters to generated displacements
        if params.warpType == "affine":
            J = np.concatenate([np.stack([X, Y, I, O, O, O], axis=-1),
                                np.stack([O, O, O, X, Y, I], axis=-1)], axis=1)
            dXY = tf.expand_dims(tf.concat([dX, dY], 1), -1)
            dpBatch = tf.matrix_solve_ls(J, dXY)
        elif params.warpType == "homography":
            A = tf.concat([tf.stack([X, Y, I, O, O, O, -X * (X + dX), -Y * (X + dX)], axis=-1),
                           tf.stack([O, O, O, X, Y, I, -X * (Y + dY), -Y * (Y + dY)], axis=-1)], 1)
            b = tf.expand_dims(tf.concat([X + dX, Y + dY], 1), -1)
            dpBatch = tf.matrix_solve_ls(A, b)
            dpBatch -= tf.to_float(tf.reshape([1, 0, 0, 0, 1, 0, 0, 0], [1, 8, 1]))
        dpBatch = tf.reduce_sum(dpBatch, reduction_indices=-1)
        dpMtrxBatch = warp.vec2mtrxBatch(dpBatch, params)
    return dpBatch

def evaluate(data, imageRawBatch,labelBatch, prediction, sess, params):
    dataN = len(data["image"])
    batchN = int(np.ceil(float(dataN) / params.batchSize))
    accurateCount = 0
    for b in range(batchN):
        # use some dummy data (0) as batch filler if necesaary
        if b != batchN - 1:
            realIdx = np.arange(params.batchSize * b, params.batchSize * (b + 1))
        else:
            realIdx = np.arange(params.batchSize * b, dataN)
        idx = np.zeros([params.batchSize], dtype=int)
        idx[:len(realIdx)] = realIdx
        batch = {
            imageRawBatch: data["image"][idx],
            labelBatch: data["label"][idx],
        }
        predictionBatch = sess.run(prediction, feed_dict=batch) # predictionBatch is an array ,it's the result
        accurateCount += predictionBatch[:len(realIdx)].sum()
    accuracy = float(accurateCount) / dataN
    return accuracy
    
def evaluate1(data, imageRawBatch,labelBatch, result,real_result,sess, params):
    dataN = len(data["image"])
    batchN = int(np.ceil(float(dataN) / params.batchSize))
    accurateCount = 0
    raw,total = np.array([],dtype = np.int64),np.array([],dtype = np.int64)
    start_Real = time.time()
    for b in range(batchN):
        # use some dummy data (0) as batch filler if necesaary
        if b != batchN - 1:
            realIdx = np.arange(params.batchSize * b, params.batchSize * (b + 1))
        else:
            realIdx = np.arange(params.batchSize * b, dataN)
        idx = np.zeros([params.batchSize], dtype=int)
        idx[:len(realIdx)] = realIdx
        batch = {
            imageRawBatch: data["image"][idx],
            labelBatch: data["label"][idx],
        }
        resultBatch = sess.run(result, feed_dict=batch) # predictionBatch is an array ,it's the result
        real_resultBatch = sess.run(real_result, feed_dict=batch)
        for j in range(len(resultBatch)):
             if resultBatch[j] == real_resultBatch[j]:
                 accurateCount +=1 
        raw = np.append(raw,resultBatch)
        total = np.append(total,real_resultBatch)
    accuracy = float(accurateCount) / dataN
    end_End = time.time()
    return accuracy,raw,total,end_End-start_Real
    
# evaluation on validation/test sets
def knnevaluate(data, imageRawBatch,labelBatch, result,real_result, sess, params):
    params.batchSize = 4500
    dataN = len(data["image"])
    batchN = int(np.ceil(float(dataN) / params.batchSize))
    accurateCount = 0
    start_Real = time.time()
    for b in range(batchN):
        # use some dummy data (0) as batch filler if necesaary
        if b != batchN - 1:
            realIdx = np.arange(params.batchSize * b, params.batchSize * (b + 1))
        else:
            realIdx = np.arange(params.batchSize * b, dataN)
        idx = np.zeros([params.batchSize], dtype=int)
        idx[:len(realIdx)] = realIdx
        batch = {
            imageRawBatch: data["image"][idx],
            labelBatch: data["label"][idx],
        }
        resultBatch = sess.run(result,feed_dict=batch) # predictionBatch is an array ,it's the result
        real_resultBatch = sess.run(real_result, feed_dict=batch)
        for k in range(len(resultBatch)):
            if k % 9 == 0:
                num0,num1,num2,num3,num4,num5 = 0,0,0,0,0,0
                t = k
                for h in range(t,t+9):
                    if resultBatch[h] == 0:
                        num0 = num0 + 1
                    elif resultBatch[h] == 1:
                        num1 = num1 + 1
                    elif resultBatch[h] == 2:
                        num2 = num2 + 1
                    elif resultBatch[h] == 3:
                        num3 = num3 + 1
                    elif resultBatch[h] == 4:
                        num4 = num4 + 1
                num5 = max(num0,num1,num2,num3,num4)
                if num5 == num4:
                    resultBatch[k:k+9] = np.array([4,4,4,4,4,4,4,4,4])
                elif num5 == num3:
                    resultBatch[k:k+9] = np.array([3,3,3,3,3,3,3,3,3])
                elif num5 == num2:
                    resultBatch[k:k+9] = np.array([2,2,2,2,2,2,2,2,2])
                elif num5 == num1:
                    resultBatch[k:k+9] = np.array([1,1,1,1,1,1,1,1,1])
                elif num5 == num0:
                    resultBatch[k:k+9] = np.array([0,0,0,0,0,0,0,0,0])
        for j in range(len(resultBatch)):
             if resultBatch[j] == real_resultBatch[j]:
                 accurateCount +=1              
    accuracy = float(accurateCount) / dataN
    end_End = time.time()
    return end_End-start_Real

def cmevaluate(data, imageRawBatch,labelBatch, result,real_result, softmax ,sess, params):#wo de fang fa
    params.batchSize = 4500
    dataN = len(data["image"])
    batchN = int(np.ceil(float(dataN) / params.batchSize))
    accurateCount1, accurateCount2 = 0,0
    c1,c2,c3 = np.array([],dtype = np.int64),np.array([],dtype = np.int64),np.array([],dtype = np.int64)
    a1= np.zeros([13500,5],dtype = np.float32)
    result1,result2,result3 = np.zeros([1500,],dtype = np.int64),np.zeros([1500,],dtype = np.int64),np.zeros([1500,],dtype = np.int64)
    for b in range(batchN):
        # use some dummy data (0) as batch filler if necesaary
        if b != batchN - 1:
            realIdx = np.arange(params.batchSize * b, params.batchSize * (b + 1))
        else:
            realIdx = np.arange(params.batchSize * b, dataN)
        idx = np.zeros([params.batchSize], dtype=int)
        idx[:len(realIdx)] = realIdx
        batch = {
            imageRawBatch: data["image"][idx],
            labelBatch: data["label"][idx],
        }
        resultBatch1 = np.zeros([params.batchSize,],dtype=np.int64)
        resultBatch2 = np.zeros([params.batchSize,],dtype=np.int64)
#        resultBatch = sess.run(result,feed_dict=batch) # predictionBatch is an array ,it's the result
        real_resultBatch = sess.run(real_result, feed_dict=batch)
        softmaxbatch =  sess.run(softmax, feed_dict=batch)
        if b == 0:
            a1[0:4500] = softmaxbatch
        elif b == 1:
            a1[4500:9000] = softmaxbatch
        else:
            a1[9000:13500] = softmaxbatch
        resultBatch = np.argmax(softmaxbatch,axis = 1)
 #       softmaxbatch = softmaxbatch.argsort()
        k = 0
        for k in range(len(resultBatch)):
            if k % 9 == 0:
                num0,num1,num2,num3,num4,num5 = 0,0,0,0,0,0
                t = k
                for h in range(t,t+9):
                    if resultBatch[h] == 0:
                        num0 = num0 + 1
                    elif resultBatch[h] == 1:
                        num1 = num1 + 1
                    elif resultBatch[h] == 2:
                        num2 = num2 + 1
                    elif resultBatch[h] == 3:
                        num3 = num3 + 1
                    elif resultBatch[h] == 4:
                        num4 = num4 + 1
                num5 = max(num0,num1,num2,num3,num4)
                if resultBatch[k+5] == resultBatch[k+8]:
                    resultBatch1[k:k+9] = np.array([resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5],resultBatch[k+5]])
                else:
                    if num5 == num4:
                        resultBatch1[k:k+9] = np.array([4,4,4,4,4,4,4,4,4])
                    elif num5 == num3:
                        resultBatch1[k:k+9] = np.array([3,3,3,3,3,3,3,3,3])
                    elif num5 == num2:
                        resultBatch1[k:k+9] = np.array([2,2,2,2,2,2,2,2,2])
                    elif num5 == num1:
                        resultBatch1[k:k+9] = np.array([1,1,1,1,1,1,1,1,1])
                    elif num5 == num0:
                        resultBatch1[k:k+9] = np.array([0,0,0,0,0,0,0,0,0])
                if num5 == num4:
                    resultBatch2[k:k+9] = np.array([4,4,4,4,4,4,4,4,4])
                elif num5 == num3:
                    resultBatch2[k:k+9] = np.array([3,3,3,3,3,3,3,3,3])
                elif num5 == num2:
                    resultBatch2[k:k+9] = np.array([2,2,2,2,2,2,2,2,2])
                elif num5 == num1:
                    resultBatch2[k:k+9] = np.array([1,1,1,1,1,1,1,1,1])
                elif num5 == num0:
                    resultBatch2[k:k+9] = np.array([0,0,0,0,0,0,0,0,0])
        for j in range(len(resultBatch1)):
             if resultBatch1[j] == real_resultBatch[j]:
                 accurateCount1 +=1
             if resultBatch2[j] == real_resultBatch[j]:
                 accurateCount2 +=1
        c1 = np.append(c1,resultBatch1)
        c2 = np.append(c2,resultBatch2)
        c3 = np.append(c3,real_resultBatch)
    for l in range(len(c1)//9):
        if c1[l*9] == c1[l*9+8]:
            result1[l] = c1[l*9]
        if c2[l*9] == c2[l*9+8]:
            result2[l] = c2[l*9]
        if c3[l*9] == c3[l*9+8]:
            result3[l] = c3[l*9]        
    accuracy1 = float(accurateCount1) / dataN
    accuracy2 = float(accurateCount2) / dataN
    return accuracy1,accuracy2,c1,c2,c3,result1,result2,result3,a1,resultBatch
    

# generate batch of warped images from batch (bilinear interpolation)
def imageWarpIm(imageBatch, pMtrxBatch, params, name=None):
    with tf.name_scope("ImWarp"):
        imageBatch = tf.expand_dims(imageBatch, -1)
        batchSize = tf.shape(imageBatch)[0]
        imageH, imageW = params.H, params.H
        H, W = params.H, params.W
        warpGTmtrxBatch = tf.tile(tf.expand_dims(params.warpGTmtrx, 0), [batchSize, 1, 1])
        transMtrxBatch = tf.matmul(warpGTmtrxBatch, pMtrxBatch)
        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        XYhom = tf.transpose(tf.stack([X.reshape([-1]), Y.reshape([-1]), np.ones([X.size])], axis=1))
        XYhomBatch = tf.tile(tf.expand_dims(XYhom, 0), [batchSize, 1, 1])
        XYwarpHomBatch = tf.matmul(transMtrxBatch, tf.to_float(XYhomBatch))
        XwarpHom, YwarpHom, ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
        Xwarp = tf.reshape(XwarpHom / ZwarpHom, [batchSize, H, W])
        Ywarp = tf.reshape(YwarpHom / ZwarpHom, [batchSize, H, W])
        # get the integer sampling coordinates
        Xfloor, Xceil = tf.floor(Xwarp), tf.ceil(Xwarp)
        Yfloor, Yceil = tf.floor(Ywarp), tf.ceil(Ywarp)
        XfloorInt, XceilInt = tf.to_int32(Xfloor), tf.to_int32(Xceil)
        YfloorInt, YceilInt = tf.to_int32(Yfloor), tf.to_int32(Yceil)
        imageIdx = tf.tile(tf.reshape(tf.range(batchSize), [batchSize, 1, 1]), [1, H, W])
        imageVec = tf.reshape(imageBatch, [-1, tf.shape(imageBatch)[3]])
        imageVecOutside = tf.concat([imageVec, tf.zeros([1, tf.shape(imageBatch)[3]])], 0)
        idxUL = (imageIdx * imageH + YfloorInt) * imageW + XfloorInt
        idxUR = (imageIdx * imageH + YfloorInt) * imageW + XceilInt
        idxBL = (imageIdx * imageH + YceilInt) * imageW + XfloorInt
        idxBR = (imageIdx * imageH + YceilInt) * imageW + XceilInt
        idxOutside = tf.fill([batchSize, H, W], batchSize * imageH * imageW)

        def insideIm(Xint, Yint):
            return (Xint >= 0) & (Xint < imageW) & (Yint >= 0) & (Yint < imageH)

        idxUL = tf.where(insideIm(XfloorInt, YfloorInt), idxUL, idxOutside)
        idxUR = tf.where(insideIm(XceilInt, YfloorInt), idxUR, idxOutside)
        idxBL = tf.where(insideIm(XfloorInt, YceilInt), idxBL, idxOutside)
        idxBR = tf.where(insideIm(XceilInt, YceilInt), idxBR, idxOutside)
        # bilinear interpolation
        Xratio = tf.reshape(Xwarp - Xfloor, [batchSize, H, W, 1])
        Yratio = tf.reshape(Ywarp - Yfloor, [batchSize, H, W, 1])
        ImUL = tf.to_float(tf.gather(imageVecOutside, idxUL)) * (1 - Xratio) * (1 - Yratio)
        ImUR = tf.to_float(tf.gather(imageVecOutside, idxUR)) * (Xratio) * (1 - Yratio)
        ImBL = tf.to_float(tf.gather(imageVecOutside, idxBL)) * (1 - Xratio) * (Yratio)
        ImBR = tf.to_float(tf.gather(imageVecOutside, idxBR)) * (Xratio) * (Yratio)
        ImWarpBatch = ImUL + ImUR + ImBL + ImBR
        ImWarpBatch = tf.identity(ImWarpBatch, name=name)
    return ImWarpBatch


# generate batch of warped images from batch (bilinear interpolation)
def ImWarpIm(ImBatch, pMtrxBatch, params, name=None):
    with tf.name_scope("ImWarp"):
        batchSize = tf.shape(ImBatch)[0]
        H, W = params.H, params.W
        warpGTmtrxBatch = tf.tile(tf.expand_dims(params.warpGTmtrx, 0), [batchSize, 1, 1])
        transMtrxBatch = tf.matmul(warpGTmtrxBatch, pMtrxBatch)
        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        XYhom = tf.transpose(tf.stack([X.reshape([-1]), Y.reshape([-1]), np.ones([X.size])], axis=1))
        XYhomBatch = tf.tile(tf.expand_dims(XYhom, 0), [batchSize, 1, 1])
        XYwarpHomBatch = tf.matmul(transMtrxBatch, tf.to_float(XYhomBatch))
        XwarpHom, YwarpHom, ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
        Xwarp = tf.reshape(XwarpHom / ZwarpHom, [batchSize, H, W])
        Ywarp = tf.reshape(YwarpHom / ZwarpHom, [batchSize, H, W])
        # get the integer sampling coordinates
        Xfloor, Xceil = tf.floor(Xwarp), tf.ceil(Xwarp)
        Yfloor, Yceil = tf.floor(Ywarp), tf.ceil(Ywarp)
        XfloorInt, XceilInt = tf.to_int32(Xfloor), tf.to_int32(Xceil)
        YfloorInt, YceilInt = tf.to_int32(Yfloor), tf.to_int32(Yceil)
        ImIdx = tf.tile(tf.reshape(tf.range(batchSize), [batchSize, 1, 1]), [1, H, W])
        ImVecBatch = tf.reshape(ImBatch, [-1, tf.shape(ImBatch)[3]])
        ImVecBatchOutside = tf.concat([ImVecBatch, tf.zeros([1, tf.shape(ImBatch)[3]])], 0)
        idxUL = (ImIdx * H + YfloorInt) * W + XfloorInt
        idxUR = (ImIdx * H + YfloorInt) * W + XceilInt
        idxBL = (ImIdx * H + YceilInt) * W + XfloorInt
        idxBR = (ImIdx * H + YceilInt) * W + XceilInt
        idxOutside = tf.fill([batchSize, H, W], batchSize * H * W)

        def insideIm(Xint, Yint):
            return (Xint >= 0) & (Xint < W) & (Yint >= 0) & (Yint < H)

        idxUL = tf.where(insideIm(XfloorInt, YfloorInt), idxUL, idxOutside)
        idxUR = tf.where(insideIm(XceilInt, YfloorInt), idxUR, idxOutside)
        idxBL = tf.where(insideIm(XfloorInt, YceilInt), idxBL, idxOutside)
        idxBR = tf.where(insideIm(XceilInt, YceilInt), idxBR, idxOutside)
        # bilinear interpolation
        Xratio = tf.reshape(Xwarp - Xfloor, [batchSize, H, W, 1])
        Yratio = tf.reshape(Ywarp - Yfloor, [batchSize, H, W, 1])
        ImUL = tf.to_float(tf.gather(ImVecBatchOutside, idxUL)) * (1 - Xratio) * (1 - Yratio)
        ImUR = tf.to_float(tf.gather(ImVecBatchOutside, idxUR)) * (Xratio) * (1 - Yratio)
        ImBL = tf.to_float(tf.gather(ImVecBatchOutside, idxBL)) * (1 - Xratio) * (Yratio)
        ImBR = tf.to_float(tf.gather(ImVecBatchOutside, idxBR)) * (Xratio) * (Yratio)
        ImWarpBatch = ImUL + ImUR + ImBL + ImBR
        ImWarpBatch = tf.identity(ImWarpBatch, name=name)
    return ImWarpBatch

def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.binary):
    """Matplotlib绘制混淆矩阵图
    parameters
    ----------
        y_truth: 真实的y的值, 1d array
        y_predict: 预测的y的值, 1d array
        cmap: 画混淆矩阵图的配色风格, 使用cm.Blues，更多风格请参考官网
    """
    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)  # 混淆矩阵图
    plt.colorbar()  # 颜色标签
 
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
 
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()  # 显示作图结果
    
def plot_confusion_matrix(y_true, y_pred, labels,name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.4):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=16, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=16, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,fontsize=16)
    plt.yticks(xlocations, labels,fontsize=16)
    plt.ylabel('Index of True Classes',fontsize=16)
    plt.xlabel('Index of Predict Classes',fontsize=16)
    plt.savefig(name, dpi=300)
    plt.show()