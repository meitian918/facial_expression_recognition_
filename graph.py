import numpy as np
import tensorflow as tf
import time

import data, warp
import tensorflow.contrib.slim as slim


# auxiliary function for creating weight and bias
def createVariable(shape, stddev, zeroInit=False):
    if zeroInit:
        weight = tf.Variable(tf.zeros(shape), name="weight")
        bias = tf.Variable(tf.zeros([shape[-1]]), name="bias")
    else:
        weight = tf.Variable(tf.random_normal(shape, stddev=stddev), name="weight")
        bias = tf.Variable(tf.random_normal([shape[-1]], stddev=stddev), name="bias")
    return weight, bias
   
# build classification perceptron for MNIST
def buildPerceptron(image, dimShape ,stddev, params):
    [conv1dim,conv2dim] = dimShape
    imageVec = 14 * 14 * conv1dim
    with tf.variable_scope("conv1"):
        weight, bias = createVariable([5, 5, 1, conv1dim], stddev)
        conv1 = tf.matmul(image, weight) + bias
        relu1 = tf.nn.relu(conv1)
        maxpool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    vecmaxpool1 = tf.reshape(maxpool1, [-1, imageVec])
    with tf.variable_scope("fc1"):
        weight, bias = createVariable([imageVec, conv2dim], stddev)
        fc3 = tf.matmul(vecmaxpool1, weight) + bias
        fc = tf.nn.dropout(fc3, 0.5)
    with tf.variable_scope("fc2"):
        weight, bias = createVariable([conv2dim, 5], stddev)
        fc3 = tf.matmul(fc, weight) + bias
    return fc3
    
# build classification CNN for MNIST
def buildCNN(image,dimShape, stddev, params):
    [conv1dim] = dimShape
    conv1fcDim = 14 * 14 * conv1dim
    with tf.variable_scope("conv1"):
        weight, bias = createVariable([5, 5, 1, conv1dim], stddev)
        conv1 = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu1 = tf.nn.relu(conv1)
        maxpool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    relu1vec = tf.reshape(maxpool1, [-1, conv1fcDim])
    with tf.variable_scope("fc2"):
        weight, bias = createVariable([conv1fcDim, 5], stddev)
        fc2 = tf.matmul(relu1vec, weight) + bias
    return fc2

def buildFullCNN(image, dimShape, stddev, params):
    [conv1dim, conv2dim, conv3dim, fc4dim] = dimShape
    conv3fcDim = 1 * conv3dim
    with tf.variable_scope("conv1"):
        weight, bias = createVariable([3, 3, 1, conv1dim], stddev)
        conv1 = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu1 = tf.nn.relu(conv1)
        maxpool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    with tf.variable_scope("conv2"):
        weight, bias = createVariable([4, 4, conv1dim, conv2dim], stddev)
        conv2 = tf.nn.conv2d(maxpool1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu2 = tf.nn.relu(conv2)
        maxpool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    with tf.variable_scope("conv3"):
        weight, bias = createVariable([5, 5, conv2dim, conv3dim], stddev)
        conv3 = tf.nn.conv2d(maxpool2, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu3 = tf.nn.relu(conv3)
        maxpool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        relu4vec = tf.reshape(maxpool3, [-1, conv3fcDim])
#        weight, bias = createVariable([3, 3, conv3dim, conv4dim], stddev)
#        conv4 = tf.nn.conv2d(relu3, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
#        relu4 = tf.nn.relu(conv4)
    with tf.variable_scope("fc4"):
        weight, bias = createVariable([conv3fcDim, fc4dim], stddev)
        fc4 = tf.matmul(relu4vec, weight) + bias
        fc = tf.nn.dropout(fc4, 0.5)
    with tf.variable_scope("fc5"):
        weight, bias = createVariable([fc4dim, 5], stddev)
        fc5 = tf.matmul(fc, weight) + bias
    return fc5
    
# make image_summary from image batch
def makeImageSummary(tag, image, params):
    with tf.name_scope("imageSummary"):
        blockSize = params.visBlockSize
        imageSlice = tf.slice(image, [0, 0, 0, 0], [blockSize ** 2, -1, -1, -1])
        imageOne = tf.batch_to_space(imageSlice, crops=[[0, 0], [0, 0]], block_size=blockSize)
        imagePermute = tf.reshape(imageOne, [params.H, blockSize, params.W, blockSize, 1])
        imageTransp = tf.transpose(imagePermute, [1, 0, 3, 2, 4])
        imageBlocks = tf.reshape(imageTransp, [1, params.H * blockSize, params.W * blockSize, 1])
        tf.summary.image(tag, imageBlocks)


# set optimizer for different learning rates
def setOptimizer(loss, learningRate, params):
    varList = tf.global_variables()
    momentum = 0.9
    varListST = [v for v in varList if "ST" in v.name]
    varListOther = [v for v in varList if "ST" not in v.name]
    lrST, lrOther = tf.unstack(learningRate)
    gradients = tf.gradients(loss, varListST + varListOther)
    optimizerOther = tf.train.MomentumOptimizer(lrOther,momentum)
    gradientsOther = gradients[len(varListST):]
    trainStepOther = optimizerOther.apply_gradients(zip(gradientsOther, varListOther))
    if len(varListST) > 0:
        optimizerST = tf.train.MomentumOptimizer(lrST,momentum)
        gradientsST = gradients[:len(varListST)]
        trainStepST = optimizerST.apply_gradients(zip(gradientsST, varListST))
        trainStep = tf.group(trainStepST, trainStepOther)
    else:
        trainStep = trainStepOther
    return trainStep
