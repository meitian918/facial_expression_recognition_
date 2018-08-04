import numpy as np
import tensorflow as tf
import time

import data, warp
from graph import createVariable, makeImageSummary


# build Spatial Transformer (depth=4)
def ST_depth4_CCFF(ImWarp,p,STlayerN, dimShape, stddev, params):
    makeImageSummary("image", ImWarp, params)
    for l in range(STlayerN):
        with tf.name_scope("ST{0}".format(l)):
            [STconv1dim, STconv2dim, STfc3dim] = dimShape
            STconv2fcDim = (params.H - 12) // 2 * (params.W - 12) // 2 * STconv2dim
            with tf.variable_scope("conv1"):
                weight, bias = createVariable([7, 7, 1, STconv1dim], stddev)
                STconv1 = tf.nn.conv2d(ImWarp, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu1 = tf.nn.relu(STconv1)
            with tf.variable_scope("conv2"):
                weight, bias = createVariable([7, 7, STconv1dim, STconv2dim], stddev)
                STconv2 = tf.nn.conv2d(STrelu1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu2 = tf.nn.relu(STconv2)
                STmaxpool2 = tf.nn.max_pool(STrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            STmaxpool2vec = tf.reshape(STmaxpool2, [-1, STconv2fcDim])
            with tf.variable_scope("fc3"):
                weight, bias = createVariable([STconv2fcDim, STfc3dim], stddev)
                STfc3 = tf.matmul(STmaxpool2vec, weight) + bias
                STrelu3 = tf.nn.relu(STfc3)
            with tf.variable_scope("fc4"):
                weight, bias = createVariable([STfc3dim, params.pDim], 0, True)
                STfc4 = tf.matmul(STrelu3, weight) + bias
            warpMtrx = warp.vec2mtrxBatch(STfc4, params)
            ImWarp = data.ImWarpIm(ImWarp, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            p = warp.compose(p, STfc4, params)
    return ImWarp,p

def MYST_depth4_CCFF(ImWarp,STlayerN, dimShape, stddev, params):
    makeImageSummary("image", ImWarp, params)
    for l in range(STlayerN):
        with tf.name_scope("ST{0}".format(l)):
            [STconv1dim, STconv2dim] = dimShape
            STconv2fcDim = 6*6 * STconv2dim
            with tf.variable_scope("conv1"):
                weight, bias = createVariable([3, 3, 1, STconv1dim], stddev)
                STconv1 = tf.nn.conv2d(ImWarp, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu1 = tf.nn.relu(STconv1)
                STmaxpool1 = tf.nn.max_pool(STrelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            with tf.variable_scope("conv2"):
                weight, bias = createVariable([4, 4, STconv1dim, STconv2dim], stddev)
                STconv2 = tf.nn.conv2d(STmaxpool1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu2 = tf.nn.relu(STconv2)
                STmaxpool2 = tf.nn.max_pool(STrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            STmaxpool2vec = tf.reshape(STmaxpool2, [-1, STconv2fcDim])
            with tf.variable_scope("fc4"):
                weight, bias = createVariable([STconv2fcDim, params.pDim], 0, True)
                STfc4 = tf.matmul(STmaxpool2vec, weight) + bias
            warpMtrx = warp.vec2mtrxBatch(STfc4, params)
            ImWarp = data.ImWarpIm(ImWarp, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
    
# build Spatial Transformer (depth=2)
def ST_depth2_CF(ImWarp, p, STlayerN, dimShape, stddev, params):
    makeImageSummary("image", ImWarp, params)
    for l in range(STlayerN):
        with tf.name_scope("ST{0}".format(l)):
            [STconv1dim,STconv2dim] = dimShape
            STconv1fcDim = 36 * STconv2dim
            with tf.variable_scope("conv1"):
                weight, bias = createVariable([6, 6, 1, STconv1dim], stddev)
                STconv1 = tf.nn.conv2d(ImWarp, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu1 = tf.nn.relu(STconv1)
                STmaxpool1 = tf.nn.max_pool(STrelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            with tf.variable_scope("conv2"):
                weight, bias = createVariable([8, 8, STconv1dim, STconv2dim], stddev)
                STconv2 = tf.nn.conv2d(STmaxpool1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu2 = tf.nn.relu(STconv2)
                STmaxpool2 = tf.nn.max_pool(STrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            STrelu1vec = tf.reshape(STmaxpool2, [-1, STconv1fcDim])
            with tf.variable_scope("fc3"):
                weight, bias = createVariable([STconv1fcDim, params.pDim], 0, True)
                STfc3 = tf.matmul(STrelu1vec, weight) + bias
            warpMtrx = warp.vec2mtrxBatch(STfc3, params)
            ImWarp = data.ImWarpIm(ImWarp, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            p = warp.compose(p, STfc3, params)
    return ImWarp, p


# build Spatial Transformer (depth=1)
def ST_depth1_F(ImWarp, p, STlayerN, dimShape, stddev, params):
    makeImageSummary("image", ImWarp, params)
    for l in range(STlayerN):
        with tf.name_scope("ST{0}".format(l)):
            ImWarpVec = tf.reshape(ImWarp, [-1, params.H * params.W])
            with tf.variable_scope("fc1"):
                weight, bias = createVariable([params.H * params.W, params.pDim], 0, True)
                STfc1 = tf.matmul(ImWarpVec, weight) + bias
            warpMtrx = warp.vec2mtrxBatch(STfc1, params)
            ImWarp = data.ImWarpIm(ImWarp, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            p = warp.compose(p, STfc1, params)
    return ImWarp, p


# build compositional Spatial Transformer (depth=4)
def cST_depth4_CCFF(imageInput, p, STlayerN, dimShape, stddev, params):
    ImWarp = imageInput
    k = 1
    for l in range(STlayerN):
        with tf.name_scope("cST{0}".format(l)):
#            warpMtrx = warp.vec2mtrxBatch(p, params)
#            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
#            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            [STconv1dim, STconv2dim, STfc3dim] = dimShape
            STconv2fcDim = (params.H - 12) // 2 * (params.W - 12) // 2 * STconv2dim
            with tf.variable_scope("conv1"):
                weight, bias = createVariable([7, 7, 1, STconv1dim], stddev)
                STconv1 = tf.nn.conv2d(ImWarp, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu1 = tf.nn.relu(STconv1)
            with tf.variable_scope("conv2"):
                weight, bias = createVariable([7, 7, STconv1dim, STconv2dim], stddev)
                STconv2 = tf.nn.conv2d(STrelu1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu2 = tf.nn.relu(STconv2)
                STmaxpool2 = tf.nn.max_pool(STrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            STmaxpool2vec = tf.reshape(STmaxpool2, [-1, STconv2fcDim])
            with tf.variable_scope("fc3"):
                weight, bias = createVariable([STconv2fcDim, STfc3dim], stddev)
                STfc3 = tf.matmul(STmaxpool2vec, weight) + bias
                STrelu3 = tf.nn.relu(STfc3)
            with tf.variable_scope("fc4"):
                weight, bias = createVariable([STfc3dim, params.pDim], 0, True)
                STfc4 = tf.matmul(STrelu3, weight) + bias
            if k == 1 :
                p = STfc4
            else:
                p = warp.compose(p, STfc4, params)
            k = k+1
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    return ImWarp, p


# build compositional Spatial Transformer (depth=2)
def cST_depth2_CF(imageInput, p, STlayerN, dimShape, stddev, params):
    for l in range(STlayerN):
        with tf.name_scope("cST{0}".format(l)):
            warpMtrx = warp.vec2mtrxBatch(p, params)
            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            [STconv1dim] = dimShape
            STconv1fcDim = (params.H - 8) * (params.W - 8) * STconv1dim
            with tf.variable_scope("conv1"):
                weight, bias = createVariable([9, 9, 1, STconv1dim], stddev)
                STconv1 = tf.nn.conv2d(ImWarp, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
                STrelu1 = tf.nn.relu(STconv1)
            STrelu1vec = tf.reshape(STrelu1, [-1, STconv1fcDim])
            with tf.variable_scope("fc2"):
                weight, bias = createVariable([STconv1fcDim, params.pDim], 0, True)
                STfc2 = tf.matmul(STrelu1vec, weight) + bias
            p = warp.compose(p, STfc2, params)
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    return ImWarp, p


# build compositional Spatial Transformer (depth=1)
def cST_depth1_F(imageInput, p, STlayerN, dimShape, stddev, params):
    for l in range(STlayerN):
        with tf.name_scope("cST{0}".format(l)):
            warpMtrx = warp.vec2mtrxBatch(p, params)
            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            imageWarpVec = tf.reshape(ImWarp, [-1, params.H * params.W])
            with tf.variable_scope("fc1"):
                weight, bias = createVariable([params.H * params.W, params.pDim], 0, True)
                STfc1 = tf.matmul(imageWarpVec, weight) + bias
            p = warp.compose(p, STfc1, params)
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    
    return ImWarp, p


# build compositional Spatial Transformer (recurrent, depth=4)
def cSTrecur_depth4_CCFF(imageInput, p, STlayerN, dimShape, stddev, params):
    [STconv1dim, STconv2dim, STfc3dim] = dimShape
    STconv2fcDim = (params.H - 12) // 2 * (params.W - 12) // 2 * STconv2dim
    with tf.name_scope("cSTrecur"):
        with tf.variable_scope("conv1"):
            weight1, bias1 = createVariable([7, 7, 1, STconv1dim], stddev)
        with tf.variable_scope("conv2"):
            weight2, bias2 = createVariable([7, 7, STconv1dim, STconv2dim], stddev)
        with tf.variable_scope("fc3"):
            weight3, bias3 = createVariable([STconv2fcDim, STfc3dim], stddev)
        with tf.variable_scope("fc4"):
            weight4, bias4 = createVariable([STfc3dim, params.pDim], 0, True)
        ImWarp = imageInput
    k = 1
    for l in range(STlayerN):
        with tf.name_scope("cSTrecur{0}".format(l)):
            with tf.variable_scope("conv1"):
                STconv1 = tf.nn.conv2d(ImWarp, weight1, strides=[1, 1, 1, 1], padding="VALID") + bias1
                STrelu1 = tf.nn.relu(STconv1)
            with tf.variable_scope("conv2"):
                STconv2 = tf.nn.conv2d(STrelu1, weight2, strides=[1, 1, 1, 1], padding="VALID") + bias2
                STrelu2 = tf.nn.relu(STconv2)
                STmaxpool2 = tf.nn.max_pool(STrelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            STmaxpool2vec = tf.reshape(STmaxpool2, [-1, STconv2fcDim])
            with tf.variable_scope("fc3"):
                STfc3 = tf.matmul(STmaxpool2vec, weight3) + bias3
                STrelu3 = tf.nn.relu(STfc3)
            with tf.variable_scope("fc4"):
                STfc4 = tf.matmul(STrelu3, weight4) + bias4
            if k == 1:
                p = STfc4
            else:
                p = warp.compose(p, STfc4, params)
            k = k + 1
            warpMtrx = warp.vec2mtrxBatch(p, params)
            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    return ImWarp, p


# build compositional Spatial Transformer (recurrent, depth=2)
def cSTrecur_depth2_CF(imageInput, p, STlayerN, dimShape, stddev, params):
    [STconv1dim] = dimShape
    STconv1fcDim = (params.H - 8) * (params.W - 8) * STconv1dim
    with tf.name_scope("cSTrecur"):
        with tf.variable_scope("conv1"):
            weight1, bias1 = createVariable([9, 9, 1, STconv1dim], stddev)
        with tf.variable_scope("fc2"):
            weight2, bias2 = createVariable([STconv1fcDim, params.pDim], 0, True)
    for l in range(STlayerN):
        with tf.name_scope("cSTrecur{0}".format(l)):
            warpMtrx = warp.vec2mtrxBatch(p, params)
            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            with tf.variable_scope("conv1"):
                STconv1 = tf.nn.conv2d(ImWarp, weight1, strides=[1, 1, 1, 1], padding="VALID") + bias1
                STrelu1 = tf.nn.relu(STconv1)
            STrelu1vec = tf.reshape(STrelu1, [-1, STconv1fcDim])
            with tf.variable_scope("fc2"):
                STfc2 = tf.matmul(STrelu1vec, weight2) + bias2
            p = warp.compose(p, STfc2, params)
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    return ImWarp, p


# build compositional Spatial Transformer (recurrent, depth=1)
def cSTrecur_depth1_F(imageInput, p, STlayerN, dimShape, stddev, params):
    with tf.name_scope("cSTrecur"):
        with tf.variable_scope("fc1"):
            weight, bias = createVariable([params.H * params.W, params.pDim], 0, True)
    for l in range(STlayerN):
        with tf.name_scope("cSTrecur{0}".format(l)):
            warpMtrx = warp.vec2mtrxBatch(p, params)
            ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
            makeImageSummary("imageST{0}".format(l), ImWarp, params)
            imageWarpVec = tf.reshape(ImWarp, [-1, params.H * params.W])
            with tf.variable_scope("fc1"):
                STfc1 = tf.matmul(imageWarpVec, weight) + bias
            p = warp.compose(p, STfc1, params)
    warpMtrx = warp.vec2mtrxBatch(p, params)
    ImWarp = data.imageWarpIm(imageInput, warpMtrx, params)
    makeImageSummary("imageST{0}".format(STlayerN), ImWarp, params)
    return ImWarp, p
