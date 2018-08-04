import numpy as np
import scipy.linalg
import os, time
import tensorflow as tf  
import os    
import os
from PIL import Image  

import warp
import numpy as np
import params
import numpy as np
import time, os, sys
import argparse

global best_validation_accuracy
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", metavar="TYPE", help="type of network", default="cSTN" ,choices=["CNN", "STN", "cSTN", "ICSTN"])
parser.add_argument("--group", default="0", help="name for group")
parser.add_argument("--model", default="test", help="name for model instance")
parser.add_argument("--recurN", type=int, default=4, help="number of recurrent transformations (for IC-STN)")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for classification network")
parser.add_argument("--lrST", type=float, default=1e-4, help="learning rate for geometric predictor")
parser.add_argument("--batchSize", type=int, default=100, help="batch size for SGD")
parser.add_argument("--maxIter", type=int, default=150000, help="maximum number of training iterations")
parser.add_argument("--warpType", metavar="WARPTYPE", help="type of warp function on images", default="affine",
                    choices=["translation", "similarity", "affine", "homography"])
parser.add_argument("--resume", type=int, default=0, help="resume from iteration number")
parser.add_argument("--gpu", type=int, default=0, help="ID of GPU device (if there are multiple)")
args = parser.parse_args()

import tensorflow as tf
import data, graph, graphST, warp
from params import Params
momentum = 0.9

print("=======================================================")
print("train.py (training on MNIST)")
print("=======================================================")

# load data
print("loading MNIST dataset...")
trainData, validData, testData = data.loadMNIST("data/MNIST.npz")

# set parameters
print("setting configurations...")
params = Params(args)

best_validation_accuracy = 0.00
# create directories for model output
suffix = args.group
if not os.path.exists("models_{0}".format(suffix)): os.mkdir("models_{0}".format(suffix))
if not os.path.exists("models_{0}/interm".format(suffix)): os.mkdir("models_{0}/interm".format(suffix))
if not os.path.exists("models_{0}/final".format(suffix)): os.mkdir("models_{0}/final".format(suffix))
saveFname = args.model

print("training model {0}...".format(saveFname))
print("------------------------------------------")
print("warpScale: (pert) {0} (trans) {1}".format(params.warpScale["pert"], params.warpScale["trans"]))
print("warpType: {0}".format(params.warpType))
print("batchSize: {0}".format(params.batchSize))
print("GPU device: {0}".format(args.gpu))
print("------------------------------------------")

tf.reset_default_graph()
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True

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