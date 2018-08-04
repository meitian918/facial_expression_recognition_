import numpy as np
import time, os, sys
import argparse

global best_validation_accuracy
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", metavar="TYPE", help="type of network", default="CNN" ,choices=["CNN", "STN", "cSTN", "ICSTN"])
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
parser.add_argument("--gpu", type=int, default=1, help="ID of GPU device (if there are multiple)")
args = parser.parse_args()

import tensorflow as tf
import data, graph, graphST, warp,mydata
from params import Params
momentum = 0.9

print("=======================================================")
print("train.py (training on MNIST)")
print("=======================================================")

# load data
print("loading MNIST dataset...")
trainData, validData, testData = data.loadMNIST("data/MNIST.npz")
testData1 = mydata.loadMNIST("data/MNIST.npz")

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
# build graph
with tf.device(params.GPUdevice):
    # generate training data on the fly
    imageRawBatch = tf.placeholder(tf.float32, shape=[None, 32, 32,None], name="image")
    pInitBatch = data.genPerturbations(params)  #produce raodong
    pInitMtrxBatch = warp.vec2mtrxBatch(pInitBatch, params) #jiang rao dong can shu zhuan hua wei ju zhen 
    ImBatch = data.imageWarpIm(imageRawBatch, pInitMtrxBatch, params, name=None) #cong mei pi tuxiang zhuanhua wei bianhua hou de tuxiang 
    # build network
    if args.type == "CNN":
        outputBatch = graph.buildFullCNN(imageRawBatch, [64, 64, 128, 300], 0.1, params)
    elif args.type == "STN":
        ImWarpBatch,pBatch = graphST.ST_depth1_F(imageRawBatch, pInitBatch,1, [65], 0.01, params)
        outputBatch = graph.buildFullCNN(ImWarpBatch, [64, 64, 128, 300], 0.03, params)
    elif args.type == "cSTN":
        ImWarpBatch, pBatch = graphST.cST_depth4_CCFF(imageRawBatch, pInitBatch, 1, [65, 65, 128], 0.01, params)
        outputBatch = graph.buildCNN(ImWarpBatch, [300], 0.03, params)
    elif args.type == "ICSTN":
        ImWarpBatch, pBatch = graphST.cSTrecur_depth4_CCFF(imageRawBatch, pInitBatch, args.recurN, [65, 65, 128], 0.01,
                                                           params)
        outputBatch = graph.buildCNN(ImWarpBatch, [300], 0.03, params)
    # define loss/optimizer/summaries
    imageSummaries = tf.summary.merge_all()
    labelBatch = tf.placeholder(tf.float32, shape=[None, 5], name="label")
    validerror = tf.placeholder(tf.float32, shape=[1], name="validerror")
    softmaxLoss = tf.nn.softmax_cross_entropy_with_logits(logits=outputBatch, labels=labelBatch)
    loss = tf.reduce_mean(softmaxLoss)
    lossSummary = tf.summary.scalar("training loss", loss)
    learningRate = tf.placeholder(tf.float32, shape=[2])
    trainStep = graph.setOptimizer(loss, learningRate, params)
    softmax = tf.nn.softmax(outputBatch)
    result = tf.argmax(softmax, 1)
    real_result = tf.argmax(labelBatch, 1)
    prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labelBatch, 1))

print("starting backpropagation...")
trainN = len(trainData["image"])
timeStart = time.time()
# define some more summaries
tfSaver, tfSaverInterm, tfSaverFinal = tf.train.Saver(max_to_keep=10), \
                                       tf.train.Saver(max_to_keep=10), \
                                       tf.train.Saver()
validErrorPH = tf.placeholder(tf.float32, shape=[])
validSummary = tf.summary.scalar("test error", validErrorPH)
tfSummaryWriter = tf.summary.FileWriter("summary_{1}/{0}".format(saveFname, suffix))
resumeIterN = 0
maxIterN = 30000
with tf.Session(config=tfConfig) as sess:
    if resumeIterN == 0:
        sess.run(tf.global_variables_initializer())
    else:
        tfSaver.restore(sess, "models_{2}/{0}_it{1}k.ckpt".format(saveFname, resumeIterN // 1000, suffix))
        print("resuming from iteration {0}...".format(resumeIterN))
    tfSummaryWriter.add_graph(sess.graph)
    params.baseLRST = 0.0001
    # training loop
    for i in range(resumeIterN, maxIterN):        
        currLearningRate = params.baseLRST, params.baseLR  # this can be modified to be scheduled learning rates
        randIdx = np.random.randint(trainN, size=[params.batchSize])
        trainBatch = {
            imageRawBatch: trainData["image"][randIdx],
            labelBatch: trainData["label"][randIdx],
            learningRate: currLearningRate
        }
        # run one step
        _, trainBatchLoss, summary = sess.run([trainStep, loss, lossSummary], feed_dict=trainBatch)
        if (i + 1) % 10 == 0:
            tfSummaryWriter.add_summary(summary, i + 1)
        if (i + 1) % 100 == 0:
            print("it. {0}/{1} (lr={5:.2e},{4:.2e}), loss={2:.4f}, time={3:.4f}"
                  .format(i + 1, maxIterN, trainBatchLoss, time.time() - timeStart, currLearningRate[0],
                          currLearningRate[1]))
        if (i + 1) % 5000 == 0:
            # update image summaries
            if imageSummaries is not None:
                summary = sess.run(imageSummaries, feed_dict=trainBatch)
                tfSummaryWriter.add_summary(summary, i + 1)
            # evaluate on validation and test sets
            validAccuracy = data.evaluate(validData, imageRawBatch,labelBatch, prediction, sess, params)
            validError = (1 - validAccuracy) * 100
            print('Iter {} Accuracy: {}'.format(i, validAccuracy))
            if validAccuracy > best_validation_accuracy:
                best_validation_accuracy = validAccuracy
            else:
                 params.baseLR = params.baseLR / 10
                 if params.baseLR <= 0.0001:
                     params.baseLR  = 0.0001
            summary = sess.run(validSummary, feed_dict={validErrorPH: validError})
            tfSummaryWriter.add_summary(summary, i + 1)
            # save model
            savePath = tfSaver.save(sess, "models_{2}/{0}_it{1}k.ckpt".format(saveFname, (i + 1) // 1000, suffix))
            print("model saved: {0}".format(savePath))
        if (i + 1) % 10000 == 0:
            # save intermediate model
            tfSaverInterm.save(sess, "models_{2}/interm/{0}_it{1}k.ckpt".format(saveFname, (i + 1) // 1000, suffix))
    # save final model
    testNormalAccuracy,raw,total,t1 = data.evaluate1(testData1, imageRawBatch,labelBatch, result,real_result, sess, params)
    t2 = data.knnevaluate(testData, imageRawBatch,labelBatch, result,real_result, sess, params)
#    testknnAccuracy = data.knnevaluate(testData, imageRawBatch, labelBatch, result,real_result, sess, params)
    accuracy1,accuracy2,c1,c2,c3,result1,result2,result3,a1,resultBatch = data.cmevaluate(testData, imageRawBatch, labelBatch, result,real_result, softmax ,sess, params)
    annotion = ["Neutral","Happy","Sad","Surprised","Anger"]
    name = ['raw','ROI-CP','ROI-KNN','ROI-KNN2'] 
    data.plot_confusion_matrix(total,raw,annotion,name[0])
    data.plot_confusion_matrix(result3,result1,annotion,name[1])
    data.plot_confusion_matrix(result3,result2,annotion,name[2])
#    data.plot_confusion_matrix(result3,result4,annotion,name[3])
    tfSaverFinal.save(sess, "models_{1}/final/{0}.ckpt".format(saveFname, suffix))
print("======= backpropagation done =======")
