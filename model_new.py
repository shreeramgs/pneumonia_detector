
################################################################################
# Hyperparameters
# ----------
#
# First, let's import all other necessary libraries.
from comet_ml import Experiment
import mxnet as mx
import numpy as np
import logging
import os, time, shutil
import itertools

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

################################################################################
# We set the hyperparameters as following:
experiment = Experiment(api_key="uVlCssu6MSf2jDhTxV9guqoDI",project_name="model",workspace="yatharth-123")

classes = 2

epochs = 40
lr = 0.001
per_device_batch_size = 64
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

################################################################################
# Things to keep in mind:
#
# 1. ``epochs = 5`` is just for this tutorial with the tiny dataset. please change it to a larger number in your experiments, for instance 40.
# 2. ``per_device_batch_size`` is also set to a small number. In your experiments you can try larger number like 64.
# 3. remember to tune ``num_gpus`` and ``num_workers`` according to your machine.
# 4. A pre-trained model is already in a pretty good status. So we can start with a small ``lr``.
#
# Data Augmentation
# -----------------
#
# In transfer learning, data augmentation can also help.
# We use the following augmentation in training:
#
# 2. Randomly crop the image and resize it to 224x224
# 3. Randomly flip the image horizontally
# 4. Randomly jitter color and add noise
# 5. Transpose the data from height*width*num_channels to num_channels*height*width, and map values from [0, 255] to [0, 1]
# 6. Normalize with the mean and standard deviation from the ImageNet dataset.
#
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),     #Randomly crop the image and resize it to 224x224
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

################################################################################
# With the data augmentation functions, we can define our data loaders:

path = r'C:\Users\rka_j\Desktop\x_ray\chest_xray'
train_data_normal =  r'C:\Users\rka_j\Desktop\x_ray\chest_xray\train'
val_data_normal = r'C:\Users\rka_j\Desktop\x_ray\chest_xray\val'
test_data_normal = r'C:\Users\rka_j\Desktop\x_ray\chest_xray\test'

train_data_normal = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_data_normal).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data_normal = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_data_normal).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

test_data_normal = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_data_normal).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

################################################################################
#
# Note that only ``train_data`` uses ``transform_train``, while
# ``val_data`` and ``test_data`` use ``transform_test`` to produce deterministic
# results for evaluation.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2`` model, which has balanced accuracy and
# computation cost.

model_name = 'ResNet50_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################################
# Here's an illustration of the pre-trained model
# and our newly defined model:
#
# |image-model|
#
# Specifically, we define the new model by::
#
# 1. load the pre-trained model
# 2. re-define the output layer for the new task
# 3. train the network
#
# This is called "fine-tuning", i.e. we have a model trained on another task,
# and we would like to tune it for the dataset we have in hand.
#
# We define a evaluation function for validation and testing.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    experiment.log_figure(figure_name='X-Ray Confusion Matrix', figure=plt)

def create_confusion_matrix(ctx, val_data):
    all_labels = []
    all_outputs = []

    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(
            batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(
            batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]

        for l in label:
            all_labels.extend(l.asnumpy().tolist())

        for o in outputs[0]:
            all_outputs.append(np.argmax(o.asnumpy()))

    cm = confusion_matrix(all_labels, all_outputs)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True,)


def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

################################################################################
# Training Loop
# -------------
#
# Following is the main training loop. It is the same as the loop in
# `CIFAR10 <dive_deep_cifar10.html>`__
# and ImageNet.
#
# .. note::
#
#     Once again, in order to go through the tutorial faster, we are training on a small
#     subset of the original ``MINC-2500`` dataset, and for only 5 epochs. By training on the
#     full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.

lr_counter = 0
num_batch = len(train_data_normal)

for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data_normal):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data_normal, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))
finetune_net.export("resnet", epoch=1)
_, test_acc = test(finetune_net, test_data_normal, ctx)
create_confusion_matrix(ctx=ctx, val_data=test_data_normal)
print('[Finished] Test-acc: %.3f' % (test_acc))

################################################################################
#
# Next
# ----
#
# Now that you have learned to muster the power of transfer
# learning, to learn more about training a model on
# ImageNet, please read `this tutorial <dive_deep_imagenet.html>`__.
#
# The idea of transfer learning is the basis of
# `object detection <../examples_detection/index.html>`_ and
# `semantic segmentation <../examples_segmentation/index.html>`_,
# the next two chapters of our tutorial.
#
# .. |image-minc| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/MINC-2500.png
# .. |image-model| image:: https://zh.gluon.ai/_images/fine-tuning.svg
