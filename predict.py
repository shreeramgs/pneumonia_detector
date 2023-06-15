from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
import mxnet as mx 
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
from matplotlib import image
import matplotlib.pyplot as plt
import warnings
import cv2


num_gpus = 0
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deserialized_net = gluon.nn.SymbolBlock.imports(r"E:\desktop\x_ray\resnet-symbol.json", ['data'], r"E:\desktop\x_ray\resnet-0001.params", ctx=ctx)


#mnist_valid = datasets.FashionMNIST(train=False)
sample_data = image.imread(r'E:\desktop\x_ray\predict\NORMAL2-IM-1436-0001.jpeg')
if (len(sample_data.shape))!=3:
    sample_data = cv2.cvtColor(sample_data,cv2.COLOR_GRAY2RGB)    
#X = sample_data
preds = []
#for x in X:
x = transformer(mx.nd.array(sample_data)).expand_dims(axis=0)
pred = deserialized_net(x).argmax(axis=1)
preds.append(pred.astype('int32').asscalar())

_, figs = plt.subplots(1, 1, figsize=(15, 15))
text_labels = ['normal','pneumonia']
display.set_matplotlib_formats('svg')
print(text_labels[preds[0]])
