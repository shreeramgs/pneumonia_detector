import pandas as pd
import numpy as np
from os import listdir
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
#from scipy.misc import imresize, imread
from scipy import misc
import sklearn
from skimage.transform import resize
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tqdm import tqdm

train_path = r'C:\Users\rka_j\Desktop\x_ray\chest_xray\train'
val_path = r'C:\Users\rka_j\Desktop\x_ray\chest_xray\val'
test_path= r'C:\Users\rka_j\Desktop\x_ray\chest_xray\test'


imagesize = 100
def loadBatchImages(path):
    catList = listdir(path)
    loadedImagesTrain = []
    loadedLabelsTrain = []
    for cat in catList:
        if not cat.startswith('.'):
            deepPath = path+'\\'+cat
            imageList = listdir(deepPath)
            for images in tqdm(imageList):
                if not images.startswith('.'):
#                    img = load_img(deepPath +'\\'+ images)
                    img = cv2.imread(deepPath +'\\'+ images)
                    img = cv2.resize(img, (imagesize,imagesize), interpolation = cv2.INTER_AREA)
                    img = img_to_array(img)
                    loadedLabelsTrain.append(cat)
                    loadedImagesTrain.append(img)   
       
    return loadedImagesTrain, loadedLabelsTrain

loadedImagesTrain, loadedLabelsTrain = loadBatchImages(train_path)


encoder = LabelEncoder()
loadedLabelsTrain = np.asarray(loadedLabelsTrain)
encoder.fit(loadedLabelsTrain)
encoded_loadedLabelsTrain = encoder.transform(loadedLabelsTrain)

del loadedLabelsTrain
import gc
gc.collect()

loadedImagesVal, loadedLabelsVal = loadBatchImages(val_path)

loadedLabelsVal = np.asarray(loadedLabelsVal)
encoder.fit(loadedLabelsVal)
encoded_loadedLabelsVal = encoder.transform(loadedLabelsVal)

del loadedLabelsVal
import gc
gc.collect()

X_train=np.array(loadedImagesTrain)
X_train=X_train/255.0

X_test=np.array(loadedImagesVal)
X_test=X_test/255.0



# Helper Functions  Learning Curves and Confusion Matrix

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy',allow_pickle=True)[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')

num_classes = 2
y_trainHot = to_categorical(encoded_loadedLabelsTrain, num_classes = num_classes)
y_testHot = to_categorical(encoded_loadedLabelsVal, num_classes = num_classes)

imageSize =100
pretrained_model_1 = VGG16(include_top=False, input_shape=(imageSize, imageSize, 3))
base_model = pretrained_model_1 # Topless
num_classes = 2
optimizer1 = keras.optimizers.Adam()
# Add top layer
x = base_model.output
x = Conv2D(100, kernel_size = (3,3), padding = 'valid')(x)
x = Flatten()(x)
x = Dropout(0.75)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Train top layer
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer1, 
                  metrics=['accuracy'])
model.summary()


history = model.fit(X_train,y_trainHot, 
                        epochs=10, 
                        batch_size = 32,
                        validation_data=(X_test,y_testHot), 
                        verbose=1,callbacks = [MetricsCheckpoint('logs')])
                        
plotKerasLearningCurve()
plt.show()
plot_learning_curve(history)
plt.show()