
import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn


from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

# data preparation : run one time
# see https://github.com/abhinavsagar/Grocery-Product-Classification/blob/master/grocery2.ipynb

beans = '../../DL_course_hw/Homework 3/data/BEANS'
cake = '../../DL_course_hw/Homework 3/data/CAKE'
candy = '../../DL_course_hw/Homework 3/data/CANDY'
cereal = '../../DL_course_hw/Homework 3/data/CEREAL'
chips = '../../DL_course_hw/Homework 3/data/CHIPS'
chocolate = '../../DL_course_hw/Homework 3/data/CHOCOLATE'
coffee = '../../DL_course_hw/Homework 3/data/COFFEE'
corn = '../../DL_course_hw/Homework 3/data/CORN'
fish = '../../DL_course_hw/Homework 3/data/FISH'
flour = '../../DL_course_hw/Homework 3/data/FLOUR'
honey = '../../DL_course_hw/Homework 3/data/HONEY'
jam = '../../DL_course_hw/Homework 3/data/JAM'
juice = '../../DL_course_hw/Homework 3/data/JUICE'
milk = '../../DL_course_hw/Homework 3/data/MILK'
nuts = '../../DL_course_hw/Homework 3/data/NUTS'
oil = '../../DL_course_hw/Homework 3/data/OIL'
pasta = '../../DL_course_hw/Homework 3/data/PASTA'
rice = '../../DL_course_hw/Homework 3/data/RICE'
soda = '../../DL_course_hw/Homework 3/data/SODA'
spices = '../../DL_course_hw/Homework 3/data/SPICES'
sugar = '../../DL_course_hw/Homework 3/data/SUGAR'
tea = '../../DL_course_hw/Homework 3/data/TEA'
tomato_sauce = '../../DL_course_hw/Homework 3/data/TOMATO_SAUCE'
vinegar = '../../DL_course_hw/Homework 3/data/VINEGAR'
water = '../../DL_course_hw/Homework 3/data/WATER'

X = []
Z = []
imgsize = 150


def label_assignment(img, label):
        return label


def training_data(label, data_dir):
        for img in tqdm(os.listdir(data_dir)):
                label = label_assignment(img, label)
                path = os.path.join(data_dir, img)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (imgsize, imgsize))

                X.append(np.array(img))
                Z.append(str(label))

training_data('beans',beans)
training_data('cake',cake)
training_data('candy',candy)
training_data('cereal',cereal)
training_data('chips',chips)
training_data('chocolate',chocolate)
training_data('coffee',coffee)
training_data('corn',corn)
training_data('fish',fish)
training_data('flour',flour)
training_data('honey',honey)
training_data('jam',jam)
training_data('juice',juice)
training_data('milk',milk)
training_data('nuts',nuts)
training_data('oil',oil)
training_data('psata',pasta)
training_data('rice',rice)
training_data('soda',soda)
training_data('spices',spices)
training_data('sugar',sugar)
training_data('tea',tea)
training_data('tomato sauce',tomato_sauce)
training_data('vinegar',vinegar)
training_data('water',water)


label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y,25)
X = np.array(X)
X=X/255

np.savez('data/XYZ.npz',name1=X,name2=Y,name3=Z)

#-------------------------
# run from here
#-------------------------

XYZ = np.load('data/XYZ.npz')
X = XYZ['name1']
Y = XYZ['name2']
Z = XYZ['name3']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

augs_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

# ----------------------------------
# can skip
# ----------------------------------

# have a look at 25 samples
fig, ax = plt.subplots(5, 5)
fig.set_size_inches(15, 15)
for i in range(5):
        for j in range(5):
                l = rn.randint(0, len(Z))
                ax[i, j].imshow(X[l])
                ax[i, j].set_title('Grocery: ' + Z[l])

plt.tight_layout()
plt.show()

img_id = 250
cat_generator = augs_gen.flow(x_train[img_id:img_id+1],
 Z[img_id:img_id+1], batch_size=1)
cat = [next(cat_generator) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in cat])
l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
plt.show()

# convert the dummy back to label
label_encoder.fit(Z)
y_label_test = [np.where(y_test[250].astype(int))[0][0]+1]
label_encoder.inverse_transform(y_label_test)

# ----------------------------------
# end skip
# ----------------------------------

# modeling

train_generator = augs_gen.flow(x_train,y_train,batch_size=1)
test_generator = augs_gen.flow(x_test,y_test,batch_size=1)


resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))
x = resnet.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(25, activation="softmax")(x)
model = Model(inputs = resnet.input, outputs = predictions)

for layer in model.layers:
    layer.trainable = False

resnet.summary()

from keras.optimizers import SGD, Adam
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs = 10, batch_size = 64)

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=test_generator,
                              validation_steps=50,
                              verbose=1)

model.save('resnet50.h5')