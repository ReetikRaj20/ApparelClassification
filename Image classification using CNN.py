#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import requests
import cv2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


# In[2]:


df = pd.read_csv(r'C:\Users\Reetik Raj\Downloads\dress.csv')
df.head(20)


# In[3]:


df.category


# In[4]:


def image_extraction(image_url):
    response = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


# In[5]:


df.count()


# In[6]:


image = image_extraction(df["image_url"][69])
plt.imshow(image)    


# In[7]:


df.category.unique()


# In[8]:


df[df.category =="OTHER"].index


# In[9]:


df.drop(df[df.category =="OTHER"].index, inplace= True)


# In[10]:


df.category.unique()


# In[11]:


df[df.category =="OTHER"].index


# In[12]:


def image_processing(image_url):
    image = urllib.request.urlopen(image_url)
    image_array = np.asarray(bytearray(image.read()), dtype="uint8") 
    image_color = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    image_grayscale = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image_hsv, (0,255,255), (0,255,255))
    if len(np.where(mask != 0)[0]) != 0:
        y1 = min(np.where(mask !=0)[0])
        y2 = max(np.where(mask !=0)[0])
    else:
        y1 = 0
        y2 = len(mask)
        
    if len(np.where(mask != 0)[1]) != 0:
        x1 = min(np.where(mask !=0)[1])
        x2 = max(np.where(mask !=0)[1])
    else:
        x1 = 0
        x2 = len(mask)
    
    image_crop = image_grayscale[y1:y2, x1:x2]
    image_resized = cv2.resize(image_crop, (100, 100))
    image_flatten = image_resized.flatten()
    return image_flatten


# In[13]:


image_processing(df["image_url"][564]).size


# In[14]:


image_list = []
image_processing(df["image_url"][564])
image_list.append(image_processing(df["image_url"][564]))
print(image_list)
image_list.append(image_processing(df["image_url"][565]))
print(image_list)
print(image_list[1])
X = np.array(image_list)
print(X[0])
X.size


# In[15]:


index = []
for i in range(len(df.image_url)):
    index.append(i)
df.index=index


# In[16]:


# Splitting of data in train and test
import pickle
image_list = []
def url_to_array(image_url):
    image_list.append(image_processing(image_url))

for i in range(5):
    url_to_array(df["image_url"][i])

with open('image_array', 'wb') as fp:
        pickle.dump(image_list, fp)


# In[17]:


def read_list():
    # for reading also binary mode is important
    with open('image_array', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
a = read_list()
print(a[0])


# In[18]:


plt.imshow(a[0].reshape(100, 100)), plt.axis('off')


# In[19]:


X = np.load(r'C:\Users\Reetik Raj\Downloads\X.npy')
X[0:3]
X.shape


# In[20]:


np.random.seed(17)
for i in np.random.randint(0, len(X), 5):
    plt.figure()
    plt.imshow(X[i].reshape(100, 100)), plt.axis('off')


# In[21]:


# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()
  
# Encode labels in column 'species'.
Targets = label_encoder.fit_transform(df['category'])
  
df['category'].unique()


# In[22]:


Targets


# In[23]:


one_hot_encoding = to_categorical(Targets, num_classes = 16)
print(one_hot_encoding[0:3])


# In[24]:


len(one_hot_encoding)


# In[25]:


Y = one_hot_encoding


# In[26]:


X[0:3]


# In[27]:


Y[0:3]


# In[31]:


X_test = X[14000:]
Y_test = Y[14000:]

X_train , X_val, Y_train, Y_val = train_test_split(X[:14000], Y[:14000], test_size=0.15, random_state=13)
print(len(X_train))


# In[32]:


img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)


# In[34]:


n_classes = 16
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='softmax'))

learning_rate = 0.001

model.compile(loss = categorical_crossentropy,
              optimizer = Adam(learning_rate),
              metrics=['accuracy'])

model.summary()


# In[35]:


save_at = "fashionDatasetModel.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')


# In[36]:


history = model.fit( X_train, Y_train, 
                    epochs = 5, batch_size = 100, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_val, Y_val))


# In[47]:


np.round(model.predict(X_train))


# In[46]:


image = image_extraction(df["image_url"][2])
plt.imshow(image) 







