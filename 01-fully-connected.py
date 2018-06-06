
# coding: utf-8

# In[2]:


import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


# shape of data
x_train[0].shape


# In[5]:


# image
img0 = x_train[0]


# In[7]:


plt.imshow(img0)


# In[8]:


img0.min(), img0.max()


# In[9]:


# scaling
(img0/255)[14, :]


# In[10]:


# scaling
x_train_normalized = x_train/255
x_test_normalized = x_test/255
y_train_vector = keras.utils.to_categorical(y_train, 10)
y_test_vector = keras.utils.to_categorical(y_test, 10)


# In[11]:


# getting validation set
x_train_final = x_train_normalized[:50000]
y_train_final = y_train_vector[:50000]
x_validation_final = x_train_normalized[50000:]
y_validation_final = y_train_vector[50000:]


# In[12]:


x_train_final.shape, x_validation_final.shape


# In[13]:


input_shape = x_train_normalized.shape[1:]


# In[14]:


model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[15]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# verbosity 0, 2, 1
hist = model.fit(x_train_final, y_train_final, batch_size=32, epochs=3,
                 validation_data=(x_validation_final, y_validation_final), verbose=1, shuffle=True)


# In[16]:


model.evaluate(x_test_normalized, y_test_vector)


# In[17]:


model.predict(x_test_normalized[0:3])


# In[18]:


for i in range(3):
    plt.imshow(x_test_normalized[i])
    plt.show()


# In[30]:


e = hist.epoch
tl = hist.history['loss']
vl = hist.history['val_loss']
ta = hist.history['acc']
va = hist.history['val_acc']


# In[31]:


plt.plot(e, tl, label='train')
plt.plot(e, vl, label='validation')
plt.legend()
plt.show()


# In[32]:


plt.plot(e, ta, label='train')
plt.plot(e, va, label='validation')
plt.legend()
plt.show()
