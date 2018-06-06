
# coding: utf-8

# In[3]:


import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[5]:


plt.imshow(x_train[0]);


# In[6]:


x_train[0].min(), x_train[0].max()


# In[7]:


# reshape so channel is last
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[8]:


# scaling
x_train_normalized = x_train/255
x_test_normalized = x_test/255
y_train_vector = keras.utils.to_categorical(y_train, 10)
y_test_vector = keras.utils.to_categorical(y_test, 10)


# In[9]:


x_trn_final = x_train_normalized[:50000]
y_trn_final = y_train_vector[:50000]
x_val_final = x_train_normalized[50000:]
y_val_final = y_train_vector[50000:]


# In[10]:


input_shape = x_trn_final.shape[1:]


# In[11]:


input_shape


# In[18]:


model1 = Sequential()

model1.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=2))
model1.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
model1.add(MaxPooling2D(pool_size=2))

model1.add(Flatten())         
model1.add(Dense(128, activation='relu')) 
model1.add(Dropout(0.2))                  
model1.add(Dense(10, activation='softmax'))

model1.summary()


# In[19]:


model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# verbosity 0, 2, 1
hist = model1.fit(x_trn_final, y_trn_final, batch_size=32, epochs=5,
          validation_data=(x_val_final, y_val_final), verbose=1, shuffle=True)


# In[20]:


model1.evaluate(x_test_normalized, y_test_vector)


# In[21]:


e = hist.epoch
tl = hist.history['loss']
vl = hist.history['val_loss']
ta = hist.history['acc']
va = hist.history['val_acc']


# In[22]:


plt.plot(e, tl, label='train')
plt.plot(e, vl, label='validation')
plt.legend()
plt.show()


# In[23]:


plt.plot(e, ta, label='train')
plt.plot(e, va, label='validation')
plt.legend()
plt.show()

