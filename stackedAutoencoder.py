import tensorflow as tf
from keras.backend import tensorflow_backend


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import tensorflow
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=2)
import matplotlib.pyplot as plt
from keras.layers.core import Dense
import csv
import numpy as np
import os
from keras.models import Sequential,load_model
from keras import optimizers
from keras.utils import np_utils


nb_hidden_layers = [15*45, 2048, 1024, 512]


train_flag= True # True: Making Train model and Test, False:Loading model and Test 
sgd  = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
opt=adam        #'sgd' or 'adam'
act='sigmoid'   #'relu' or 'sigmoid'
pep=1   # pretrainig epoch
fep=5   # finetuning epoch
PBS=100	# pretraining mini-batch
FBS=100 # finetuning mini-batch
train_num = 1000 # number of train patches

##-----------------------------------------------------------------------------
# Load Test Images
print "loading test images..."
testdir = 'Test/Joi/'
testfiles = os.listdir(testdir)

for file in testfiles :
    fullpath = testdir + file
    x_test = np.loadtxt(fullpath, delimiter=',')
    #x_test = img_pix / 255

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

##-----------------------------------------------------------------------------
# Load Train Images
x_train = []

print "loading training images..."
traindir = 'Train/Joi/'
files = os.listdir(traindir)

for file in files:
    fullpath = traindir + file
    img_pix = np.loadtxt(fullpath, delimiter=',')
    pickup = [a * int(len(img_pix) / train_num) for a in range(train_num)]
    x_train = img_pix[pickup]
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

##-----------------------------------------------------------------------------
# train_flag == True  : Making AE Network1 and saving Layer-wize pretraining model
# train_flag == False : Making AE Network1  
X_train_tmp = np.copy(x_train)
encoders = []
decoders = []

# Making AE Network 1
for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
    print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
    # Create AE and training
    autoencoder = Sequential()

    encoder = Dense(n_out, input_dim=n_in, activation=act)
    decoder = Dense(n_in, activation=act)

    autoencoder.add(encoder)
    autoencoder.add(decoder)
    autoencoder.compile(optimizer=opt, loss='mse')
    
    # saving Layer-wize pretraining model
    if train_flag==True:
        print("Pretraining...")
        autoencoder.fit(X_train_tmp, X_train_tmp, batch_size=PBS, epochs=pep)

    temp = Sequential()
    temp.add(encoder)
    temp.compile(loss='mse', optimizer=opt)
    X_train_tmp = temp.predict(X_train_tmp)
    encoders.append(encoder)
    decoders.append(decoder)
    '''
    if train_flag==True:
        print "Saving model..."
        svmodelName='PREmodelAE'+i+'Layer.h5'
        autoencoder.save_weights(svmodelName)
        svmodelName='PREmodelTemp'+i+'Layer.h5'
        temp.save_weights(svmodelName)
    '''
##-----------------------------------------------------------------------------
# train_flag == True  : Making AE Network 2 and saving finetuning model
# train_flag == False : Making AE Network 2 and loading fintuning model  

# Making AE Network 2
ae2 = Sequential()
temp2 = Sequential()
for encode in encoders:
    ae2.add(encode)

decoders.reverse()
for decode in decoders:
    ae2.add(decode)

ae2.compile(optimizer=opt, loss='mse')

# saving finetuning model
if train_flag ==True:
    print("Finetuning...")
    hist=ae2.fit(x_train, x_train, batch_size=FBS, epochs=fep)
    
    # making learning loss graph
    print(hist.history)
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    #plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    en='epoch.png'
    plt.savefig(en)
    plt.close()

    temp2.add(encoders[0])
    temp2.add(encoders[1])
    temp2.add(encoders[2]) 
    temp2.compile(loss='mse', optimizer='Adam')
     
   # Save model
    print "Saving model..."
    svmodelName='modelAE.h5'
    ae2.save(svmodelName)
    svmodelName = 'modelTemp.h5'
    temp2.save(svmodelName)

# loading fintuning model
if train_flag == False:
    print "Loding model..."
    svmodelName='modelAE.h5'
    ae2=load_model(svmodelName)
    svmodelName = 'modelTemp.h5'
    temp2=load_model(svmodelName)
##-----------------------------------------------------------------------------
# Test
for file in testfiles :
    fullpath = testdir + file
    x_test = np.loadtxt(fullpath, delimiter=',')

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = temp2.predict(x_test)
    decoded_imgs = ae2.predict(x_test)

    fname =  'TestAEoutput/' + file
    f = open(fname, 'w')  
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(encoded_imgs)
    f.close()

    fname =  'TestAEoutput/AE' + file
    f = open(fname, 'w')
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(decoded_imgs)
    f.close()

    print decoded_imgs.shape

##-----------------------------------------------------------------------------
# Make Image
# use Matplotlib (don't ask)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.swapaxes(x_test[i].reshape(15, 45),0,1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.swapaxes(decoded_imgs[i].reshape(15, 45),0,1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
iname='image.png'
plt.savefig(iname)
plt.close()

