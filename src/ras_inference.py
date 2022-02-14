import tensorflow as tf

import numpy as np
import os, shutil, random, sys, glob

from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras import Input, Model

from PIL import Image

import warnings

warnings.filterwarnings("ignore")

print("keras        {}".format(tf.keras.__version__))
print("tensorflow   {}".format(tf.__version__))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#allow GPU memory growth
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#K.set_session(tf.Session(config=config))

ORIGINAL_SIZE = 256
IMG_SIZE = 256
BATCH_SIZE = 196*3
LRATE = 1e-5
JOBID = '135237'
WUP = 5
NUM_GPUS = 3

ROOT = '/tmp/$USERID'

if len(sys.argv) == 2:
    JOBID = sys.argv[1]

def batchPredict(l, o, m):
    #break down the list into chunks
    batches = [l[i:i+BATCH_SIZE] for i in range(0, len(l), BATCH_SIZE)]

    for b in batches:
        inputs = np.ndarray((len(b), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    
        for i in range(len(b)):
            img = Image.open(b[i])
            img = img.resize((IMG_SIZE, IMG_SIZE), resample=1)
            img = np.array(img)[:,:,:3]
            img = img[np.newaxis, :]/255.
            inputs[i] = img
        
        scores = m.predict_on_batch(inputs)
    
        for ff, ss in zip(b, scores):
            print(ff, ss)
            o.write(os.path.basename(ff)[:6] + ' ' + str(ss)[1:-1]+'\n')

    o.close()
#end def batchPredict

with strategy.scope():
    #load Keras model
    base_model = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for l in base_model.layers[:int(len(base_model.layers)*0.5)]:
        l.trainable = False
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    # compile the model
    model.compile(loss = "categorical_crossentropy",
                   optimizer = tf.keras.optimizers.Nadam(lr=LRATE),
                   metrics=["accuracy"])
    model.load_weights('../' + JOBID+'.h5')

print(model.summary())

verbose = 0

imgs = glob.glob(os.path.join(ROOT, 'testing', '*.jpg'))

statFile = open(JOBID+'.txt', 'w+')

batchPredict(imgs, statFile, model)

print('done.')
