import os
import datetime
import itertools
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.layers import Input, Dense, LeakyReLU, Concatenate, Lambda, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from mtcnn.mtcnn import MTCNN


# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 16
BATCH_SIZE_VALID = 4
FACE_DEFAULT_SHAPE = (224, 224) #(218, 178) # (128, 96)

train_df = pd.read_csv('./labels/train.pre.small.csv')
valid_df = pd.read_csv('./labels/valid.pre.small.csv')

# 데이터 root 경로
PATH = './data/img_celeba_preprocess'

############################################################
# triplet loss 
############################################################
def triplet_loss(y_true, y_pred, cosine=True, alpha=0.2):
    embedding_size = K.int_shape(y_pred)[-1] // 3
    ind = int(embedding_size * 2)
    a_pred = y_pred[:, :embedding_size]
    p_pred = y_pred[:, embedding_size:ind]
    n_pred = y_pred[:, ind:]
    if cosine:
        positive_distance = 1 - K.sum((a_pred * p_pred), axis=-1)
        negative_distance = 1 - K.sum((a_pred * n_pred), axis=-1)
    else:
        positive_distance = K.sqrt(K.sum(K.square(a_pred - p_pred), axis=-1))
        negative_distance = K.sqrt(K.sum(K.square(a_pred - n_pred), axis=-1))

    loss = K.maximum(0.0, positive_distance - negative_distance + alpha)
    return loss

############################################################
# 데이터 제너레이터
############################################################
class TripletImageGenerator():
    OUTPUT_TYPES = ({'anchor': tf.dtypes.float32, 'positive':tf.dtypes.float32, 'negative': tf.dtypes.float32}, tf.dtypes.float32)
    OUTPUT_SHAPES = ({'anchor': FACE_DEFAULT_SHAPE + (3,), 'positive':FACE_DEFAULT_SHAPE + (3,), 'negative': FACE_DEFAULT_SHAPE + (3,)}, (1,))
    
    def __init__(self, df, preprocess_function, img_shape, is_aug=False):
        self.files = df['image'].values
        self.y = df['label'].values
        self.N = len(self.y)
        self.shape = img_shape
        self.function = preprocess_function
        self.is_aug = is_aug
        self.detector = MTCNN()
        self.train_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.01, 
            height_shift_range=0.01,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1./255
        )
        self.valid_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

    def _load_image(self, file):
        img = image.load_img(os.path.join(PATH, file))
        img = image.img_to_array(img)
        img = self.function(img)
        img = np.expand_dims(img, 0) 
        if self.is_aug:
            it = self.train_img_gen.flow(img, batch_size=1)
        else:    
            it = self.valid_img_gen.flow(img, batch_size=1)
        
        batch = it.next()
        img = batch[0]
        return img
    
    def generator(self):
        for sample_idx in range(self.y.shape[0]):
            ancor_label = self.y[sample_idx]
            ancor_image = self.files[sample_idx]
            pos_idx = np.where(self.y==ancor_label)[0]
            neg_idx = np.where(self.y!=ancor_label)[0]
            neg_image = self.files[np.random.choice(neg_idx)]
            pos_image = self.files[np.random.choice(pos_idx)]
            ANCHOR = self._load_image(ancor_image)
            POSITIVE = self._load_image(pos_image)
            NEGATIVE = self._load_image(neg_image)
            yield ({
                'anchor': ANCHOR,
                'positive': POSITIVE,
                'negative': NEGATIVE,
            }, [0])


##############################
# 데이터 제너레이터 생성
##############################
train_gen = TripletImageGenerator(train_df, preprocess_input, FACE_DEFAULT_SHAPE, is_aug=True)
valid_gen = TripletImageGenerator(valid_df, preprocess_input, FACE_DEFAULT_SHAPE, is_aug=False)
train_dataset = tf.data.Dataset.from_generator(train_gen.generator, 
                                               output_types=TripletImageGenerator.OUTPUT_TYPES,
                                               output_shapes=TripletImageGenerator.OUTPUT_SHAPES
                                              ).repeat().shuffle(buffer_size=BATCH_SIZE*100).batch(batch_size=BATCH_SIZE)

valid_dataset = tf.data.Dataset.from_generator(valid_gen.generator, 
                                               output_types=TripletImageGenerator.OUTPUT_TYPES,
                                               output_shapes=TripletImageGenerator.OUTPUT_SHAPES
                                              ).repeat().batch(batch_size=BATCH_SIZE_VALID)
##############################
#  모델 생성
##############################
base_model = DenseNet169(include_top=False, input_shape = FACE_DEFAULT_SHAPE + (3,))
base_model_output = GlobalAveragePooling2D()(base_model.output)
base_model_output = Dense(512)(base_model_output)
base_model_output = LeakyReLU(alpha=0.1)(base_model_output)
base_model_output = Dense(128)(base_model_output)
normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
base_model_output = normalize(base_model_output)
base_model = Model(base_model.input, base_model_output)
inp_shape = K.int_shape(base_model.input)[1:]
input_a = Input( inp_shape,  name='anchor')
input_p = Input( inp_shape,  name='positive')
input_n = Input( inp_shape,  name='negative')
encoded_anchor = base_model(input_a)
encoded_positive = base_model(input_p)
encoded_negative = base_model(input_n)
merged_vector = Concatenate(axis=-1)([encoded_anchor, encoded_positive, encoded_negative])
model = Model(inputs=[input_a, input_p, input_n], outputs=merged_vector)

for layer in model.layers:
    layer.trainable = True

model.compile(Adam(lr = 0.001, decay=0.5), loss=triplet_loss)

##############################
# 콜백 생성
##############################
log_dir=os.path.join('logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    
# 텐서보드
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir) 
# 모델 저장
checkpoint = ModelCheckpoint('outputs/siamese.densenet169.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

##############################
# 모델 학습
##############################
model.fit(
    x=train_dataset, 
    steps_per_epoch=int(train_df.shape[0]/BATCH_SIZE), 
    epochs=30, 
    validation_data=valid_dataset, 
    validation_steps=int(valid_df.shape[0]/BATCH_SIZE_VALID),
    workers=4, 
    use_multiprocessing=True, 
    verbose=1, 
    callbacks=[tensorboard_callback, checkpoint])