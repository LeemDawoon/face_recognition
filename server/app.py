
import glob
import tqdm
import os
from collections import Counter


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
from utils import infer, get_label, get_max_sim_img

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]
)

FACE_DEFAULT_SHAPE = (224, 224)
DETECTOR = MTCNN()

TRAIN_DF = pd.read_csv('/home/dawoon/proj/face_recognition/labels/train.pre.small.csv')
TRAIN_VEC= np.load('/home/dawoon/proj/face_recognition/outputs/train_vec.small.128.npy')

model = load_model('/home/dawoon/proj/face_recognition/outputs/siamese.densenet169.h5', compile=False)
inp = model.input[0]
base_model = model.layers[3]
INFER_MODEL = Model(inp, base_model(inp))
THRESH = 0.6 

app = Flask(__name__)
app.config['INPUT_DIR'] = os.path.join('static', 'data')
app.config['FACE_DIR'] = os.path.join('static', 'db', 'img_celeba_preprocess' )


@app.route('/')
def index():
    return render_template('index.html' )


@app.route('/upload_face_img', methods=['POST'])
def upload_face_img():
    """ 파일 업로드 처리

    :return:
    """
    # 파일 업로드
    file = request.files['file']
    new_filename = file.filename
    save_file_path = os.path.join(app.config['INPUT_DIR'], new_filename)
    file.save(save_file_path)
    
    pred_vec = infer(INFER_MODEL, DETECTOR, save_file_path, FACE_DEFAULT_SHAPE, is_debug=False) 
    if type(pred_vec) == type(None):
        return render_template('index.html', query_face=save_file_path, sim_face_list=[])
    
    sim = cosine_similarity(TRAIN_VEC, pred_vec)
    sorted_similar = np.argsort(sim*-1, axis=0)
    if np.max(sim) < THRESH: 
        return render_template('index.html', query_face=save_file_path, sim_face_list=[])
    
    # 라벨별 유사도의 평균을 구하고, 1~3위의 라벨 구하기
    TRAIN_DF['sim'] = sim
    TRAIN_DF.loc[TRAIN_DF['sim'] < THRESH, 'sim'] = 0
    df = TRAIN_DF.groupby(['label'])['sim'].mean().reset_index()
    df = df.sort_values(by=['sim'], ascending=False)
    df = df.reset_index()
    sim_label_1 = df['label'].values[0]
    sim_label_2 = df['label'].values[1]
    sim_label_3 = df['label'].values[2]
    
    #  1~3위의 라벨 구하기
    max_sim_img_1, max_sim_1 = get_max_sim_img(sim_label_1, sim, TRAIN_DF)
    max_sim_img_2, max_sim_2 = get_max_sim_img(sim_label_2, sim, TRAIN_DF)
    max_sim_img_3, max_sim_3 = get_max_sim_img(sim_label_3, sim, TRAIN_DF)
    
    face_file_list = [max_sim_img_1, max_sim_img_2, max_sim_img_3]
    face_file_list = [os.path.join(app.config['FACE_DIR'], f) for f in face_file_list] 
    sim_list = [round(max_sim_1, 3), round(max_sim_2, 3), round(max_sim_3, 3)]
    
    return render_template('index.html', query_face=save_file_path, sim_face_list=face_file_list, sim_list=sim_list)


if __name__ == "__main__":              
    host_addr = "0.0.0.0"
    port_num = "8890"
    app.run(host=host_addr, port=port_num, debug=False)
