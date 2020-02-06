import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


def preprocess(detector, img, face_shape, file=None):
    # 얼굴 크롭
    faces = detector.detect_faces(img)
    if len(faces) != 1 :  # detect된 얼굴이 1개가 아닌 경우, 해당 데이터는 처리하지 않는다.
        return None
    
    x, y, w, h = faces[0]['box']
    
    if w < 10 or h < 10:  # detect 된 얼굴 pixel이 너무 작으면, 크롭하지 않는다.
        print('face does not exists', faces[0]['box'], file)
    else:
        # detect된 좌표를 보정한다.
        if x < 0 :
            w += x
            x = 0
        if y < 0 :
            h += y
            y = 0    
        
        # detect 패치의 중심좌표를 기준으로 정사각형으로 크롭한다.
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        margin = max(w, h)
        x = center_x - int(margin/2)
        y = center_y - int(margin/2)
        if x < 0 :
            x = 0
        if y < 0 :
            y = 0  

        img = img[y:y+margin, x:x+margin, :]

    (h, w, _) = img.shape
    if h > w:
        # target shape에 행을 기준으로 맞추기
        new_w = int(w/h*face_shape[0])
        new_h = int(face_shape[0])
    else:
        # target shape에 열을 기준으로 맞추기
        new_w = int(face_shape[1])
        new_h = int(h/w*face_shape[1])
        
    try:
        img = cv2.resize(img, (new_w, new_h))    
    except Exception as e:
        msg = 'None ==>' +  file +' '+ str(img.shape) + ' ' + str(new_w) + ' ' + str(new_h) + ' ' + str(faces[0])
        print(msg)
        return None
    
    new_img = np.zeros(face_shape + (3, ))
    (h, w, _) = new_img.shape
    if face_shape[0] == new_h:
        new_img[:, int(w/2-new_w/2):int(w/2+new_w/2), :] = img
    else:    
        new_img[int(h/2-new_h/2):int(h/2+new_h/2):, :, :] = img
    return new_img




def infer(model, detector, img_path, input_shape, is_debug=True):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    new_img = preprocess(detector, img, input_shape)
    if type(new_img) == type(None):
        return None
    
    if is_debug:
        plt.figure()
        plt.imshow(new_img.astype('uint8'))
        plt.show()
        
    x = np.expand_dims(new_img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    return pred


def get_label(data_df, img_path):
    # 이미지 경로로 부터 이미지 라벨(인물 식별자)을 리턴.
    img_name = os.path.basename(img_path)
    _df = data_df[data_df['image']==img_name]
    if _df.shape[0] == 0:
        return None
    return _df['label'].values[0]

def get_max_sim_img(sim_label, sim, sim_target_df):
    # 이미지 라벨(인물 식별자)중에서 최대 유사도를 갖는  이미지와 유사도 점수 리턴/
    sim_list = list(sim[sim_target_df[sim_target_df['label']==sim_label].index].reshape(-1))
    sim_img_list = sim_target_df[sim_target_df['label']==sim_label]['image'].values.tolist()
    max_sim = 0
    max_sim_img = ''
    for _sim, _sim_img in zip(sim_list, sim_img_list):
        if _sim > max_sim:
            max_sim = _sim
            max_sim_img = _sim_img

    return max_sim_img, max_sim  
