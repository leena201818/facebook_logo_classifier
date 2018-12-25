from keras.models import load_model
from keras.models import *
from keras.preprocessing.image import *
from keras.applications import *
from keras.layers import *

import numpy as np
import os
import pickle
import config

with open(os.path.join(config.CUR_PATH, config.CLASSES_INDICES), 'rb') as f:
    class_indices = pickle.load(f)

def preval2class(preval):
    for k,v in class_indices.items():
        if int(round(preval)) == v:
            return k

def preprocess_model(MODEL, image_size,lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    return model

model_ResNet50 = preprocess_model(ResNet50,(224,224))
model_InceptionV3 = preprocess_model(InceptionV3, (299, 299),inception_v3.preprocess_input)
model_Xception = preprocess_model(Xception, (299, 299),xception.preprocess_input)

#获得预训练模型输出的特征向量
def get_feature(model,img_file,image_size):
    img = load_img(img_file,target_size=image_size)
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)

    fea = model.predict(img)
    return fea

model = load_model('model.h5')

#预测图片list,返回{图片文件名：类名}
def predict(img_files):
    pre_result = {}
    for f in img_files:
        image_size = (224,224)
        fea_resnet50 = get_feature(model_ResNet50,f,image_size)

        image_size = (299,299)
        fea_inceptionv3 = get_feature(model_InceptionV3,f,image_size)
        fea_xception = get_feature(model_Xception,f,image_size)

        img_fea = np.concatenate([fea_resnet50, fea_xception, fea_inceptionv3], axis=1)

        y_pred = model.predict(img_fea)

        # print(y_pred[:1])
        pre_result[f] = preval2class(y_pred[0][0])

    return pre_result,y_pred

img_dir = os.path.join(config.CUR_PATH,'dataset/training_set/nor')

#预测一个文件夹，返回{图片文件名：类名}
def predict_dir(img_dir):
    # test_path = os.path.join(config.CUR_PATH,'dataset/prediction')
    test_path = img_dir

    img_files = []
    for f in os.listdir(test_path):
        if not os.path.isdir(f):
            img_files.append(os.path.join(test_path,f))

    pre_result,y_pred = predict(img_files)

    return pre_result,y_pred

#预测一个文件夹（至少两个子文件夹）,采用数据发生器预测，返回预测值和真实值
def predict_gen(img_dir):
    gen = ImageDataGenerator()
    image_size = (224, 224)
    train_generator = gen.flow_from_directory(img_dir, image_size, shuffle=False,batch_size=16)
    fea_resnet50 = model_ResNet50.predict_generator(train_generator)

    image_size = (299, 299)
    train_generator = gen.flow_from_directory(img_dir, image_size, shuffle=False, batch_size=16)
    fea_inceptionv3 = model_InceptionV3.predict_generator(train_generator)
    fea_xception = model_Xception.predict_generator(train_generator)

    img_fea = np.concatenate([fea_resnet50, fea_xception, fea_inceptionv3], axis=1)

    y_pred = model.predict(img_fea)
    y_pred = np.reshape(y_pred,(len(y_pred),))
    predict_labels = [preval2class(p_y) for p_y in y_pred]

    pre_result = {}
    # pre_result = dict(zip(train_generator.filenames,predict_labels))  #the same to below
    pre_result = { file:label for file,label in zip(train_generator.filenames,predict_labels)}
    y_true = train_generator.classes

    return pre_result,y_pred,y_true

def main():
    img_dir = os.path.join(config.CUR_PATH,'dataset/test_set')

    _,y_pred,y_true = predict_gen(img_dir)
    y_pred = np.round(np.reshape(y_pred,(len(y_pred),)))
    y_true = np.round(np.reshape(y_true,(len(y_true),)))

    # print(y_pred[:10])
    # print(y_true[:10])
    a = (y_pred == y_true)
    # print(a[:10])
    print('预测正确的样本数量：')
    print(np.sum(a,axis=0))

if __name__ == '__main__':
    main()