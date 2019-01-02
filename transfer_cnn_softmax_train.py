#!encoding=utf-8
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

import h5py
import os
import pickle
import config

np.random.seed(2018)

'''
    本模块的作用是生成训练样本的特征向量，并保存到.h5文件中，方便模型fine-tune
    具体方法是：采用迁移学习方法，运行模型推断过程，输出模型全局平均池化层（Bottleneck）作为特征
'''
class train:
    def __init__(self,train_dir,model_file = 'model.h5',epochs = 10,batch_size = 128,validation_split = 0.8,rewrite_feature = False):
        self.train_dir = train_dir
        self.model_file = model_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.class_indices_softmax_file = os.path.join(config.CUR_PATH, config.CLASSES_INDICES_SOFTMAX)
        self.rewrite_gap = rewrite_feature

        n = 0
        for subdir in sorted(os.listdir(train_dir)):
            if os.path.isdir(os.path.join(train_dir,subdir)):
                n += 1
        self.nb_classes = n

    def write_gap(self,MODEL, image_size,MODEL_name,train_dir,lambda_func=None):
        width = image_size[0]
        height = image_size[1]
        input_tensor = Input((height, width, 3))
        x = input_tensor
        if lambda_func:
            x = Lambda(lambda_func)(x)
        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

        # gen = ImageDataGenerator(rescale=1. / 255) #不要预处理，因为Xception等模型已经进行预处理了
        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory(train_dir, image_size, shuffle=False,
                                                  batch_size=16)

        train = model.predict_generator(train_generator)

        gap_file = os.path.join(config.CUR_PATH,"gap_%s.h5"%MODEL_name)
        with h5py.File(gap_file,mode = 'w') as h:
            h.create_dataset("train", data=train)
            print(train.shape)
            h.create_dataset("label", data=train_generator.classes)

        print( train_generator.class_indices )
        return train_generator.class_indices

    #生成特征向量，保存到文件
    def generate_feature(self):
        self.write_gap(ResNet50, (224, 224),'ResNet50',self.train_dir)
        self.write_gap(InceptionV3, (299, 299),'InceptionV3',self.train_dir,inception_v3.preprocess_input)
        return self.write_gap(Xception, (299, 299),'Xception',self.train_dir,xception.preprocess_input)


    #采用多分类时，预测Y必须为onehot向量，classes是分类总数，labels是形状如（samples,）的ndarray,其值表示类序号
    def to_onehot(self,classes,labels):
        r = np.zeros(shape=(len(labels),classes))
        for i in range( len(labels) ):
            r[i,labels[i]] = 1.0
        return r

    # 将多个模型特征向量拼接起来
    def concat_feature(self):
        X_train = []
        for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
            with h5py.File(os.path.join(config.CUR_PATH,filename), 'r') as h:
                X_train.append(np.array(h['train']))
                Y_train = np.array(h['label'])
                Y_train = self.to_onehot(classes=self.nb_classes,labels=Y_train)

        X_train = np.concatenate(X_train, axis=1)       #[ [batch_size,feature],[],[] ] 拼接

        X_train, Y_train = shuffle(X_train, Y_train)

        return X_train,Y_train

    # 微调模型，保存模型到model.h5
    def fit(self,X_train,Y_train):
        weight_file = self.model_file

        def scheduler(epoch):
            if int(epoch % 40) == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
                print("lr changed to {} at epoch{}".format(lr * 0.5, epoch))
                return K.get_value(model.optimizer.lr)
            else:
                print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
                return K.get_value(model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

        input_tensor = Input(X_train.shape[1:])
        x = input_tensor
        # x = Dropout(0.5)(x)
        # x = Dense(128,activation='relu')(x)       #容易过拟合
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input_tensor, x)

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )

        history = model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split,callbacks=[reduce_lr,ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='auto', period=1)])

        model.save(self.model_file)

        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train(self):
        #生成特征向量
        if (self.rewrite_gap):
            self.class_indices = self.generate_feature(self.train_dir)
            with open(self.class_indices_softmax_file, 'wb') as f:
                pickle.dump(self.class_indices, f)

        #拼接特征向量
        X_train, Y_train = self.concat_feature()

        #训练微调，保存模型
        self.fit(X_train, Y_train)

if __name__ == '__main__':
    model_file = os.path.join(config.CUR_PATH,config.MODEL_FILE)
    #目录中必须含有子目录，子目录名称是类标签名，子目录中图片为训练样本
    train_dir = os.path.join(config.CUR_PATH, 'dataset/training_set')

    tr = train(train_dir,model_file,epochs = 1,batch_size = 128,validation_split = 0.2,rewrite_feature=False)

    tr.train()