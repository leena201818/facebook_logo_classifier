# facebook_logo_classifier
Facebook账号头像分类

基本思路：迁移学习

  采用ResNet50、Xception、InceptionV3作为特征提取器，拼接个预训练网络的Bottlenet构成模型输入，添加简单的分类Dense层即可。
  
  本程序采用二分 x = Dense(1, activation='sigmoid')(x)，   loss='binary_crossentropy'，经过简单的设置可以用于图片多分类问题。
  
  比如修改模型最后一层x = Dense(classes, activation='softmax')(x),采用损失loss='categorical_crossentropy'。
