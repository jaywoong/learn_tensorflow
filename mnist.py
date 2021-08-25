import matplotlib.pyplot as plt
import pandas as pd;
import numpy as np;
import tensorflow as tf;
from tensorflow.python.keras.utils.np_utils import to_categorical

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data Loading
from tensorflow.keras.datasets.mnist import load_data;
(x_train,y_train),(x_test,y_test) = load_data(path='mnist.npz');

print(x_test.shape, y_test.shape)
print(x_train.shape, y_train.shape)
print(x_train)
print(y_train)

# Show image
# import matplotlib.pyplot as plt;
# img = x_train[7, :];
# print(img);
# label = y_train[7];
# plt.figure();
# plt.imshow(img);
# plt.title('%d %d' % (7,label), fontsize=15);
# plt.show();

# 훈련/검증데이터로 분리
# 데이터 정규화
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import MinMaxScaler;

x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.3, random_state=777);
print(x_train.shape, y_train.shape)
print(x_vali.shape, y_vali.shape)

x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_vali = x_vali.reshape(x_vali.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

x_train_scaler = MinMaxScaler().fit_transform(x_train);
x_vali_scaler = MinMaxScaler().fit_transform(x_vali);
x_test_scaler = MinMaxScaler().fit_transform(x_test);

print(x_train_scaler[0,:])

# 데이터를 범주형으로 변환
y_train_cate = to_categorical(y_train)
y_vali_cate = to_categorical(y_vali)
y_test_cate = to_categorical(y_test)

# 신경망 모델 구성
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;

model = Sequential();
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc']);

model.fit(x_train_scaler, y_train_cate,
          epochs=10,
          batch_size=128,
          validation_data=(x_vali_scaler, y_vali_cate),
          verbose=0);

print(model.evaluate(x_test_scaler,y_test_cate));

model.save("mnist.h5")


result = model.predict(x_test_scaler);
print(result.shape)
print(result[0])

arg_results = np.argmax(result[0], axis = -1)

plt.imshow(x_test[0].reshape(28,28));
plt.title(str(arg_results));
plt.show();