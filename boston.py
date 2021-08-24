import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import datasets

b_data = datasets.load_boston()
x_data = b_data.data
y_data = b_data.target
print(x_data.shape)
print(y_data.shape)

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler
x_data_scaled = MinMaxScaler().fit_transform(x_data)
print(x_data)
print(x_data_scaled)

# 학습 데이터셋 분할
from sklearn.model_selection import train_test_split, KFold

x_train, x_test, y_train, y_test = train_test_split(
    x_data_scaled, y_data,
    test_size=0.2,
    shuffle=True,
    random_state=12
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 심층신경망 구축
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(num_input=1):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=num_input))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(num_input=13)
model.summary()

# 모델 훈련
# history1 = model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=0)
# print(model.evaluate(x_test,y_test))

# 교차검증1
# model2 = build_model(num_input=13)
# history2 = model2.fit(x_train, y_train, validation_split=0.25, epochs=300, batch_size=32, verbose=0)
# print(model2.evaluate(x_test,y_test))

# 교차검증2
k = 3
kfold = KFold(n_splits=k, random_state=777, shuffle=True)
mae_list = []

for train_index, val_index in kfold.split(x_train):
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    kmodel = build_model(num_input=13)
    kmodel.fit(x_train_fold, y_train_fold, epochs=300, validation_data=(x_val_fold, y_val_fold))
    result = kmodel.evaluate(x_test, y_test)
    mae_list.append(result)

print(mae_list)