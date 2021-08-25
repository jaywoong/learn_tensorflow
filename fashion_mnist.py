import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets.fashion_mnist import load_data;
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


(x_train,y_train),(x_test,y_test) = load_data();
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

class_name = ['Top','Trouser','Pullover','Dress','Coat',
              'Sendal','Shirt','Sneaker','Bag','Boot'];

plt.imshow(x_train[10])
plt.title(class_name[y_train[10]]);
plt.show()

# 데이터 분리
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train,
                                                    test_size = 0.3, random_state = 777)
# 정규화
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_vali = x_vali.reshape(x_vali.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

x_train_scaler = MinMaxScaler().fit_transform(x_train);
x_vali_scaler = MinMaxScaler().fit_transform(x_vali);
x_test_scaler = MinMaxScaler().fit_transform(x_test);

y_train_cate = to_categorical(y_train)
y_vali_cate = to_categorical(y_vali)
y_test_cate = to_categorical(y_test)

# 신경망 구축
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))# (28, 28) -> .(28 * 28) # 128개의 출력을 가지는 Dense 층을 추가합니다.
model.add(Dense(64, activation = 'relu')) # 64개의 출력을 가지는 Dense 층
model.add(Dense(32, activation = 'relu')) # 32개의 출력을 가지는 Dense 층
model.add(Dense(10, activation = 'softmax')) # 10개의 출력을 가지는 신경망

model.compile(optimizer='adam', # 옵티마이저: Adam
              loss = 'categorical_crossentropy', # 손실 함수: categorical_crossentropy
              metrics=['acc']) # 모니터링 할 평가지표: acc(정확도)

history = model.fit(x_train_scaler, y_train_cate,
                    epochs = 30,
                    batch_size = 128,
                    validation_data = (x_vali_scaler, y_vali_cate), verbose=1)
print(model.evaluate(x_test_scaler,y_test_cate));