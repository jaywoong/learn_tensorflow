import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
titanic = sns.load_dataset("titanic")
titanic.head()
tmp = []
raw_data = titanic.copy()

for i in titanic['sex']:
    if i == 'female':
        tmp.append(1)
    elif i == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)

raw_data['sex'] = tmp

raw_data['survived'] = titanic['survived'].astype('float')
raw_data['pclass'] = titanic['pclass'].astype('float')
raw_data['sex'] = raw_data['sex'].astype('float')
raw_data['sibsp'] = titanic['sibsp'].astype('float')
raw_data['parch'] = titanic['parch'].astype('float')
raw_data['fare'] = titanic['fare'].astype('float')

raw_data = raw_data[raw_data['age'].notnull()]
raw_data = raw_data[raw_data['sibsp'].notnull()]
raw_data = raw_data[raw_data['parch'].notnull()]
raw_data = raw_data[raw_data['fare'].notnull()]

x = raw_data[['pclass','sex','age', 'sibsp', 'parch','fare']]
y = raw_data['survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1, random_state=777)

from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;

model = Sequential()
model.add(Dense(128, input_shape=(6,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense((1), activation='sigmoid'))
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics=['acc']);

model.fit(X_train, y_train,
          epochs=300,
          batch_size= 128,
          validation_data=(X_test, y_test),
          verbose=1);

print(model.evaluate(X_test,y_test));