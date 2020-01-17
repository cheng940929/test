import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from  keras.layers import Conv1D, MaxPooling1D,LSTM,TimeDistributed
import pandas as pd
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import keras.backend as k
from keras.optimizers import Adam
from keras.callbacks import TensorBoard




#### import data
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')

x_train,y_train, x_test, y_test = x_train.to_numpy().astype('float32'),y_train.to_numpy().astype('float32'), x_test.to_numpy(), y_test.to_numpy()
smo = SMOTE(random_state=42)
x_train, y_train = smo.fit_sample(x_train,y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train[0])
x_train = np.expand_dims(x_train,axis=2)
x_train = np.expand_dims(x_train,axis=1)

x_test = np.expand_dims(x_test,axis=2)
x_test = np.expand_dims(x_test,axis=1)
#y_train = np.expand_dims(y_train,axis=2)


seq_lenth = x_train.shape[0]
seq_width = x_train.shape[1]
print(x_train.shape)
print(x_test.shape)
##### build up cnn model

model = Sequential()
model.add(TimeDistributed(Conv1D(64,60,activation='relu',padding='same'),batch_input_shape=(None,None,29,1)))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64,60,activation='relu',padding='same')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(256,activation = 'relu')))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(2,activation='softmax'))



def matthews_correlation(y_true,y_pred):
    y_pred_pos = k.round(k.clip(y_pred,0,1))
    y_pred_neg = 1-y_pred_pos
    y_pos = k.round(k.clip(y_true,0,1))
    y_neg = 1-y_pos
    tp = k.sum(y_pos*y_pred_pos)
    tn = k.sum(y_neg*y_pred_neg)
    fp =k.sum(y_neg*y_pred_pos)
    fn = k.sum(y_pos*y_pred_neg)

    numerator = (tp*tn-fp*fn)
    denominator = k.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return numerator/(denominator+k.epsilon())
labmda = 0.9
def punish_loss(y_true,y_pred):
    custom_loss_value = k.binary_crossentropy(y_true,y_pred)+labmda*(1-matthews_correlation(y_true,y_pred))
    return custom_loss_value

adam = Adam(lr = 0.00006)
model.compile(loss=punish_loss,optimizer=adam,metrics=[matthews_correlation])

print(model.summary())
#tbcallback = TensorBoard(log_dir='./Graph',update_freq=1000)
model.fit(x_train,y_train,batch_size=60,epochs=1,validation_split=0.2)
score = model.evaluate(x_test,y_test,batch_size=60)
print(score)
yt = np.argmax(y_test,axis=1)
yp = model.predict(x_test)
yp1 = np.argmax(yp,axis=1)
cf = confusion_matrix(yt, yp1)
cf = cf*100./cf.sum(axis=1, keepdims=True)
print(cf)

#allbacks=[tbcallback]

