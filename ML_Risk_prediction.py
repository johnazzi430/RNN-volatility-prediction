


######## Goal: Using expanded prediction parameters see if ML can yeild useful results
from __future__ import division

import pandas as pd
import numpy as np
import datetime
from pandas_datareader import Options , data , wb

import numpy as np
import theano
import keras
import os
import pickle
import pandas as pd


'''pandas_datareader.__version__'''

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

import random


def context():
      return
### normal normalization
def Normalize(df_data):
    for column in df_data:
        if column == 'Y' :
            pass
        else:
            Ave = np.mean(df_data[column])
            STD = np.std(df_data[column])
            norm =  (df_data[column] - Ave )/ STD
            norms =[]
            for n in norm:
                if pd.isnull(n) == False:
                    norms.append(n)
                else:
                    norms.append(0)
            df_data[column] = norms
    return df_data

def prepareXandY(dataset):
    context.end = len(dataset['Close'])
    exclude = random.sample(xrange(0, context.end - 1), context.test_n)

    X_learn = []
    Y_learn = []
    X_test = []
    Y_test = []

    for row in dataset.iterrows():
        index, data = row

        if any(N == True for N in pd.isnull(data.tolist())):
            continue
        elif any(N == 0 for N in data.tolist()):
            continue
        else:
            Y = data.tolist()[-1]
            if Y > 0 :
                Y2=[1,0]
            else:
                Y2=[0,1]
            X = data.tolist()[:-1]

            if any(z == index for z in exclude) == False:
                X_learn.append(X)  ## trying to predict total return, total return y is in position -2
                Y_learn.append(Y)
            else:
                X_test.append(X)
                Y_test.append(Y)

    context.X_learn = X_learn
    context.Y_learn = Y_learn
    context.X_test = X_test
    context.Y_test = Y_test
    context.exclude = exclude
    return

                ### ----------------- starting parameters ------------------------- ##

def relative_move(data):
    closelast = data[0]
    move =[]
    for close in data:
        move.append((close - closelast) / closelast)
        closelast = close
    return move

def volatility(data , period):
    out = []
    for i in range(len(data) ):
        if i <= period:
            out.append(0)
        else:
            vol = np.std(data[i-period:i])
            out.append(vol)
    return out

def prepare_Y(data, period) :
    out = []
    for i in range(len(data)):
        if i < len(data)-1:
            out.append(data.iloc[i+1])
        else:
            out.append(float('NaN'))
    return out

##################### ------------------------------------------------------------------------ #####################
path = 'C:\Users\John\Desktop\Investing\Machine Learning\Data Dump'

context.test_n = 50
context.lookback = 9000

context.securtities =  'BA'

end = (datetime.date.today() - datetime.timedelta(days=1))
start = end - datetime.timedelta(days=context.lookback)
df_data = data.DataReader(context.securtities, 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
df_sap = data.DataReader('SPY', 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

df_data['move'] = relative_move(df_data['Close'])
df_data['moveS&P'] = relative_move(df_sap['Close'])
df_data['vol_30'] = volatility(df_data['move'] , 30)
df_data['vol_60'] = volatility(df_data['move'] , 60)
df_data['vol_90'] = volatility(df_data['move'] , 90)
df_data['vix_30'] = volatility(df_data['moveS&P']  , 30)
df_data['vix_60'] = volatility(df_data['moveS&P']  , 60)
df_data['vix_90'] = volatility(df_data['moveS&P']  , 90)
df_data = df_data.reset_index()
df_data = df_data.drop('Date', 1)

Y = prepare_Y ( df_data['vol_30'] , 30 )
X = np.asarray(df_data)

split = 400
x_train = X[0:len(Y)-split]
y_train = Y[0:len(Y)-split]
x_test = X[len(Y)-split:len(Y)]
y_test = Y[len(Y)-split:len(Y)]

batch_size = 1
variables = x_train.shape[1]

x_train2 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test2 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, variables, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, variables, 1), stateful=True))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train2, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

model.reset_states()
trainPredict = model.predict(x_train2, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(x_test2, batch_size=batch_size)
score = model.evaluate(x_test2, y_test, batch_size=batch_size)




#prepareXandY(df_data)

