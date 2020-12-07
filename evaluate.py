import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow_probability import distributions
from sklearn.metrics import r2_score
from rul_utils import split_data

fig_size=(9, 3)

def plot_pred_scatter(y_pred, y_true, figsize=fig_size, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()
    plt.show()

stop=10000

def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=fig_size, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
       # if stddev is not None:
       #     ax.fill_between(range(len(pred)),
       #             pred-stddev, pred+stddev,
       #             alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()
    plt.show()

tr=pd.read_csv('data/tr.csv')
ts=pd.read_csv('data/ts.csv')
dt_in = list(tr.columns[3:-1])

#model 1
def build_regressor(hidden):
    input_shape = (len(dt_in), )
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    model_out = layers.Dense(1, activation='linear')(x)
    model = keras.Model(model_in, model_out)
    return model


#modello 2 
def build_cnn2(self):
        inp = KL.Input(shape=(self.input_shape))
        x = inp
        x = KL.Conv1D(32,16,activation='relu')(x)
        x = KL.MaxPool1D(4)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(4)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
	x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
        x = KL.Flatten()(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(128,activation='relu')(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(self.feature_size,activation='relu',name='feature')(x)
        out = KL.Dense(1)(x)
	model = keras.Model(inp,out)
        #model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='mse')
        return model

#modello 1
#nn=build_regressor(hidden=[64, 64,64,64])

#modello2

nn=build_cnn2()

nn.load_weights("checkpoint.ckt")




tr_in=tr[dt_in]
tr_out=tr['rul']
ts_in=ts[dt_in]
ts_out=ts['rul']


input("Press enter to valuate prediction on training set")
result = nn.predict(tr_in).ravel()
print(f"Result:{r2_score(result,tr_out)}")
plot_pred_scatter(result ,tr_out,fig_size)
plot_rul(result[:stop] ,tr_out[:stop],fig_size)

input("Press enter to valuate prediction on training set")
result = nn.predict(ts_in).ravel()
print(f"Result:{r2_score(result,ts_out)}")
plot_pred_scatter(result ,ts_out,fig_size)
plot_rul(result[:stop] ,ts_out[:stop],fig_size)



