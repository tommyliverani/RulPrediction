import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow_probability import distributions
from sklearn.metrics import r2_score
from rul_utils import split_data

data=pd.read_csv('data/final_rul_data.csv')
tr,ts=split_data(data,0.7)

fig_size=(9, 3)
trmaxrul=7797
def plot_training_history(history, 
        figsize=fig_size, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], label='val. loss')
        plt.legend()
    plt.tight_layout()
    plt.show()


# Load data
input("Press enter to load the data")
path='data'
tr_s = pd.read_csv(path+'/tr.csv')
ts_s = pd.read_csv(path+'/ts.csv')
dt_in = list(tr_s.columns[3:-1])


#build the model
#modello 1 
#nn = build_regressor(hidden=[64,64,64,64])
input("Press enter to build the model")
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
def build_cnn2():
        inp = layers.Input(shape=(2560,2),dtype='float32')
        x = inp
        x = layers.Conv1D(32,16,activation='relu')(x)
        x = layers.MaxPool1D(4)(x)
        x = layers.Conv1D(64,3,activation='relu')(x)
        x = layers.MaxPool1D(4)(x)
        x = layers.Conv1D(64,3,activation='relu')(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(64,3,activation='relu')(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(64,3,activation='relu')(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128,activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16,activation='relu',name='feature')(x)
        out = layers.Dense(1)(x)
        model = keras.Model(inp,out)
        #model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='mse')
        return model



#train the model
input("Press enter to train the model")
#modello 1
#nn = build_regressor(hidden=[64,64,64,64])
#nn.compile(optimizer='Adam', loss='mse')

#modello 2
nn=build_cnn2()

cb = [callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
history = nn.fit(tr_s[dt_in], tr_s['rul'], validation_split=0.2,
                 callbacks=cb, batch_size=32, epochs=10, verbose=1)

nn.save_weights("checkpoint.ckt")

#plot the history
input("Press enter to plot the history")
plot_training_history(history, figsize=fig_size)
trl, vll = history.history["loss"][-1], np.min(history.history["val_loss"])
print(f'Final loss: {trl:.4f} (training), {vll:.4f} (validation)')






