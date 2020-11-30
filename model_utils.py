import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow_probability import distributions



def build_regressor(hidden):
	input_shape = (len(dt_in), )
	model_in = keras.Input(shape=input_shape, dtype='float32')
	x = model_in
	for h in hidden:
		x = layers.Dense(h, activation='relu')(x)
	model_out = layers.Dense(1, activation='linear')(x)
	model = keras.Model(model_in, model_out)
	return model
