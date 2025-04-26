# Importing Libraries

import tensorflow as tf
from functools import partial

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten



class VGGNet():
    def __init__(self, checkpoint_path):
        self.model = Sequential()
        self.checkpoint_path = checkpoint_path
    
    
    def build_model(self, input_shape, lr=1e-3):
        self.model.add(Rescaling(1./255, input_shape=input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=Adam(learning_rate=lr),
                    loss=binary_crossentropy,
                    metrics=['binary_accuracy'])

        """self.model.compile(optimizer=Adam(learning_rate=lr),
                    loss=['sparse_categorical_crossentropy'],
                    metrics=['accuracy'])"""








v = VGGNet('')
v.build_model([48, 48, 1])
print(len(v.model.layers))







