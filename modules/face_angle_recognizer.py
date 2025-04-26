import os
import constants
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




class Model():
    def __init__(self, classifier_name):
        self.model = None 
        self.n_classes = len(constants.ANGLES)
        self.classifier_name = classifier_name
        self.model_path = os.path.join(r'models\keras_models\{}.h5'.format(self.classifier_name))
        self.earlystopping_cb = EarlyStopping(patience = 5)
        self.modelcheckpoint_cb = ModelCheckpoint(self.model_path)
        #os.mkdir(r'models\openvino_models\{}'.format(self.classifier_name))


    def build_model(self, input_shape):
        self.model = Sequential([
            InputLayer(input_shape = input_shape),
            Dense(20, activation='elu'),
            #Dropout(0.5),
            Dense(10, activation='elu'),
            Dense(self.n_classes, activation='softmax')
        ])
        
        self.model.compile(
            loss = 'sparse_categorical_crossentropy',
            optimizer = Adam(),
            metrics = 'accuracy'
        )



    def train_model(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(
            X_train,
            y_train,
            batch_size = 32,
            epochs = 1000,
            validation_data = (X_valid, y_valid),
            callbacks = [self.modelcheckpoint_cb, self.earlystopping_cb]
        )
        tf.saved_model.save(self.model, r'models\saved_models\{}'.format(self.classifier_name))

    
    def load_model(self):
        model = load_model(self.model_path)
        return model



if __name__ == '__main__':
    import pandas as pd

    model = Model('face_angle_classifier_tilts_3')
    input_shape = [478 * 2,]
    model.build_model(input_shape)
    X, y = np.load(r'dataset\landmarks\face_angle_landmarks_tilts_3.npy'), np.load(r'dataset\labels\face_angle_labels_tilts_3.npy')

    print(X.shape, y.shape)

    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size = 0.25,
        shuffle=True
    )

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)


    model.train_model(X_train, y_train, X_valid, y_valid)

 
    