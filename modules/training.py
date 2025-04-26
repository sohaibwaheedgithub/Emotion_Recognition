import utils
from model import VGGNet, ResNet34
import tensorflow as tf
from constants import Constants
from data_preparation import Data_Preparation
import numpy as np


if __name__ == "__main__":
    import os
    constants = Constants()
    data_preparation = Data_Preparation()

    """X_path = r'datasets\X_multiclass_unbiased.npy'
    y_path = r'datasets\y_multiclass_unbiased.npy'

    model_path = r'models\keras_models\multiclass_classifier_unbiased.h5'

    X_train, y_train, X_valid, y_valid = data_preparation.split_data(X_path, y_path)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    print(np.unique(y_train, return_counts=True))



    X_train = utils.reshape_images(X_train)
    X_valid = utils.reshape_images(X_valid)





    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30,
                shear_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


    batch_size = constants.batch_size
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_valid, y_valid)

    steps_per_epoch = train_generator.n // train_generator.batch_size
    input_shape = X_train[0].shape
    
    
    vggnet = VGGNet(model_path)
    resnet = ResNet34(model_path) 


    vggnet.build_model(input_shape)
    resnet.build_model(input_shape)

    epochs = 200
    cp = tf.keras.callbacks.ModelCheckpoint(vggnet.checkpoint_path)  
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=20)

    vggnet.model.fit(
        X_train,
        y_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        batch_size = constants.batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[lr, es, cp]
     )"""
    

    emotions = ['angry']


    
    for emotion in emotions:

        print("Training {} classifier \n".format(emotion))
        X_path = os.path.join(r'datasets\Emotions\{}\X_biased_D123B1.npy'.format(emotion))
        y_path = os.path.join(r'datasets\Emotions\{}\y_biased_D123B1.npy'.format(emotion))

        #dataset_path = constants.emotions[emotion]['dataset_path']
        #model_path = constants.emotions[emotion]['model_path']
        model_path = r'models\keras_models\{}_classifier_biased_D123B1.h5'.format(emotion)

        X_train, y_train, X_valid, y_valid = data_preparation.split_data(X_path, y_path)

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_valid, return_counts=True))


        X_train = utils.reshape_images(X_train)
        X_valid = utils.reshape_images(X_valid)

        
        


        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30,
                shear_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True)

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


        batch_size = constants.batch_size
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_valid, y_valid)

        steps_per_epoch = train_generator.n // train_generator.batch_size
        input_shape = X_train[0].shape

        vggnet = vggnet = VGGNet(model_path)
        

        vggnet.build_model(input_shape)

        epochs = 200
        cp = tf.keras.callbacks.ModelCheckpoint(vggnet.checkpoint_path)
        lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=20)

        vggnet.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=[lr, es, cp]
        )
        """
        vggnet.model.fit(
                X_train,
                y_train,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                batch_size = constants.batch_size,
                validation_data=(X_valid, y_valid),
                callbacks=[lr, es, cp]
        )
        """