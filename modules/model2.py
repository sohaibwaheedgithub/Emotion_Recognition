import tensorflow as tf
from keras_vggface.vggface import VGGFace


class custom_loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        pass

    def call(self, y_true, y_pred):
        y_final = y_true - y_pred
        y_final = tf.reshape(y_final, [-1, 512])
        y_final = tf.reduce_mean(y_final, axis = 0)
        loss = tf.keras.regularizers.L2(1)(y_final)
        return loss


class Model():
    def __init__(self):
        self.ExpNet = None
        self.FaceNet = None

    def build_model(self, input_shape):
        self.FaceNet = VGGFace(include_top = False, input_shape = input_shape)

        # Building Expression Net
        self.ExpNet = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape = input_shape),
            
            tf.keras.layers.Conv2D(64, 3),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(3, 2),
            
            tf.keras.layers.Conv2D(128, 3),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(3, 2),
            
            tf.keras.layers.Conv2D(256, 3),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(3, 2),
            
            tf.keras.layers.Conv2D(512, 3),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(3, 2),
            
            tf.keras.layers.Conv2D(512, 3),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(3, 2),

            tf.keras.layers.Conv1D(512, 3, padding = 'same'),
            tf.keras.layers.Conv2DTranspose(512, 4, padding = 'valid')
        ])



class Model2():
    
        
    

if __name__ == "__main__":
    m = Model()
    m.build_model((256, 256, 3))
    print(m.FaceNet.summary())
    print(m.ExpNet.summary())
    