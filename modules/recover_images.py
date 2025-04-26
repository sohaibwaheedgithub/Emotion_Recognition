import numpy as np
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

imgs_array = np.load(r'datasets\Emotions\angry\X_biased_D123B1.npy')
labs_array = np.load(r'datasets\Emotions\angry\y_biased_D123B1.npy')
no = 0
img_no = 1
for lab, img in zip(labs_array, imgs_array):
    if lab == 1:
        no += 1
        if no == img_no:

            img = img.reshape(48, 48, 1)
            img_name = 'angry_{:05d}.jpg'.format(img_no)
            img_path = os.path.join(r'datasets\Emotions\angry\images\positive', img_name)
            tf.keras.utils.save_img(img_path, img)
            img_no += 5
        
        