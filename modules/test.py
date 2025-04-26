from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = VGGFace(include_top = True, input_shape = [224, 224, 3])


img = plt.imread(r'C:\Users\sohai\Pictures\Camera Roll\WIN_20221214_11_10_20_Pro.jpg')
img = cv2.resize(img, (224, 224))
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
rgb_img = np.expand_dims(rgb_img, axis = 0)

print(len(model.predict(rgb_img)[0]))




