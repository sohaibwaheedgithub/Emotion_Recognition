import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image


img_fs = glob.glob(r'C:\Users\sohai\Desktop\sad_face_imgs\*')

for idx, img_f in enumerate(img_fs, 1):
    img =  plt.imread(img_f)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = Image.fromarray(img)
    #img.save(r'datasets\Emotions\sad\images\positive\FACES_sad_{}'.format(idx))
    img.save(r'C:\Users\sohai\Desktop\sad_face_imgs\a.jpg')
    break