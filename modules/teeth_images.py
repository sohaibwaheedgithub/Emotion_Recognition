import glob
from PIL import Image
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0)


img_files = glob.glob(r'C:\Users\sohai\Desktop\teeths\*')

for img_file in img_files[1:]:
    _, img_name = os.path.split(img_file)
    img = plt.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48))
    img = Image.fromarray(img)
    img.save(os.path.join(r'C:\Users\sohai\Desktop\teeths', img_name))
    
    
    