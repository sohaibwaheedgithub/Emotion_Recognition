import glob
from PIL import Image
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0)


img_files = glob.glob(r'C:\Users\sohai\Desktop\sad_face_imgs\*')

for idx, img_file in enumerate(img_files, 1):
    img = plt.imread(img_file)
    H, W, _ = img.shape
    try:

        results = face_detection.process(img)
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                #mp_drawing.draw_detection(image, detection)

                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)

                face = img[y1:y2,x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                img = Image.fromarray(face)
                #img.save(os.path.join(r'C:\Users\sohai\Desktop\google_sad', '_g_sad{:04d}.jpg'.format(idx)))
                img.save(r'datasets\Emotions\sad\images\positive\FACES_sad_{}.jpg'.format(idx))
                break
    except:
        continue
    
    
    
            
            
            
    
            
    