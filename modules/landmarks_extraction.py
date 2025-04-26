import cv2
import glob
import constants
import numpy as np
import mediapipe as mp
from utils import normalize_lmks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


if __name__ == "__main__":
  positions = constants.Constants().positions
  face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
  )

  landmarks_list = []
  labels_list = []
  for position_id, position in enumerate(positions):
    print(f"PROCESSING {position.upper()}\n")
    IMAGE_FILES = glob.glob(r'datasets\Face_Position\{}\*.jpg'.format(position))
      
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    class_lmk_list = []
    for idx, file in enumerate(IMAGE_FILES):
      
      image = cv2.imread(file)
      frame_height, frame_width = image.shape[0], image.shape[1]
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(image)  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        continue
      
      annotated_image = image.copy()
      
      for face_landmarks in results.multi_face_landmarks:
        #print('face_landmarks:', face_landmarks)
        face_landmarks_list = []

        for lmk in face_landmarks.landmark:
          face_landmarks_list.append(lmk.x  * frame_width)
          face_landmarks_list.append(lmk.y  * frame_height)
      

        face_landmarks_array = np.array(face_landmarks_list, dtype = np.float16)
        face_landmarks_array = np.expand_dims(face_landmarks_array, axis = 0)
        class_lmk_list.append(face_landmarks_array)


      class_lmk_array = np.concatenate(class_lmk_list, axis = 0)
      class_lmk_array = normalize_lmks(class_lmk_array)
      class_labels_array = np.array([position_id] * class_lmk_array.shape[0], dtype=np.uint8)
 
    landmarks_list.append(class_lmk_array)
    labels_list.append(class_labels_array)


  landmarks_array = np.concatenate(landmarks_list, axis = 0)
  labels_array = np.concatenate(labels_list, axis = 0)



  np.save(r'dataset\landmarks\face_angle_landmarks_tilts_3.npy', landmarks_array)
  np.save(r'dataset\labels\face_angle_labels_tilts_3.npy', labels_array)

