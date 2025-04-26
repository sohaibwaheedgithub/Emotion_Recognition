import copy
import itertools
import numpy as np
import mediapipe as mp
import tensorflow as tf




def calc_landmark_list(image, detection):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for keypoint in detection.relative_keypoints:
        landmark_x = min(int(keypoint.x * image_width), image_width - 1)
        landmark_y = min(int(keypoint.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point



def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list







mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)


rgb_image: np.ndarray
results = face_detection.process(rgb_image)
for detection in results.detections:

    landmark_list = calc_landmark_list(
        rgb_image, 
        detection.location_data
    )
    
    pre_processed_landmark_list = pre_process_landmark(landmark_list)

    pre_processed_landmark_tensor = tf.convert_to_tensor(
        pre_processed_landmark_list,
        tf.float32
    )
    
    pre_processed_landmark_tensor = tf.expand_dims(
        pre_processed_landmark_tensor,
        axis = 0
    )

    # Input this pre_processed_landmark_tensor to model




