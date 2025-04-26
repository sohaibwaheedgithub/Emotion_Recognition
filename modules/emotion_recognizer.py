# Importing Libraries

import cv2
import utils
import mediapipe as mp
import tensorflow as tf
from constants import Constants
from openvino.runtime import Core




class Emotion_Recognizer():
    def __init__(self):
        self.label = ''
        self.constants = Constants()
        self.classifiers = {}
        self.inference_engine = Core()
        self.emotion_counter = 0
        self.emotions = self.constants.emotions
        self.classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        
        for emotion in self.emotions:
            compiled_model = self.inference_engine.compile_model(
                    self.inference_engine.read_model(self.emotions[emotion]['xml_filepath']),
                    device_name = 'CPU'
                )
            output_layer = compiled_model.output(0)
            
            self.classifiers[emotion] = {
                'compiled_model': compiled_model,
                'output_layer': output_layer
            }
        
        # For Binary FAD
        self.face_angle_detector = self.inference_engine.compile_model(
            self.inference_engine.read_model(self.constants.face_angle_detector_binary),
            device_name='CPU'
        )

        self.fad_output_layer = self.face_angle_detector.output(0)


        # For Multiclass FAD
        '''self.face_angle_detector = self.inference_engine.compile_model(
            self.inference_engine.read_model(self.constants.face_angle_detector),
            device_name='CPU'
        )

        self.fad_output_layer = self.face_angle_detector.output(0)'''



        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5,
        model_selection=0)


    def detection_preprocessing(self, image, h_max=360):
        h, w, _ = image.shape
        if h > h_max:
            ratio = h_max / h
            w_ = int(w * ratio)
            image = cv2.resize(image, (w_,h_max))
        return image

    
    def resize_face(self, face):
        x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
        return tf.image.resize(x, (48,48))

    
    def recognition_preprocessing(self, faces):
        x = tf.convert_to_tensor([self.resize_face(f) for f in faces])
        return x


    def recognize_emotion(self, image, classifier):
        H, W, _ = image.shape
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if results.detections:
            faces = []
            pos = []
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                
                landmark_list = utils.calc_landmark_list(
                    rgb_image, 
                    detection.location_data
                )
                
                pre_processed_landmark_list = utils.pre_process_landmark(landmark_list)

                pre_processed_landmark_tensor = tf.convert_to_tensor(
                    pre_processed_landmark_list,
                    tf.float32
                )
                
                pre_processed_landmark_tensor = tf.expand_dims(
                    pre_processed_landmark_tensor,
                    axis = 0
                )

                fad_score = self.face_angle_detector([pre_processed_landmark_tensor])[self.fad_output_layer][0][0]
                
                cv2.putText(
                        image,
                        f'Forward Score: {fad_score * 100}',
                        (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                
                if not fad_score >= 0.50:   # 0.30 threshold for face_angle_detector1_binary
                    cv2.putText(
                        image,
                        'Please Face Forward',
                        (10, 90),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )


                    return image
                
                
                '''fad_scores = self.face_angle_detector([pre_processed_landmark_tensor])[self.fad_output_layer][0]
                face_angle_id = tf.argmax(fad_scores)
                face_angle = self.constants.face_angles[face_angle_id]
                
                #face_angle = 'forward' if fad_scores[1] >= 0.10 else face_angle

                if not face_angle == 'forward':
                    cv2.putText(
                        image,
                        f'You are facing {face_angle}',
                        (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        image,
                        f'Forward score is :{fad_scores[1] * 100}',
                        (10, 90),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )



                    return image'''
                    

                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)

                x1 = max(0, x) 
                y1 = max(0, y) 
                x2 = min((x + w), W) 
                y2 = min((y + h), H)

                face = image[y1:y2,x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
                faces.append(face)
                pos.append((x1, y1, x2, y2))
            
        
            x = self.recognition_preprocessing(faces)
            
            
            compiled_model = self.classifiers[classifier]['compiled_model']
            output_layer = self.classifiers[classifier]['output_layer']



            if classifier != 'multiclass':
                score = compiled_model([x])[output_layer][0][0]

                threshold = self.emotions[classifier]['threshold']
                if score > threshold:
                    self.label = classifier.title() + ":  " + str(score)
                    self.emotion_counter += 1
                    label_color = (0, 0, 0)
                    bbox_color = (0, 255, 0)
                else:
                    self.label = "Not_" + classifier.title() + ":  " + str(score)
                    label_color = (0, 0, 0)
                    bbox_color = (255, 255, 0)

            else:
                scores = compiled_model([x])[output_layer][0]

                
                class_id = tf.argmax(scores)
                class_score = scores[class_id]
                self.label = self.classes[class_id].title() + ':  ' + str(class_score)
                

                label_color = self.emotions[self.classes[class_id]]['label_color']
                bbox_color = self.emotions[self.classes[class_id]]['bbox_color']


            
            for i in range(len(faces)):
                cv2.rectangle(
                    image, 
                    (pos[i][0],pos[i][1]),
                    (pos[i][2],pos[i][3]), 
                    bbox_color, 
                    2, 
                    lineType=cv2.LINE_AA
                )
                
                cv2.rectangle(
                    image, 
                    (pos[i][0],pos[i][1]-20),
                    (pos[i][2]+20,pos[i][1]),
                    bbox_color,
                    -1, 
                    lineType=cv2.LINE_AA
                )

                
                cv2.putText(
                    image, 
                    self.label, 
                    (pos[i][0],pos[i][1]-5),
                    0, 
                    0.6, 
                    label_color, 
                    2, 
                    lineType=cv2.LINE_AA
                )

            
        return image

