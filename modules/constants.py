import os

class Constants():
    def __init__(self):
        
        self.emotions = {
            'angry': {
                'model_path': r'models\keras_models\angry_classifier_filtered_5.h5',
                'xml_filepath': r'models\openvino_models\angry_classifier_biased_D123B1_2\saved_model.xml',
                'threshold': 0.90,   # 0.40
                'label_color': (0, 0, 0),
                'bbox_color': (0,0,255)
            },
            'happy': {
                'model_path': r'models\keras_models\happy_classifier.h5',
                'xml_filepath': r'models\openvino_models\happy_classifier_biased_D123B1\saved_model.xml',
                'threshold': 0.80,   #0.20
                'label_color': (0, 0, 0),
                'bbox_color': (153,0,153)
            },
            'neutral': {
                'model_path': r'models\keras_models\neutral_classifier.h5',
                'xml_filepath': r'models\openvino_models\neutral_classifier_2\saved_model.xml',
                'threshold': 0.80,
                'label_color': (0, 0, 0),
                'bbox_color': (160,160,160)
            },
            'sad': {
                'model_path': r'models\keras_models\sad_classifier_2.h5',
                'xml_filepath': r'models\openvino_models\sad_classifier_biased_D123B1SR12F\saved_model.xml',
                'threshold': 0.30,
                'label_color': (0, 0, 0),
                'bbox_color': (255,255,0)
            },
            'surprise': {
                'model_path': r'models\keras_models\surprise_classifier_2.h5',
                'xml_filepath': r'models\openvino_models\surprise_classifier_biased_D123B1\saved_model.xml',
                'threshold': 0.50,
                'label_color': (0, 0, 0),
                'bbox_color': (0,255,0)
            },
            'multiclass': {
                'xml_filepath': r'models\openvino_models\multiclass_classifier_unbiased_nn2\saved_model.xml'
            }
        }
        
        self.face_angles = os.listdir('datasets/Face_Angles')
        self.face_angle_detector = r'models\openvino_models\fad_multiclass_model_B\saved_model.xml'
        self.face_angle_detector_binary = r'models\openvino_models\fad_binary_model_B\saved_model.xml'
        self.multiclass_classifier = r'models\openvino_models\multiclass_classifier_unbiased_NDF_D123B12SR12F\saved_model.xml'
        
        self.input_shape = (48, 48, 1)        

        self.target_width = 48
        self.target_height = 48
        self.batch_size = 32

